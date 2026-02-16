"""Projection distortion worker: scores neighborhood preservation from embedding to 3D."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)
DISTANCE_WEIGHT_EPS = 1e-6


def _parse_embedding(value) -> np.ndarray:  # type: ignore[no-untyped-def]
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        txt = value.strip()
        if txt.startswith("[") and txt.endswith("]"):
            txt = txt[1:-1]
        return np.fromstring(txt, sep=",", dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _distortion_version(projection_version: str, neighbor_k: int) -> str:
    payload = {
        "projection_version": projection_version,
        "neighbor_k": neighbor_k,
        "metric": "distance_weighted_overlap_v1",
        "embedding_dim": settings.embedding_dim,
        "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"distortion_{digest}"


def _distance_weighted_overlap(
    embedding_neighbor_ids: np.ndarray,
    embedding_neighbor_distances: np.ndarray,
    projected_neighbor_ids: np.ndarray,
    projected_neighbor_distances: np.ndarray,
) -> float:
    """Compute weighted overlap in [0, 1] via weighted-Jaccard of inverse-distance neighbor weights."""
    emb_weights: dict[int, float] = {}
    proj_weights: dict[int, float] = {}

    for idx, dist in zip(embedding_neighbor_ids, embedding_neighbor_distances, strict=False):
        emb_weights[int(idx)] = 1.0 / (DISTANCE_WEIGHT_EPS + float(dist))

    for idx, dist in zip(projected_neighbor_ids, projected_neighbor_distances, strict=False):
        proj_weights[int(idx)] = 1.0 / (DISTANCE_WEIGHT_EPS + float(dist))

    keys = set(emb_weights.keys()).union(proj_weights.keys())
    if not keys:
        return 1.0

    numer = 0.0
    denom = 0.0
    for key in keys:
        a = emb_weights.get(key, 0.0)
        b = proj_weights.get(key, 0.0)
        numer += min(a, b)
        denom += max(a, b)

    if denom <= 0.0:
        return 0.0
    return max(0.0, min(1.0, numer / denom))


@app.task(
    bind=True,
    name="marketmap.workers.projection_distortion_worker.compute_projection_distortion_scores",
    max_retries=2,
)
def compute_projection_distortion_scores(self, neighbor_k: int | None = None) -> dict:  # type: ignore[type-arg]
    """Compute per-node distortion_score = 1 - distance-weighted overlap of local neighbors."""
    started = datetime.now(timezone.utc)
    session = SyncSessionLocal()

    try:
        projection_version = session.execute(
            text(
                """
                SELECT projection_version
                FROM market_projection_3d
                ORDER BY updated_at DESC
                LIMIT 1
                """
            )
        ).scalar()

        if not projection_version:
            return {"status": "success", "markets_scored": 0, "reason": "no_projection"}

        result = session.execute(
            text(
                """
                SELECT p.market_id, me.embedding::text, p.x, p.y, p.z
                FROM market_projection_3d p
                JOIN market_embeddings me ON me.market_id = p.market_id
                JOIN markets m ON m.id = p.market_id
                WHERE p.projection_version = :projection_version
                  AND m.is_active = 1.0
                ORDER BY p.market_id ASC
                """
            ),
            {"projection_version": projection_version},
        )
        rows = result.fetchall()
        if len(rows) < 3:
            return {
                "status": "success",
                "projection_version": projection_version,
                "markets_scored": 0,
                "reason": "not_enough_markets",
            }

        market_ids: list[str] = []
        embedding_rows: list[np.ndarray] = []
        projected_rows: list[tuple[float, float, float]] = []

        for market_id, embedding, x, y, z in rows:
            vec = _parse_embedding(embedding)
            if vec.shape[0] != settings.embedding_dim:
                continue
            if x is None or y is None or z is None:
                continue
            market_ids.append(market_id)
            embedding_rows.append(vec)
            projected_rows.append((float(x), float(y), float(z)))

        n = len(market_ids)
        if n < 3:
            return {
                "status": "success",
                "projection_version": projection_version,
                "markets_scored": 0,
                "reason": "not_enough_valid_rows",
            }

        k_requested = neighbor_k or settings.projection_distortion_k_neighbors
        k = max(1, min(k_requested, n - 1))

        embedding_matrix = np.vstack(embedding_rows).astype(np.float32)
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        embedding_matrix = embedding_matrix / norms
        projected_matrix = np.asarray(projected_rows, dtype=np.float32)

        emb_nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
        emb_nn.fit(embedding_matrix)
        emb_distances, emb_neighbors = emb_nn.kneighbors(return_distance=True)

        proj_nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
        proj_nn.fit(projected_matrix)
        proj_distances, proj_neighbors = proj_nn.kneighbors(return_distance=True)

        distortion_version = _distortion_version(projection_version, k)

        session.execute(
            text(
                """
                DELETE FROM market_projection_distortion
                WHERE projection_version = :projection_version
                """
            ),
            {"projection_version": projection_version},
        )

        upsert_sql = text(
            """
            INSERT INTO market_projection_distortion
                (market_id, projection_version, distortion_version, neighbor_k, distortion_score, updated_at)
            VALUES
                (:market_id, :projection_version, :distortion_version, :neighbor_k, :distortion_score, NOW())
            ON CONFLICT (market_id, projection_version) DO UPDATE SET
                distortion_version = EXCLUDED.distortion_version,
                neighbor_k = EXCLUDED.neighbor_k,
                distortion_score = EXCLUDED.distortion_score,
                updated_at = NOW()
            """
        )

        distortion_scores: list[float] = []
        for i in range(n):
            emb_neighbor_ids = emb_neighbors[i][1:]
            emb_neighbor_distances = emb_distances[i][1:]
            proj_neighbor_ids = proj_neighbors[i][1:]
            proj_neighbor_distances = proj_distances[i][1:]

            overlap_score = _distance_weighted_overlap(
                embedding_neighbor_ids=emb_neighbor_ids,
                embedding_neighbor_distances=emb_neighbor_distances,
                projected_neighbor_ids=proj_neighbor_ids,
                projected_neighbor_distances=proj_neighbor_distances,
            )
            distortion_score = float(max(0.0, min(1.0, 1.0 - overlap_score)))
            distortion_scores.append(distortion_score)

            session.execute(
                upsert_sql,
                {
                    "market_id": market_ids[i],
                    "projection_version": projection_version,
                    "distortion_version": distortion_version,
                    "neighbor_k": k,
                    "distortion_score": distortion_score,
                },
            )

            if i > 0 and i % 5000 == 0:
                session.commit()
                logger.info("Projection distortion upserted %s/%s", i, n)

        session.commit()

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        avg_distortion = float(np.mean(distortion_scores)) if distortion_scores else 0.0
        return {
            "status": "success",
            "projection_version": projection_version,
            "distortion_version": distortion_version,
            "neighbor_k": k,
            "markets_scored": n,
            "avg_distortion_score": avg_distortion,
            "elapsed_seconds": elapsed,
        }
    except Exception as exc:
        session.rollback()
        logger.exception("Projection distortion scoring failed")
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()
