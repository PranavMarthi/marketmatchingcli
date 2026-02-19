"""Projection worker: computes stable 3D coordinates via two-stage UMAP."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal
from marketmap.projection.hierarchical_projection import run_hierarchical_projection
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)


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


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _embedding_version(
    market_ids: list[str], vectors: list[np.ndarray], model_names: set[str]
) -> str:
    hasher = hashlib.sha1()
    for market_id, vector in zip(market_ids, vectors, strict=False):
        hasher.update(market_id.encode("utf-8"))
        hasher.update(vector.astype(np.float32, copy=False).tobytes())

    payload = {
        "models": sorted(model_names),
        "dim": settings.embedding_dim,
        "count": len(market_ids),
        "content_hash": hasher.hexdigest(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _manifold_version(embedding_version: str) -> str:
    payload = {
        "embedding_version": embedding_version,
        "metric": settings.projection_stage1_metric,
        "n_neighbors": settings.projection_stage1_n_neighbors,
        "min_dist": settings.projection_stage1_min_dist,
        "n_components": settings.projection_stage1_n_components,
        "init": settings.projection_stage1_init,
        "random_state": settings.projection_umap_random_state,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"manifold15_{digest}"


def _projection_version(embedding_version: str, manifold_version: str) -> str:
    payload = {
        "embedding_version": embedding_version,
        "manifold_version": manifold_version,
        "metric": settings.projection_stage2_metric,
        "n_neighbors": settings.projection_stage2_n_neighbors,
        "min_dist": settings.projection_stage2_min_dist,
        "n_components": settings.projection_stage2_n_components,
        "init": settings.projection_stage2_init,
        "random_state": settings.projection_umap_random_state,
        "smoothing_enabled": settings.projection_smoothing_enabled,
        "smoothing_iterations": settings.projection_smoothing_iterations,
        "smoothing_k": settings.projection_smoothing_k_neighbors,
        "smoothing_threshold": settings.projection_smoothing_distortion_threshold,
        "smoothing_alpha": settings.projection_smoothing_alpha,
        "outlier_guard_enabled": settings.projection_outlier_guard_enabled,
        "outlier_quantile": settings.projection_outlier_radius_quantile,
        "outlier_alpha": settings.projection_outlier_guard_alpha,
        "outlier_kmeans_clusters": settings.projection_outlier_kmeans_clusters,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"umap3d_{digest}"


def _cached_projection_is_complete(
    session: Any, projection_version: str, embedding_version: str, market_ids: list[str]
) -> bool:
    if not market_ids:
        return False
    count = session.execute(
        text(
            """
            SELECT COUNT(*)
            FROM market_projection_3d
            WHERE projection_version = :projection_version
              AND embedding_version = :embedding_version
              AND market_id = ANY(:market_ids)
            """
        ),
        {
            "projection_version": projection_version,
            "embedding_version": embedding_version,
            "market_ids": market_ids,
        },
    ).scalar()
    return int(count or 0) == len(market_ids)


def _load_cached_manifold(session: Any, manifold_version: str, market_ids: list[str]) -> np.ndarray | None:
    if not market_ids:
        return None

    rows = session.execute(
        text(
            """
            SELECT market_id, manifold::text
            FROM market_projection_manifold
            WHERE manifold_version = :manifold_version
              AND market_id = ANY(:market_ids)
            """
        ),
        {"manifold_version": manifold_version, "market_ids": market_ids},
    ).fetchall()

    if len(rows) != len(market_ids):
        return None

    by_id = {row[0]: _parse_embedding(row[1]) for row in rows}
    ordered: list[np.ndarray] = []
    for market_id in market_ids:
        vec = by_id.get(market_id)
        if vec is None or vec.shape[0] != settings.projection_stage1_n_components:
            return None
        ordered.append(vec)
    return np.vstack(ordered).astype(np.float32)


def _upsert_cached_manifold(
    session: Any,
    market_ids: list[str],
    manifold_matrix: np.ndarray,
    manifold_version: str,
    embedding_version: str,
) -> None:
    upsert_sql = text(
        """
        INSERT INTO market_projection_manifold
            (market_id, manifold_version, embedding_version, manifold, updated_at)
        VALUES
            (:market_id, :manifold_version, :embedding_version, :manifold, NOW())
        ON CONFLICT (market_id) DO UPDATE SET
            manifold_version = EXCLUDED.manifold_version,
            embedding_version = EXCLUDED.embedding_version,
            manifold = EXCLUDED.manifold,
            updated_at = NOW()
        """
    )

    for idx, market_id in enumerate(market_ids):
        vec = manifold_matrix[idx]
        vec_text = "[" + ",".join(str(float(x)) for x in vec) + "]"
        session.execute(
            upsert_sql,
            {
                "market_id": market_id,
                "manifold_version": manifold_version,
                "embedding_version": embedding_version,
                "manifold": vec_text,
            },
        )

        if idx > 0 and idx % 2000 == 0:
            session.commit()
            logger.info("  Upserted %s/%s manifold vectors", idx, len(market_ids))

    session.commit()


def _estimate_distortion_mask(
    embedding_matrix: np.ndarray,
    projected_matrix: np.ndarray,
    k_neighbors: int,
    threshold: float,
) -> np.ndarray:
    n = embedding_matrix.shape[0]
    if n < 4:
        return np.zeros(n, dtype=bool)

    k = max(2, min(k_neighbors, n - 1))

    emb_nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    emb_nn.fit(embedding_matrix)
    _, emb_neighbors = emb_nn.kneighbors(return_distance=True)

    proj_nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    proj_nn.fit(projected_matrix)
    _, proj_neighbors = proj_nn.kneighbors(return_distance=True)

    distortions = np.zeros(n, dtype=np.float32)
    for i in range(n):
        high_set = set(int(x) for x in emb_neighbors[i][1:])
        low_set = set(int(x) for x in proj_neighbors[i][1:])
        overlap = len(high_set.intersection(low_set)) / float(k)
        distortions[i] = float(max(0.0, min(1.0, 1.0 - overlap)))

    return distortions >= threshold


def _smooth_high_distortion_points(
    embedding_matrix: np.ndarray,
    projected_matrix: np.ndarray,
) -> np.ndarray:
    if not settings.projection_smoothing_enabled:
        return projected_matrix

    n = projected_matrix.shape[0]
    if n < 10:
        return projected_matrix

    k = max(2, min(settings.projection_smoothing_k_neighbors, n - 1))
    emb_nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    emb_nn.fit(embedding_matrix)
    _, emb_neighbors = emb_nn.kneighbors(return_distance=True)

    smoothed = projected_matrix.copy()
    iterations = max(1, settings.projection_smoothing_iterations)
    alpha = float(settings.projection_smoothing_alpha)
    threshold = float(settings.projection_smoothing_distortion_threshold)

    for _ in range(iterations):
        mask = _estimate_distortion_mask(
            embedding_matrix=embedding_matrix,
            projected_matrix=smoothed,
            k_neighbors=k,
            threshold=threshold,
        )
        if not np.any(mask):
            break

        next_matrix = smoothed.copy()
        idxs = np.where(mask)[0]
        for idx in idxs:
            neighbors = emb_neighbors[idx][1:]
            centroid = np.mean(smoothed[neighbors], axis=0)
            next_matrix[idx] = (1.0 - alpha) * smoothed[idx] + alpha * centroid
        smoothed = next_matrix

    return smoothed


def _apply_outlier_guard(projected_matrix: np.ndarray, distortion_mask: np.ndarray) -> np.ndarray:
    if not settings.projection_outlier_guard_enabled:
        return projected_matrix

    if projected_matrix.shape[0] < 50:
        return projected_matrix

    center = np.mean(projected_matrix, axis=0)
    radii = np.linalg.norm(projected_matrix - center, axis=1)
    cutoff = float(np.quantile(radii, settings.projection_outlier_radius_quantile))
    outlier_mask = np.logical_and(distortion_mask, radii >= cutoff)
    if not np.any(outlier_mask):
        return projected_matrix

    kmeans_k = max(4, min(settings.projection_outlier_kmeans_clusters, projected_matrix.shape[0] // 40))
    kmeans = MiniBatchKMeans(
        n_clusters=kmeans_k,
        random_state=settings.projection_umap_random_state,
        n_init="auto",
        batch_size=2048,
    )
    kmeans.fit(projected_matrix)
    centers = kmeans.cluster_centers_

    adjusted = projected_matrix.copy()
    alpha = float(settings.projection_outlier_guard_alpha)
    outlier_idxs = np.where(outlier_mask)[0]
    for idx in outlier_idxs:
        point = adjusted[idx]
        deltas = centers - point
        nearest = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
        target = centers[nearest]
        adjusted[idx] = (1.0 - alpha) * point + alpha * target

    return adjusted


@app.task(bind=True, name="marketmap.workers.projection_worker.compute_market_projection_3d", max_retries=2)
def compute_market_projection_3d(self, batch_limit: int | None = None) -> dict:  # type: ignore[type-arg]
    """Compute 3D coordinates with hierarchical neighborhood projection."""
    logger.info("Starting hierarchical neighborhood projection computation...")
    start = datetime.now(timezone.utc)

    session = SyncSessionLocal()
    try:
        result = run_hierarchical_projection(session=session, batch_limit=batch_limit)
        result["elapsed_seconds"] = (datetime.now(timezone.utc) - start).total_seconds()
        return result
    except Exception as exc:
        session.rollback()
        logger.exception("Projection computation failed")
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()
