"""Projection worker: computes stable 3D UMAP coordinates for market embeddings."""

import hashlib
import json
import logging
from datetime import datetime, timezone

import numpy as np
import umap
from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal
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


def _projection_version(embedding_version: str) -> str:
    payload = {
        "n_components": 3,
        "n_neighbors": settings.projection_umap_n_neighbors,
        "min_dist": settings.projection_umap_min_dist,
        "metric": settings.projection_umap_metric,
        "random_state": settings.projection_umap_random_state,
        "embedding_version": embedding_version,
        "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"umap3d_{digest}"


@app.task(bind=True, name="marketmap.workers.projection_worker.compute_market_projection_3d", max_retries=2)
def compute_market_projection_3d(self, batch_limit: int | None = None) -> dict:  # type: ignore[type-arg]
    """Compute and cache 3D UMAP coordinates for active markets with embeddings."""
    logger.info("Starting 3D projection computation...")
    start = datetime.now(timezone.utc)

    session = SyncSessionLocal()
    try:
        limit = batch_limit or settings.projection_batch_limit
        result = session.execute(
            text(
                """
                SELECT me.market_id, me.embedding::text, COALESCE(me.model_name, 'unknown') AS model_name
                FROM market_embeddings me
                JOIN markets m ON m.id = me.market_id
                WHERE m.is_active = 1.0
                ORDER BY me.market_id ASC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        )
        rows = result.fetchall()
        if not rows:
            return {
                "status": "success",
                "projected": 0,
                "elapsed_seconds": 0,
            }

        market_ids: list[str] = []
        vectors: list[np.ndarray] = []
        model_names: set[str] = set()
        for market_id, embedding, model_name in rows:
            vec = _parse_embedding(embedding)
            if vec.shape[0] != settings.embedding_dim:
                continue
            market_ids.append(market_id)
            vectors.append(vec)
            model_names.add(model_name)

        if len(vectors) < 10:
            return {
                "status": "success",
                "projected": 0,
                "reason": "not_enough_vectors",
                "elapsed_seconds": (datetime.now(timezone.utc) - start).total_seconds(),
            }

        matrix = np.vstack(vectors)
        embedding_version = hashlib.sha1(
            json.dumps({"models": sorted(model_names), "dim": settings.embedding_dim}, sort_keys=True).encode()
        ).hexdigest()[:10]
        projection_version = _projection_version(embedding_version)

        logger.info(f"Running UMAP for {len(market_ids)} markets...")
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=settings.projection_umap_n_neighbors,
            min_dist=settings.projection_umap_min_dist,
            metric=settings.projection_umap_metric,
            random_state=settings.projection_umap_random_state,
            transform_seed=settings.projection_umap_random_state,
        )
        projected = reducer.fit_transform(matrix)

        upsert_sql = text(
            """
            INSERT INTO market_projection_3d
                (market_id, x, y, z, projection_version, embedding_version, updated_at)
            VALUES
                (:market_id, :x, :y, :z, :projection_version, :embedding_version, NOW())
            ON CONFLICT (market_id) DO UPDATE SET
                x = EXCLUDED.x,
                y = EXCLUDED.y,
                z = EXCLUDED.z,
                projection_version = EXCLUDED.projection_version,
                embedding_version = EXCLUDED.embedding_version,
                updated_at = NOW()
            """
        )

        for idx, market_id in enumerate(market_ids):
            coords = projected[idx]
            session.execute(
                upsert_sql,
                {
                    "market_id": market_id,
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                    "z": float(coords[2]),
                    "projection_version": projection_version,
                    "embedding_version": embedding_version,
                },
            )

            if idx > 0 and idx % 1000 == 0:
                session.commit()
                logger.info(f"  Upserted {idx}/{len(market_ids)} projections")

        session.commit()

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(
            f"Projection complete: {len(market_ids)} markets in {elapsed:.1f}s "
            f"(projection_version={projection_version})"
        )
        return {
            "status": "success",
            "projected": len(market_ids),
            "projection_version": projection_version,
            "embedding_version": embedding_version,
            "elapsed_seconds": elapsed,
        }
    except Exception as exc:
        session.rollback()
        logger.exception("Projection computation failed")
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()
