"""Compatibility clustering worker.

Maps cluster_id to neighborhood_key and persists per projection version,
while local clusters live in markets.local_cluster_id.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal
from marketmap.workers.celery_app import app


def _clustering_version(projection_version: str) -> str:
    payload = {
        "projection_version": projection_version,
        "algorithm": "neighborhood_key",
        "seed": settings.discovery_cluster_seed,
        "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"clusters_{digest}"


@app.task(bind=True, name="marketmap.workers.clustering_worker.compute_discovery_clusters", max_retries=2)
def compute_discovery_clusters(self) -> dict:  # type: ignore[type-arg]
    session = SyncSessionLocal()
    started = datetime.now(timezone.utc)
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
            return {"status": "success", "clusters": 0, "markets_clustered": 0, "reason": "no_projection"}

        rows = session.execute(
            text(
                """
                SELECT m.id,
                       COALESCE(NULLIF(m.neighborhood_key,''), 'misc::unknown') AS cluster_id,
                       COUNT(*) OVER (
                         PARTITION BY COALESCE(NULLIF(m.neighborhood_key,''), 'misc::unknown')
                       ) AS cluster_size
                FROM markets m
                JOIN market_projection_3d p ON p.market_id = m.id
                WHERE m.is_active = 1.0
                  AND p.projection_version = :projection_version
                """
            ),
            {"projection_version": projection_version},
        ).fetchall()

        clustering_version = _clustering_version(projection_version)
        session.execute(
            text("DELETE FROM market_clusters WHERE projection_version = :projection_version"),
            {"projection_version": projection_version},
        )

        upsert_sql = text(
            """
            INSERT INTO market_clusters
                (market_id, projection_version, clustering_version, cluster_id, cluster_size, algorithm, updated_at)
            VALUES
                (:market_id, :projection_version, :clustering_version, :cluster_id, :cluster_size, :algorithm, NOW())
            ON CONFLICT (market_id, projection_version) DO UPDATE SET
                clustering_version = EXCLUDED.clustering_version,
                cluster_id = EXCLUDED.cluster_id,
                cluster_size = EXCLUDED.cluster_size,
                algorithm = EXCLUDED.algorithm,
                updated_at = NOW()
            """
        )

        for i, row in enumerate(rows):
            session.execute(
                upsert_sql,
                {
                    "market_id": row[0],
                    "projection_version": projection_version,
                    "clustering_version": clustering_version,
                    "cluster_id": row[1],
                    "cluster_size": float(row[2]),
                    "algorithm": "neighborhood_key",
                },
            )
            if i > 0 and i % 5000 == 0:
                session.commit()

        session.commit()
        cluster_count = len({row[1] for row in rows})
        largest = max((int(row[2]) for row in rows), default=0)
        return {
            "status": "success",
            "projection_version": projection_version,
            "clustering_version": clustering_version,
            "markets_clustered": len(rows),
            "clusters": cluster_count,
            "largest_cluster": largest,
            "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        }
    except Exception as exc:
        session.rollback()
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()
