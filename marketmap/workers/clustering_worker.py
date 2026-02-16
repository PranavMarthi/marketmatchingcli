"""Discovery clustering worker using Louvain community detection."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

import networkx as nx
from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)


def _clustering_version(projection_version: str) -> str:
    payload = {
        "projection_version": projection_version,
        "algorithm": "louvain",
        "min_conf": settings.discovery_cluster_min_confidence,
        "resolution": settings.discovery_cluster_resolution,
        "seed": settings.discovery_cluster_seed,
        "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"clusters_{digest}"


@app.task(
    bind=True, name="marketmap.workers.clustering_worker.compute_discovery_clusters", max_retries=2
)
def compute_discovery_clusters(self) -> dict:  # type: ignore[type-arg]
    """Compute Louvain clusters from discovery edges and persist by projection version."""
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
            return {"status": "success", "clusters": 0, "markets": 0, "reason": "no_projection"}

        node_rows = session.execute(
            text(
                """
                SELECT p.market_id
                FROM market_projection_3d p
                JOIN markets m ON m.id = p.market_id
                WHERE p.projection_version = :projection_version
                  AND m.is_active = 1.0
                """
            ),
            {"projection_version": projection_version},
        ).fetchall()
        market_ids = [row[0] for row in node_rows]

        if not market_ids:
            return {"status": "success", "clusters": 0, "markets": 0, "reason": "no_markets"}

        edge_rows = session.execute(
            text(
                """
                SELECT source_id, target_id, confidence_score
                FROM market_edges
                WHERE edge_type = 'discovery'
                  AND confidence_score >= :min_conf
                """
            ),
            {"min_conf": settings.discovery_cluster_min_confidence},
        ).fetchall()

        graph = nx.Graph()
        graph.add_nodes_from(market_ids)
        allowed = set(market_ids)
        for source_id, target_id, confidence_score in edge_rows:
            if source_id in allowed and target_id in allowed:
                graph.add_edge(source_id, target_id, weight=float(confidence_score or 0.0))

        logger.info(
            "Computing Louvain communities: %s nodes, %s edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=settings.discovery_cluster_resolution,
            seed=settings.discovery_cluster_seed,
        )

        communities_sorted = sorted(communities, key=len, reverse=True)
        clustering_version = _clustering_version(projection_version)

        rows_to_upsert: list[dict[str, object]] = []
        for idx, community in enumerate(communities_sorted):
            cluster_id = str(idx)
            size = float(len(community))
            for market_id in community:
                rows_to_upsert.append(
                    {
                        "market_id": market_id,
                        "projection_version": projection_version,
                        "clustering_version": clustering_version,
                        "cluster_id": cluster_id,
                        "cluster_size": size,
                        "algorithm": "louvain",
                    }
                )

        session.execute(
            text(
                """
                DELETE FROM market_clusters
                WHERE projection_version = :projection_version
                """
            ),
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

        for i in range(0, len(rows_to_upsert), 5000):
            batch = rows_to_upsert[i : i + 5000]
            for row in batch:
                session.execute(upsert_sql, row)
            session.commit()

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        cluster_sizes = defaultdict(int)
        for row in rows_to_upsert:
            cluster_sizes[str(row["cluster_id"])] += 1

        return {
            "status": "success",
            "projection_version": projection_version,
            "clustering_version": clustering_version,
            "markets_clustered": len(rows_to_upsert),
            "clusters": len(communities_sorted),
            "largest_cluster": max(cluster_sizes.values()) if cluster_sizes else 0,
            "elapsed_seconds": elapsed,
        }
    except Exception as exc:
        session.rollback()
        logger.exception("Discovery clustering failed")
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()
