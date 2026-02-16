"""Sync discovery graph from Postgres into Memgraph."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal
from marketmap.services.memgraph import sync_discovery_graph
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)


@app.task(
    bind=True, name="marketmap.workers.memgraph_sync_worker.sync_memgraph_discovery", max_retries=2
)
def sync_memgraph_discovery(self, min_conf: float = 0.3) -> dict:  # type: ignore[type-arg]
    """Export active markets + discovery edges from Postgres into Memgraph."""
    if not settings.memgraph_enabled:
        return {"status": "skipped", "reason": "memgraph_disabled"}

    start = datetime.now(timezone.utc)
    session = SyncSessionLocal()
    try:
        node_rows = (
            session.execute(
                text(
                    """
                SELECT m.id,
                       m.title,
                       m.link,
                       CASE
                         WHEN m.outcome_prices IS NULL THEN NULL
                         ELSE (m.outcome_prices::json->>0)::float
                       END AS prob,
                       m.volume,
                       m.liquidity,
                       m.category,
                       m.close_time,
                       m.event_id,
                       p.x,
                       p.y,
                       p.z,
                       p.projection_version,
                       mc.cluster_id,
                       mpd.distortion_score
                FROM markets m
                LEFT JOIN market_projection_3d p ON p.market_id = m.id
                LEFT JOIN market_clusters mc
                  ON mc.market_id = m.id
                 AND mc.projection_version = p.projection_version
                LEFT JOIN market_projection_distortion mpd
                  ON mpd.market_id = m.id
                 AND mpd.projection_version = p.projection_version
                WHERE m.is_active = 1.0
                ORDER BY m.volume DESC NULLS LAST
                """
                )
            )
            .mappings()
            .all()
        )

        edge_rows = (
            session.execute(
                text(
                    """
                SELECT source_id,
                       target_id,
                       confidence_score,
                       semantic_score,
                       stat_score,
                       logical_score,
                       propagation_score,
                       entity_overlap_score,
                       template_penalty,
                       explanation
                FROM market_edges
                WHERE edge_type = 'discovery'
                  AND confidence_score >= :min_conf
                ORDER BY confidence_score DESC
                """
                ),
                {"min_conf": min_conf},
            )
            .mappings()
            .all()
        )

        payload_nodes = [
            {
                "id": r["id"],
                "label": r["title"] or "",
                "link": r["link"],
                "prob": r["prob"],
                "volume": r["volume"],
                "liquidity": r["liquidity"],
                "category": r["category"],
                "close_time": r["close_time"].isoformat() if r["close_time"] else None,
                "event_id": r["event_id"],
                "x": r["x"],
                "y": r["y"],
                "z": r["z"],
                "projection_version": r["projection_version"],
                "cluster_id": r["cluster_id"],
                "distortion_score": r["distortion_score"],
            }
            for r in node_rows
        ]

        payload_edges = [dict(r) for r in edge_rows]
        sync_stats = sync_discovery_graph(payload_nodes, payload_edges)

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        out = {
            "status": "success",
            "nodes_exported": len(payload_nodes),
            "edges_exported": len(payload_edges),
            "elapsed_seconds": elapsed,
            **sync_stats,
        }
        logger.info("Memgraph sync complete: %s", out)
        return out
    except Exception as exc:
        logger.exception("Memgraph sync failed")
        raise self.retry(exc=exc, countdown=60)
    finally:
        session.close()
