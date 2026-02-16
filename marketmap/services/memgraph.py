"""Memgraph helpers for discovery graph storage and reads."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from neo4j import GraphDatabase

from marketmap.config import settings

logger = logging.getLogger(__name__)


def _auth() -> tuple[str, str] | None:
    if settings.memgraph_username:
        return (settings.memgraph_username, settings.memgraph_password)
    return None


def _driver():
    return GraphDatabase.driver(settings.memgraph_uri, auth=_auth())


def memgraph_is_available() -> bool:
    try:
        with _driver() as driver:
            with driver.session() as session:
                session.run("RETURN 1 AS ok").single()
        return True
    except Exception:
        logger.warning("Memgraph unavailable", exc_info=True)
        return False


def fetch_discovery_graph(min_conf: float = 0.3, include_edges: bool = True) -> dict[str, Any]:
    """Fetch full discovery graph from Memgraph with API-compatible shape."""
    with _driver() as driver:
        with driver.session() as session:
            nodes_rows = session.run(
                """
                MATCH (m:Market)
                RETURN m.id AS id,
                       m.label AS label,
                       m.link AS link,
                       m.prob AS prob,
                       m.volume AS volume,
                       m.liquidity AS liquidity,
                       m.category AS category,
                       m.close_time AS close_time,
                       m.event_id AS event_id,
                       m.x AS x,
                       m.y AS y,
                       m.z AS z,
                       m.projection_version AS projection_version,
                       m.cluster_id AS cluster_id,
                       m.distortion_score AS distortion_score
                """
            )
            nodes = [
                {
                    "id": r["id"],
                    "label": r["label"] or "",
                    "link": r["link"],
                    "prob": r["prob"],
                    "volume": r["volume"],
                    "liquidity": r["liquidity"],
                    "topic": r["category"],
                    "category": r["category"],
                    "close_time": r["close_time"],
                    "event_id": r["event_id"],
                    "x": r["x"],
                    "y": r["y"],
                    "z": r["z"],
                    "projection_version": r["projection_version"],
                    "cluster_id": r["cluster_id"],
                    "distortion_score": r["distortion_score"],
                }
                for r in nodes_rows
            ]

            links: list[dict[str, Any]] = []
            if include_edges:
                edge_rows = session.run(
                    """
                    MATCH (a:Market)-[r:DISCOVERY]->(b:Market)
                    WHERE r.confidence >= $min_conf
                    RETURN a.id AS source,
                           b.id AS target,
                           r.confidence AS confidence,
                           'discovery' AS type,
                           r.semantic_score AS semantic_score,
                           r.stat_score AS stat_score,
                           r.logical_score AS logical_score,
                           r.propagation_score AS propagation_score,
                           r.entity_overlap_score AS entity_overlap_score,
                           r.template_penalty AS template_penalty,
                           r.explanation AS explanation
                    ORDER BY r.confidence DESC
                    """,
                    min_conf=min_conf,
                )
                links = [
                    {
                        "source": r["source"],
                        "target": r["target"],
                        "confidence": r["confidence"],
                        "type": r["type"],
                        "weight": r["confidence"],
                        "semantic_score": r["semantic_score"],
                        "stat_score": r["stat_score"],
                        "logical_score": r["logical_score"],
                        "propagation_score": r["propagation_score"],
                        "entity_overlap_score": r["entity_overlap_score"],
                        "template_penalty": r["template_penalty"],
                        "explanation": r["explanation"],
                    }
                    for r in edge_rows
                ]

    projection_version = next(
        (n.get("projection_version") for n in nodes if n.get("projection_version")), None
    )
    return {
        "nodes": nodes,
        "links": links,
        "meta": {
            "scope": "all",
            "min_conf": min_conf,
            "include_edges": include_edges,
            "node_count": len(nodes),
            "edge_count": len(links),
            "projection_version": projection_version,
            "source": "memgraph",
        },
    }


def sync_discovery_graph(
    nodes: Iterable[dict[str, Any]], edges: Iterable[dict[str, Any]]
) -> dict[str, int]:
    """Replace Memgraph Market + DISCOVERY graph from iterable rows."""
    with _driver() as driver:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

            batch_nodes = list(nodes)
            if batch_nodes:
                session.run(
                    """
                    UNWIND $rows AS row
                    CREATE (m:Market {
                        id: row.id,
                        label: row.label,
                        link: row.link,
                        prob: row.prob,
                        volume: row.volume,
                        liquidity: row.liquidity,
                        category: row.category,
                        close_time: row.close_time,
                        event_id: row.event_id,
                        x: row.x,
                        y: row.y,
                        z: row.z,
                        projection_version: row.projection_version,
                        cluster_id: row.cluster_id,
                        distortion_score: row.distortion_score
                    })
                    """,
                    rows=batch_nodes,
                )
                session.run("CREATE INDEX ON :Market(id)")

            batch_edges = list(edges)
            if batch_edges:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:Market {id: row.source_id}), (b:Market {id: row.target_id})
                    CREATE (a)-[:DISCOVERY {
                        confidence: row.confidence_score,
                        semantic_score: row.semantic_score,
                        stat_score: row.stat_score,
                        logical_score: row.logical_score,
                        propagation_score: row.propagation_score,
                        entity_overlap_score: row.entity_overlap_score,
                        template_penalty: row.template_penalty,
                        explanation: row.explanation
                    }]->(b)
                    """,
                    rows=batch_edges,
                )

    return {"nodes_synced": len(batch_nodes), "edges_synced": len(batch_edges)}
