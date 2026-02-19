"""FastAPI application for MarketMap API."""

import hashlib
import json
import logging
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from marketmap.api.schemas import (
    EventContextResponse,
    EventRelatedNode,
    GraphLink,
    GraphResponse,
    HealthResponse,
    MarketDetail,
    MarketNode,
    MarketPriceHistory,
    NeighborhoodNode,
    PricePoint,
)
from marketmap.config import settings
from marketmap.models.database import get_async_session
from marketmap.services.memgraph import fetch_discovery_graph, memgraph_is_available

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MarketMap API",
    description="Confidence-Gated Dependency Graph for Prediction Markets",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helpers ---

# Order:
# id, title, link, outcome_prices, volume, liquidity, category, close_time,
# event_id, description, is_active, x, y, z, projection_version
MARKET_COLS = (
    "m.id, m.title, m.link, m.outcome_prices, m.volume, m.liquidity, "
    "m.category, m.close_time, m.event_id, m.description, m.is_active, "
    "COALESCE(m.global_x, p.x), COALESCE(m.global_y, p.y), COALESCE(m.global_z, p.z), p.projection_version, "
    "m.neighborhood_key, m.neighborhood_label, m.local_cluster_id, "
    "m.local_x, m.local_y, m.local_z, m.global_x, m.global_y, m.global_z, "
    "m.local_distortion, m.stitch_distortion"
)
MARKET_JOIN = "LEFT JOIN market_projection_3d p ON p.market_id = m.id"


def _get_redis_client() -> Redis | None:
    try:
        return Redis.from_url(settings.redis_url, decode_responses=True)
    except Exception:
        return None


def _parse_prob(outcome_prices_str: str | None) -> float | None:
    if not outcome_prices_str:
        return None
    try:
        prices = json.loads(outcome_prices_str)
        return float(prices[0]) if prices else None
    except (json.JSONDecodeError, ValueError, TypeError, IndexError):
        return None


def _row_to_node(row) -> MarketNode:  # type: ignore[no-untyped-def]
    return MarketNode(
        id=row[0],
        label=row[1] or "",
        link=row[2],
        prob=_parse_prob(row[3]),
        volume=row[4],
        liquidity=row[5],
        topic=row[6],
        category=row[6],
        close_time=row[7],
        event_id=row[8],
        x=row[11],
        y=row[12],
        z=row[13],
        projection_version=row[14],
        cluster_id=None,
        distortion_score=None,
        neighborhood_key=row[15],
        neighborhood_label=row[16],
        local_cluster_id=row[17],
        local_x=row[18],
        local_y=row[19],
        local_z=row[20],
        global_x=row[21],
        global_y=row[22],
        global_z=row[23],
        local_distortion=row[24],
        stitch_distortion=row[25],
    )


def _row_to_detail(row) -> MarketDetail:  # type: ignore[no-untyped-def]
    return MarketDetail(
        id=row[0],
        title=row[1] or "",
        link=row[2],
        description=row[9],
        category=row[6],
        close_time=row[7],
        liquidity=row[5],
        volume=row[4],
        probability=_parse_prob(row[3]),
        event_id=row[8],
        is_active=row[10] == 1.0 if row[10] is not None else True,
        x=row[11],
        y=row[12],
        z=row[13],
        projection_version=row[14],
        cluster_id=None,
        distortion_score=None,
        neighborhood_key=row[15],
        neighborhood_label=row[16],
        local_cluster_id=row[17],
        local_x=row[18],
        local_y=row[19],
        local_z=row[20],
        global_x=row[21],
        global_y=row[22],
        global_z=row[23],
        local_distortion=row[24],
        stitch_distortion=row[25],
    )


def _edge_row_to_link(row) -> GraphLink:  # type: ignore[no-untyped-def]
    return GraphLink(
        source=row[0],
        target=row[1],
        confidence=row[2],
        type=row[3],
        weight=row[2],
        semantic_score=row[4],
        stat_score=row[5],
        logical_score=row[6],
        propagation_score=row[7],
        entity_overlap_score=row[8],
        template_penalty=row[9],
        explanation=row[10],
    )


async def _fetch_event_links(
    session: AsyncSession,
    event_ids: list[str],
    min_conf: float,
) -> list[GraphLink]:
    if len(event_ids) < 2:
        return []

    result = await session.execute(
        text(
            """
            WITH node_set AS (
                SELECT UNNEST(CAST(:event_ids AS text[])) AS event_id
            ),
            cluster_nodes AS (
                SELECT
                    ep.event_id,
                    ep.local_cluster_id,
                    COALESCE(NULLIF(ep.neighborhood_key, ''), 'misc') AS neighborhood_key
                FROM event_projection_3d ep
                JOIN node_set ns ON ns.event_id = ep.event_id
                WHERE ep.local_cluster_id IS NOT NULL
                  AND ep.local_cluster_id >= 0
            ),
            cluster_sizes AS (
                SELECT neighborhood_key, local_cluster_id, COUNT(*)::int AS cluster_size
                FROM cluster_nodes
                GROUP BY neighborhood_key, local_cluster_id
            ),
            directed AS (
                SELECT
                    a.event_id AS source_event_id,
                    b.target_event_id AS target_event_id,
                    a.local_cluster_id AS cluster_id,
                    a.neighborhood_key AS neighborhood_key
                FROM cluster_nodes a
                JOIN LATERAL (
                    SELECT c.event_id AS target_event_id
                    FROM cluster_nodes c
                    WHERE c.local_cluster_id = a.local_cluster_id
                      AND c.neighborhood_key = a.neighborhood_key
                      AND c.event_id <> a.event_id
                    ORDER BY c.event_id
                    LIMIT :max_edges_per_node
                ) b ON TRUE
            ),
            undirected AS (
                SELECT
                    LEAST(source_event_id, target_event_id) AS source_event_id,
                    GREATEST(source_event_id, target_event_id) AS target_event_id,
                    cluster_id,
                    neighborhood_key,
                    COUNT(*)::int AS support
                FROM directed
                GROUP BY 1, 2, 3, 4
            ),
            ranked AS (
                SELECT
                    u.source_event_id,
                    u.target_event_id,
                    1.0::double precision AS confidence,
                    u.cluster_id,
                    u.neighborhood_key,
                    u.support,
                    LEAST(:cluster_edge_cap_max, cs.cluster_size * :cluster_edge_cap_factor) AS cluster_edge_cap,
                    ROW_NUMBER() OVER (
                        PARTITION BY u.neighborhood_key, u.cluster_id
                        ORDER BY u.support DESC, u.source_event_id, u.target_event_id
                    ) AS edge_rank
                FROM undirected u
                JOIN cluster_sizes cs
                  ON cs.local_cluster_id = u.cluster_id
                 AND cs.neighborhood_key = u.neighborhood_key
            )
            SELECT source_event_id, target_event_id, confidence, cluster_id
            FROM ranked
            WHERE confidence >= :min_conf
              AND edge_rank <= cluster_edge_cap
            ORDER BY neighborhood_key, cluster_id, source_event_id, target_event_id
            """
        ),
        {
            "event_ids": event_ids,
            "min_conf": min_conf,
            "max_edges_per_node": 10,
            "cluster_edge_cap_max": 500,
            "cluster_edge_cap_factor": 3,
        },
    )

    links: list[GraphLink] = []
    for row in result.fetchall():
        src = str(row[0])
        tgt = str(row[1])
        confidence = float(row[2])
        cluster_id = int(row[3])
        explanation = {
            "type": "hdbscan_cluster",
            "cluster_id": cluster_id,
            "link_rule": "same_cluster_limited",
            "max_edges_per_node": 10,
            "cluster_edge_cap_max": 500,
            "cluster_edge_cap_factor": 3,
        }

        links.append(
            GraphLink(
                source=src,
                target=tgt,
                confidence=confidence,
                type="hdbscan_cluster",
                weight=confidence,
                template_penalty=0.0,
                explanation=explanation,
            )
        )

    return links


async def _latest_projection_version(session: AsyncSession) -> str | None:
    result = await session.execute(
        text(
            """
            SELECT projection_version
            FROM market_projection_3d
            ORDER BY updated_at DESC
            LIMIT 1
            """
        )
    )
    return result.scalar()


async def _attach_cluster_ids(session: AsyncSession, nodes: list[MarketNode]) -> None:
    """Attach cluster_id to nodes based on latest clustering for projection version."""
    if not nodes:
        return

    projection_version = next((n.projection_version for n in nodes if n.projection_version), None)
    if not projection_version:
        return

    node_ids = [n.id for n in nodes]
    rows = await session.execute(
        text(
            """
            SELECT market_id, cluster_id
            FROM market_clusters
            WHERE projection_version = :projection_version
              AND market_id = ANY(:node_ids)
            """
        ),
        {"projection_version": projection_version, "node_ids": node_ids},
    )
    cluster_map = {row[0]: row[1] for row in rows.fetchall()}
    for node in nodes:
        node.cluster_id = cluster_map.get(node.id)


async def _attach_cluster_ids_to_details(session: AsyncSession, items: list[MarketDetail]) -> None:
    if not items:
        return
    projection_version = next((i.projection_version for i in items if i.projection_version), None)
    if not projection_version:
        return
    market_ids = [i.id for i in items]
    rows = await session.execute(
        text(
            """
            SELECT market_id, cluster_id
            FROM market_clusters
            WHERE projection_version = :projection_version
              AND market_id = ANY(:market_ids)
            """
        ),
        {"projection_version": projection_version, "market_ids": market_ids},
    )
    cluster_map = {row[0]: row[1] for row in rows.fetchall()}
    for item in items:
        item.cluster_id = cluster_map.get(item.id)


async def _attach_distortion_scores(session: AsyncSession, nodes: list[MarketNode]) -> None:
    """Attach distortion_score to nodes for the relevant projection version."""
    if not nodes:
        return
    projection_version = next((n.projection_version for n in nodes if n.projection_version), None)
    if not projection_version:
        return
    node_ids = [n.id for n in nodes]
    rows = await session.execute(
        text(
            """
            SELECT market_id, distortion_score
            FROM market_projection_distortion
            WHERE projection_version = :projection_version
              AND market_id = ANY(:node_ids)
            """
        ),
        {"projection_version": projection_version, "node_ids": node_ids},
    )
    score_map = {row[0]: row[1] for row in rows.fetchall()}
    for node in nodes:
        node.distortion_score = score_map.get(node.id)


async def _attach_distortion_scores_to_details(
    session: AsyncSession, items: list[MarketDetail]
) -> None:
    if not items:
        return
    projection_version = next((i.projection_version for i in items if i.projection_version), None)
    if not projection_version:
        return
    market_ids = [i.id for i in items]
    rows = await session.execute(
        text(
            """
            SELECT market_id, distortion_score
            FROM market_projection_distortion
            WHERE projection_version = :projection_version
              AND market_id = ANY(:market_ids)
            """
        ),
        {"projection_version": projection_version, "market_ids": market_ids},
    )
    score_map = {row[0]: row[1] for row in rows.fetchall()}
    for item in items:
        item.distortion_score = score_map.get(item.id)


def _lod_limits(lod: str, max_nodes: int, max_edges_per_node: int) -> tuple[str, int, int]:
    normalized = lod.lower()
    if normalized == "auto":
        normalized = "mid"
    if normalized == "far":
        return normalized, min(max_nodes, 1200), min(max_edges_per_node, 5)
    if normalized == "mid":
        return normalized, min(max_nodes, 2500), min(max_edges_per_node, 10)
    return "near", max_nodes, max_edges_per_node


def _cache_key_for_viewport(
    projection_version: str,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    lod: str,
    include_edges: bool,
    max_nodes: int,
    max_edges_per_node: int,
    min_similarity: float,
) -> str:
    payload = {
        "projection_version": projection_version,
        "min_x": round(min_x, 4),
        "max_x": round(max_x, 4),
        "min_y": round(min_y, 4),
        "max_y": round(max_y, 4),
        "min_z": round(min_z, 4),
        "max_z": round(max_z, 4),
        "lod": lod,
        "include_edges": include_edges,
        "max_nodes": max_nodes,
        "max_edges_per_node": max_edges_per_node,
        "min_similarity": min_similarity,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"discovery_viewport:{digest}"


async def _build_discovery_viewport(
    session: AsyncSession,
    projection_version: str,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    include_edges: bool,
    max_nodes: int,
    max_edges_per_node: int,
    min_similarity: float,
    lod: str,
) -> dict[str, Any]:
    node_result = await session.execute(
        text(
            f"""
            SELECT {MARKET_COLS}
            FROM markets m
            JOIN market_projection_3d p ON p.market_id = m.id
            WHERE m.is_active = 1.0
              AND p.projection_version = :projection_version
              AND p.x BETWEEN :min_x AND :max_x
              AND p.y BETWEEN :min_y AND :max_y
              AND p.z BETWEEN :min_z AND :max_z
            ORDER BY m.volume DESC NULLS LAST
            LIMIT :max_nodes
            """
        ),
        {
            "projection_version": projection_version,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "min_z": min_z,
            "max_z": max_z,
            "max_nodes": max_nodes,
        },
    )

    nodes = [_row_to_node(row) for row in node_result.fetchall()]
    await _attach_cluster_ids(session, nodes)
    await _attach_distortion_scores(session, nodes)
    node_ids = [n.id for n in nodes]

    links: list[GraphLink] = []
    if include_edges and node_ids:
        edge_result = await session.execute(
            text(
                """
                WITH ranked_edges AS (
                    SELECT source_id, target_id, confidence_score, edge_type,
                           semantic_score, stat_score, logical_score, propagation_score,
                           entity_overlap_score, template_penalty, explanation,
                           ROW_NUMBER() OVER (
                               PARTITION BY source_id
                               ORDER BY confidence_score DESC
                           ) AS rn
                    FROM market_edges
                    WHERE edge_type = 'discovery'
                      AND confidence_score >= :min_similarity
                      AND source_id = ANY(:node_ids)
                      AND target_id = ANY(:node_ids)
                )
                SELECT source_id, target_id, confidence_score, edge_type,
                       semantic_score, stat_score, logical_score, propagation_score,
                       entity_overlap_score, template_penalty, explanation
                FROM ranked_edges
                WHERE rn <= :max_edges_per_node
                ORDER BY confidence_score DESC
                LIMIT :edge_cap
                """
            ),
            {
                "node_ids": node_ids,
                "min_similarity": min_similarity,
                "max_edges_per_node": max_edges_per_node,
                "edge_cap": max_nodes * max_edges_per_node,
            },
        )
        links = [_edge_row_to_link(row) for row in edge_result.fetchall()]

    return {
        "nodes": [n.model_dump() for n in nodes],
        "links": [l.model_dump() for l in links],
        "meta": {
            "projection_version": projection_version,
            "lod": lod,
            "bounds": {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "min_z": min_z,
                "max_z": max_z,
            },
            "node_count": len(nodes),
            "edge_count": len(links),
        },
    }


async def _fetch_event_nodes(
    session: AsyncSession,
    neighborhood_key: str | None,
    min_distortion: float | None,
    include_local: bool,
) -> list[MarketNode]:
    where = ["1=1"]
    params: dict[str, Any] = {}
    if neighborhood_key:
        where.append("ep.neighborhood_key = :neighborhood_key")
        params["neighborhood_key"] = neighborhood_key
    if min_distortion is not None:
        where.append("COALESCE(ep.local_distortion, 0.0) >= :min_distortion")
        params["min_distortion"] = min_distortion

    result = await session.execute(
        text(
            f"""
            SELECT ep.event_id,
                   COALESCE(e.title, ep.event_id) AS label,
                   NULL::text AS link,
                   NULL::text AS outcome_prices,
                   NULL::double precision AS volume,
                   NULL::double precision AS liquidity,
                   ep.neighborhood_label AS category,
                   NULL::timestamptz AS close_time,
                   ep.event_id AS event_id,
                   COALESCE(e.raw->>'description', '') AS description,
                   1.0 AS is_active,
                   ep.x, ep.y, ep.z,
                   ep.projection_version,
                   ep.neighborhood_key,
                   ep.neighborhood_label,
                   ep.local_cluster_id,
                   NULL::double precision AS local_x,
                   NULL::double precision AS local_y,
                   NULL::double precision AS local_z,
                   ep.x AS global_x,
                   ep.y AS global_y,
                   ep.z AS global_z,
                   ep.local_distortion,
                   ep.stitch_distortion
            FROM event_projection_3d ep
            JOIN polymarket_events e ON e.id = ep.event_id
            WHERE {' AND '.join(where)}
            ORDER BY ep.event_id ASC
            """
        ),
        params,
    )
    nodes = [_row_to_node(row) for row in result.fetchall()]
    if not include_local:
        for node in nodes:
            node.local_cluster_id = None
            node.local_x = None
            node.local_y = None
            node.local_z = None
            node.local_distortion = None
            node.stitch_distortion = None
    for node in nodes:
        if node.local_cluster_id is not None:
            node.cluster_id = f"{node.neighborhood_key}:{node.local_cluster_id}"
        node.distortion_score = node.local_distortion
    return nodes


# --- Health ---


@app.get("/health", response_model=HealthResponse)
async def health_check(session: AsyncSession = Depends(get_async_session)) -> HealthResponse:
    try:
        result = await session.execute(text("SELECT COUNT(*) FROM markets WHERE is_active = 1.0"))
        markets_count = result.scalar()

        result = await session.execute(text("SELECT MAX(timestamp) FROM market_prices"))
        latest_snapshot = result.scalar()

        return HealthResponse(
            status="ok",
            markets_count=markets_count,
            latest_snapshot=latest_snapshot,
        )
    except Exception:
        return HealthResponse(status="ok")


# --- Discovery Graph ---


@app.get("/graph/discovery", response_model=GraphResponse)
async def get_discovery_graph(
    topic: str | None = Query(None, description="Filter by topic/category"),
    min_conf: float = Query(0.3, ge=0.0, le=1.0, description="Minimum confidence for edges"),
    limit: int = Query(200, ge=1, le=1000, description="Max nodes to return"),
    session: AsyncSession = Depends(get_async_session),
) -> GraphResponse:
    params: dict[str, Any] = {"limit": limit}
    where_clauses = ["m.is_active = 1.0"]

    if topic:
        where_clauses.append("(m.category ILIKE :topic OR m.title ILIKE :topic)")
        params["topic"] = f"%{topic}%"

    where_sql = " AND ".join(where_clauses)

    result = await session.execute(
        text(
            f"SELECT {MARKET_COLS} FROM markets m {MARKET_JOIN} WHERE {where_sql} "
            "ORDER BY m.volume DESC NULLS LAST LIMIT :limit"
        ),
        params,
    )

    nodes = [_row_to_node(row) for row in result.fetchall()]
    await _attach_cluster_ids(session, nodes)
    await _attach_distortion_scores(session, nodes)
    node_ids = [n.id for n in nodes]

    links: list[GraphLink] = []
    if node_ids:
        edge_result = await session.execute(
            text(
                """
                SELECT source_id, target_id, confidence_score, edge_type,
                       semantic_score, stat_score, logical_score, propagation_score,
                       entity_overlap_score, template_penalty, explanation
                FROM market_edges
                WHERE edge_type = 'discovery'
                  AND confidence_score >= :min_conf
                  AND source_id = ANY(:node_ids)
                  AND target_id = ANY(:node_ids)
                ORDER BY confidence_score DESC
                LIMIT 1000
                """
            ),
            {"min_conf": min_conf, "node_ids": node_ids},
        )
        links = [_edge_row_to_link(erow) for erow in edge_result.fetchall()]

    projection_version = next((n.projection_version for n in nodes if n.projection_version), None)

    return GraphResponse(
        nodes=nodes,
        links=links,
        meta={
            "topic": topic,
            "min_conf": min_conf,
            "node_count": len(nodes),
            "edge_count": len(links),
            "projection_version": projection_version,
        },
    )


@app.get("/graph/discovery/all", response_model=GraphResponse)
async def get_discovery_graph_all(
    min_conf: float = Query(0.3, ge=0.0, le=1.0, description="Minimum confidence for edges"),
    include_edges: bool = Query(True, description="Include discovery edges"),
    neighborhood_key: str | None = Query(None, description="Filter by neighborhood key"),
    min_distortion: float | None = Query(None, ge=0.0, le=1.0),
    include_local: bool = Query(True, description="Include local coordinate fields"),
    entity: str = Query("events", pattern="^(events|markets)$"),
    session: AsyncSession = Depends(get_async_session),
) -> GraphResponse:
    """Return the full discovery graph (all active markets + discovery edges)."""
    if entity == "events":
        nodes = await _fetch_event_nodes(
            session=session,
            neighborhood_key=neighborhood_key,
            min_distortion=min_distortion,
            include_local=include_local,
        )
        event_ids = [n.id for n in nodes]
        links = (
            await _fetch_event_links(
                session=session,
                event_ids=event_ids,
                min_conf=min_conf,
            )
            if include_edges and event_ids
            else []
        )
        projection_version = next((n.projection_version for n in nodes if n.projection_version), None)
        return GraphResponse(
            nodes=nodes,
            links=links,
            meta={
                "scope": "all",
                "entity": "events",
                "min_conf": min_conf,
                "include_edges": include_edges,
                "neighborhood_key": neighborhood_key,
                "min_distortion": min_distortion,
                "include_local": include_local,
                "node_count": len(nodes),
                "edge_count": len(links),
                "projection_version": projection_version,
                "source": "postgres",
            },
        )

    if settings.memgraph_enabled and memgraph_is_available() and neighborhood_key is None and min_distortion is None:
        payload = fetch_discovery_graph(min_conf=min_conf, include_edges=include_edges)
        nodes_payload = payload.get("nodes", [])
        projection_version = payload.get("meta", {}).get("projection_version")
        if isinstance(nodes_payload, list):
            missing_clusters = any(
                node.get("cluster_id") is None for node in nodes_payload if isinstance(node, dict)
            )
            missing_distortion = any(
                node.get("distortion_score") is None
                for node in nodes_payload
                if isinstance(node, dict)
            )
            if missing_clusters or missing_distortion:
                node_ids = [
                    node.get("id")
                    for node in nodes_payload
                    if isinstance(node, dict) and node.get("id")
                ]
                if node_ids:
                    cluster_map: dict[str, str] = {}
                    distortion_map: dict[str, float] = {}
                    if projection_version:
                        rows = await session.execute(
                            text(
                                """
                                SELECT market_id, cluster_id
                                FROM market_clusters
                                WHERE projection_version = :projection_version
                                  AND market_id = ANY(:node_ids)
                                """
                            ),
                            {"projection_version": projection_version, "node_ids": node_ids},
                        )
                        cluster_map = {row[0]: row[1] for row in rows.fetchall()}
                        distortion_rows = await session.execute(
                            text(
                                """
                                SELECT market_id, distortion_score
                                FROM market_projection_distortion
                                WHERE projection_version = :projection_version
                                  AND market_id = ANY(:node_ids)
                                """
                            ),
                            {"projection_version": projection_version, "node_ids": node_ids},
                        )
                        distortion_map = {row[0]: row[1] for row in distortion_rows.fetchall()}
                    if not cluster_map:
                        latest_rows = await session.execute(
                            text(
                                """
                                SELECT DISTINCT ON (market_id) market_id, cluster_id
                                FROM market_clusters
                                WHERE market_id = ANY(:node_ids)
                                ORDER BY market_id, updated_at DESC
                                """
                            ),
                            {"node_ids": node_ids},
                        )
                        cluster_map = {row[0]: row[1] for row in latest_rows.fetchall()}
                    if not distortion_map:
                        latest_distortion_rows = await session.execute(
                            text(
                                """
                                SELECT DISTINCT ON (market_id) market_id, distortion_score
                                FROM market_projection_distortion
                                WHERE market_id = ANY(:node_ids)
                                ORDER BY market_id, updated_at DESC
                                """
                            ),
                            {"node_ids": node_ids},
                        )
                        distortion_map = {
                            row[0]: row[1] for row in latest_distortion_rows.fetchall()
                        }
                    for node in nodes_payload:
                        if isinstance(node, dict):
                            market_id = node.get("id")
                            if market_id:
                                node["cluster_id"] = cluster_map.get(market_id)
                                node["distortion_score"] = distortion_map.get(market_id)
        return GraphResponse(**payload)

    where_sql = "m.is_active = 1.0"
    params: dict[str, Any] = {"min_conf": min_conf}
    if neighborhood_key:
        where_sql += " AND m.neighborhood_key = :neighborhood_key"
        params["neighborhood_key"] = neighborhood_key
    if min_distortion is not None:
        where_sql += " AND COALESCE(m.local_distortion, 0.0) >= :min_distortion"
        params["min_distortion"] = min_distortion

    result = await session.execute(
        text(
            f"SELECT {MARKET_COLS} FROM markets m {MARKET_JOIN} "
            f"WHERE {where_sql} ORDER BY m.volume DESC NULLS LAST"
        ),
        params,
    )

    nodes = [_row_to_node(row) for row in result.fetchall()]
    await _attach_cluster_ids(session, nodes)
    await _attach_distortion_scores(session, nodes)
    if not include_local:
        for node in nodes:
            node.local_x = None
            node.local_y = None
            node.local_z = None
            node.local_cluster_id = None
            node.local_distortion = None
            node.stitch_distortion = None
    node_ids = [n.id for n in nodes]

    links: list[GraphLink] = []
    if include_edges and node_ids:
        edge_result = await session.execute(
            text(
                """
                SELECT source_id, target_id, confidence_score, edge_type,
                       semantic_score, stat_score, logical_score, propagation_score,
                       entity_overlap_score, template_penalty, explanation
                FROM market_edges
                WHERE edge_type = 'discovery'
                  AND confidence_score >= :min_conf
                  AND source_id = ANY(:node_ids)
                  AND target_id = ANY(:node_ids)
                ORDER BY confidence_score DESC
                """
            ),
            {"min_conf": min_conf, "node_ids": node_ids},
        )
        links = [_edge_row_to_link(erow) for erow in edge_result.fetchall()]

    projection_version = next((n.projection_version for n in nodes if n.projection_version), None)

    return GraphResponse(
        nodes=nodes,
        links=links,
        meta={
            "scope": "all",
            "entity": "markets",
            "min_conf": min_conf,
            "include_edges": include_edges,
            "neighborhood_key": neighborhood_key,
            "min_distortion": min_distortion,
            "include_local": include_local,
            "node_count": len(nodes),
            "edge_count": len(links),
            "projection_version": projection_version,
            "source": "postgres",
        },
    )


@app.get("/graph/neighborhoods", response_model=list[NeighborhoodNode])
async def get_neighborhoods(session: AsyncSession = Depends(get_async_session)) -> list[NeighborhoodNode]:
    result = await session.execute(
        text(
            """
            SELECT neighborhood_key, label, market_count, anchor_x, anchor_y, anchor_z, scale, meta
            FROM neighborhoods
            ORDER BY market_count DESC, neighborhood_key ASC
            """
        )
    )
    return [
        NeighborhoodNode(
            neighborhood_key=row[0],
            label=row[1],
            market_count=int(row[2]),
            anchor_x=row[3],
            anchor_y=row[4],
            anchor_z=row[5],
            scale=float(row[6]),
            meta=row[7] or {},
        )
        for row in result.fetchall()
    ]


@app.get("/graph/discovery/viewport", response_model=GraphResponse)
async def get_discovery_viewport(
    min_x: float = Query(...),
    max_x: float = Query(...),
    min_y: float = Query(...),
    max_y: float = Query(...),
    min_z: float | None = Query(None),
    max_z: float | None = Query(None),
    projection_version: str | None = Query(None),
    pad: float = Query(settings.discovery_viewport_default_pad, ge=0.0, le=1.0),
    max_nodes: int = Query(settings.discovery_viewport_default_max_nodes, ge=50, le=5000),
    max_edges_per_node: int = Query(
        settings.discovery_viewport_default_max_edges_per_node, ge=1, le=30
    ),
    min_similarity: float = Query(0.55, ge=0.0, le=1.0),
    lod: str = Query("auto"),
    include_edges: bool = Query(True),
    session: AsyncSession = Depends(get_async_session),
) -> GraphResponse:
    if min_x > max_x or min_y > max_y:
        raise HTTPException(status_code=400, detail="Invalid bounds")

    pversion = projection_version or await _latest_projection_version(session)
    if not pversion:
        return GraphResponse(nodes=[], links=[], meta={"error": "No projection available"})

    span_x = max_x - min_x
    span_y = max_y - min_y
    span_z = (max_z - min_z) if (min_z is not None and max_z is not None) else max(span_x, span_y)

    min_x_pad = min_x - (span_x * pad)
    max_x_pad = max_x + (span_x * pad)
    min_y_pad = min_y - (span_y * pad)
    max_y_pad = max_y + (span_y * pad)

    z_min = min_z if min_z is not None else -1e9
    z_max = max_z if max_z is not None else 1e9
    min_z_pad = z_min - (span_z * pad)
    max_z_pad = z_max + (span_z * pad)

    lod_resolved, max_nodes_resolved, max_edges_resolved = _lod_limits(
        lod, max_nodes, max_edges_per_node
    )

    redis_client = _get_redis_client()
    cache_key = _cache_key_for_viewport(
        projection_version=pversion,
        min_x=min_x_pad,
        max_x=max_x_pad,
        min_y=min_y_pad,
        max_y=max_y_pad,
        min_z=min_z_pad,
        max_z=max_z_pad,
        lod=lod_resolved,
        include_edges=include_edges,
        max_nodes=max_nodes_resolved,
        max_edges_per_node=max_edges_resolved,
        min_similarity=min_similarity,
    )

    if redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                payload = json.loads(str(cached))
                return GraphResponse(**payload)
        except Exception:
            logger.warning("Redis cache read failed", exc_info=True)

    payload = await _build_discovery_viewport(
        session=session,
        projection_version=pversion,
        min_x=min_x_pad,
        max_x=max_x_pad,
        min_y=min_y_pad,
        max_y=max_y_pad,
        min_z=min_z_pad,
        max_z=max_z_pad,
        include_edges=include_edges,
        max_nodes=max_nodes_resolved,
        max_edges_per_node=max_edges_resolved,
        min_similarity=min_similarity,
        lod=lod_resolved,
    )

    if redis_client is not None:
        try:
            redis_client.setex(
                cache_key,
                settings.discovery_viewport_cache_ttl_seconds,
                json.dumps(payload),
            )
        except Exception:
            logger.warning("Redis cache write failed", exc_info=True)

    return GraphResponse(**payload)


@app.get("/graph/discovery/tile", response_model=GraphResponse)
async def get_discovery_tile(
    tile_x: int,
    tile_y: int,
    tile_z: int,
    tile_size: float = Query(0.5, gt=0.0),
    projection_version: str | None = Query(None),
    lod: str = Query("auto"),
    include_edges: bool = Query(True),
    min_similarity: float = Query(0.55, ge=0.0, le=1.0),
    max_nodes: int = Query(settings.discovery_viewport_default_max_nodes, ge=50, le=5000),
    max_edges_per_node: int = Query(
        settings.discovery_viewport_default_max_edges_per_node, ge=1, le=30
    ),
    session: AsyncSession = Depends(get_async_session),
) -> GraphResponse:
    min_x = tile_x * tile_size
    max_x = min_x + tile_size
    min_y = tile_y * tile_size
    max_y = min_y + tile_size
    min_z = tile_z * tile_size
    max_z = min_z + tile_size

    return await get_discovery_viewport(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        min_z=min_z,
        max_z=max_z,
        projection_version=projection_version,
        pad=settings.discovery_viewport_default_pad,
        max_nodes=max_nodes,
        max_edges_per_node=max_edges_per_node,
        min_similarity=min_similarity,
        lod=lod,
        include_edges=include_edges,
        session=session,
    )


# --- Hedge Graph ---


@app.get("/market/{market_id}/related", response_model=GraphResponse)
async def get_related_markets(
    market_id: str,
    min_conf: float = Query(0.7, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(20, ge=1, le=100, description="Max related markets"),
    edge_types: str = Query("all", description="Comma-separated edge types or 'all'"),
    session: AsyncSession = Depends(get_async_session),
) -> GraphResponse:
    focal_result = await session.execute(
        text(f"SELECT {MARKET_COLS} FROM markets m {MARKET_JOIN} WHERE m.id = :market_id"),
        {"market_id": market_id},
    )
    focal_row = focal_result.fetchone()
    if not focal_row:
        return GraphResponse(nodes=[], links=[], meta={"error": "Market not found"})

    focal_node = _row_to_node(focal_row)

    type_filter = ""
    params: dict[str, Any] = {"market_id": market_id, "min_conf": min_conf, "limit": limit}
    if edge_types != "all":
        types = [t.strip() for t in edge_types.split(",")]
        type_filter = "AND edge_type = ANY(:edge_types)"
        params["edge_types"] = types

    edge_result = await session.execute(
        text(
            f"""
            SELECT source_id, target_id, confidence_score, edge_type,
                   semantic_score, stat_score, logical_score, propagation_score,
                   entity_overlap_score, template_penalty, explanation
            FROM market_edges
            WHERE (source_id = :market_id OR target_id = :market_id)
              AND confidence_score >= :min_conf {type_filter}
            ORDER BY confidence_score DESC
            LIMIT :limit
            """
        ),
        params,
    )

    related_ids: set[str] = set()
    links = [_edge_row_to_link(erow) for erow in edge_result.fetchall()]
    for link in links:
        if link.source != market_id:
            related_ids.add(link.source)
        if link.target != market_id:
            related_ids.add(link.target)

    nodes: list[MarketNode] = [focal_node]
    if related_ids:
        related_result = await session.execute(
            text(f"SELECT {MARKET_COLS} FROM markets m {MARKET_JOIN} WHERE m.id = ANY(:ids)"),
            {"ids": list(related_ids)},
        )
        nodes.extend(_row_to_node(row) for row in related_result.fetchall())
    await _attach_cluster_ids(session, nodes)
    await _attach_distortion_scores(session, nodes)

    return GraphResponse(
        nodes=nodes,
        links=links,
        meta={
            "focal_market": market_id,
            "min_conf": min_conf,
            "related_count": len(related_ids),
            "edge_count": len(links),
        },
    )


@app.get("/event/{event_id}/context", response_model=EventContextResponse)
async def get_event_context(
    event_id: str,
    related_limit: int = Query(5000, ge=1, le=50000),
    session: AsyncSession = Depends(get_async_session),
) -> EventContextResponse:
    event_row = await session.execute(
        text(
            """
            SELECT ep.event_id,
                   COALESCE(e.title, ep.event_id) AS label,
                   ep.local_cluster_id,
                   COALESCE(NULLIF(ep.neighborhood_key, ''), 'misc') AS neighborhood_key,
                   ep.x, ep.y, ep.z
            FROM event_projection_3d ep
            JOIN polymarket_events e ON e.id = ep.event_id
            WHERE ep.event_id = :event_id
            """
        ),
        {"event_id": event_id},
    )
    row = event_row.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Event not found")

    tags_result = await session.execute(
        text(
            """
            SELECT pt.tag
            FROM polymarket_event_tags pet
            JOIN polymarket_tags pt ON pt.id = pet.tag_id
            WHERE pet.event_id = :event_id
            ORDER BY pt.tag ASC
            """
        ),
        {"event_id": event_id},
    )
    tags = [str(r[0]) for r in tags_result.fetchall() if r[0]]

    related_nodes: list[EventRelatedNode] = []
    local_cluster_id = row[2]
    neighborhood_key = str(row[3])
    if local_cluster_id is not None and int(local_cluster_id) >= 0:
        related_result = await session.execute(
            text(
                """
                SELECT ep.event_id,
                       COALESCE(e.title, ep.event_id) AS label,
                       sqrt(
                         power(ep.x - :x, 2) +
                         power(ep.y - :y, 2) +
                         power(ep.z - :z, 2)
                       ) AS distance
                FROM event_projection_3d ep
                JOIN polymarket_events e ON e.id = ep.event_id
                WHERE ep.local_cluster_id = :cluster_id
                  AND COALESCE(NULLIF(ep.neighborhood_key, ''), 'misc') = :neighborhood_key
                  AND ep.event_id <> :event_id
                ORDER BY distance ASC, ep.event_id ASC
                LIMIT :related_limit
                """
            ),
            {
                "event_id": event_id,
                "cluster_id": int(local_cluster_id),
                "neighborhood_key": neighborhood_key,
                "x": float(row[4]),
                "y": float(row[5]),
                "z": float(row[6]),
                "related_limit": related_limit,
            },
        )
        related_nodes = [
            EventRelatedNode(
                id=str(rr[0]),
                label=str(rr[1]),
                distance=float(rr[2]) if rr[2] is not None else None,
            )
            for rr in related_result.fetchall()
        ]

    return EventContextResponse(
        event_id=str(row[0]),
        label=str(row[1]),
        tags=tags,
        related_nodes=related_nodes,
    )


@app.get("/graph/hedge", response_model=GraphResponse)
async def get_hedge_graph(
    topic: str | None = Query(None, description="Filter by topic/category"),
    min_conf: float = Query(0.45, ge=0.0, le=1.0, description="Minimum confidence for hedge edges"),
    limit: int = Query(200, ge=1, le=1000, description="Max nodes to return"),
    session: AsyncSession = Depends(get_async_session),
) -> GraphResponse:
    params: dict[str, Any] = {"limit": limit}
    where_clauses = ["m.is_active = 1.0"]

    if topic:
        where_clauses.append("(m.category ILIKE :topic OR m.title ILIKE :topic)")
        params["topic"] = f"%{topic}%"

    where_sql = " AND ".join(where_clauses)

    result = await session.execute(
        text(
            f"SELECT {MARKET_COLS} FROM markets m {MARKET_JOIN} WHERE {where_sql} "
            "ORDER BY m.volume DESC NULLS LAST LIMIT :limit"
        ),
        params,
    )

    nodes = [_row_to_node(row) for row in result.fetchall()]
    await _attach_cluster_ids(session, nodes)
    await _attach_distortion_scores(session, nodes)
    node_ids = [n.id for n in nodes]

    links: list[GraphLink] = []
    if node_ids:
        edge_result = await session.execute(
            text(
                """
                SELECT source_id, target_id, confidence_score, edge_type,
                       semantic_score, stat_score, logical_score, propagation_score,
                       entity_overlap_score, template_penalty, explanation
                FROM market_edges
                WHERE edge_type = 'hedge'
                  AND confidence_score >= :min_conf
                  AND source_id = ANY(:node_ids)
                  AND target_id = ANY(:node_ids)
                ORDER BY confidence_score DESC
                LIMIT 1000
                """
            ),
            {"min_conf": min_conf, "node_ids": node_ids},
        )
        links = [_edge_row_to_link(erow) for erow in edge_result.fetchall()]

    return GraphResponse(
        nodes=nodes,
        links=links,
        meta={
            "topic": topic,
            "min_conf": min_conf,
            "node_count": len(nodes),
            "edge_count": len(links),
            "edge_type": "hedge",
        },
    )


# --- Market detail and price history ---


@app.get("/market/{market_id}", response_model=MarketDetail)
async def get_market(
    market_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> MarketDetail:
    result = await session.execute(
        text(f"SELECT {MARKET_COLS} FROM markets m {MARKET_JOIN} WHERE m.id = :market_id"),
        {"market_id": market_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Market not found")
    detail = _row_to_detail(row)
    await _attach_cluster_ids_to_details(session, [detail])
    await _attach_distortion_scores_to_details(session, [detail])
    return detail


@app.get("/market/{market_id}/prices", response_model=MarketPriceHistory)
async def get_market_prices(
    market_id: str,
    hours: int = Query(24, ge=1, le=720, description="Hours of history"),
    session: AsyncSession = Depends(get_async_session),
) -> MarketPriceHistory:
    result = await session.execute(
        text(
            f"""
            SELECT timestamp, probability, volume
            FROM market_prices
            WHERE market_id = :market_id
              AND timestamp >= NOW() - INTERVAL '{hours} hours'
            ORDER BY timestamp ASC
            """
        ),
        {"market_id": market_id},
    )
    prices = [
        PricePoint(timestamp=row[0], probability=row[1], volume=row[2]) for row in result.fetchall()
    ]
    return MarketPriceHistory(market_id=market_id, prices=prices)


# --- Markets listing ---


@app.get("/markets", response_model=list[MarketDetail])
async def list_markets(
    category: str | None = Query(None),
    active_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_async_session),
) -> list[MarketDetail]:
    where_clauses = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    if active_only:
        where_clauses.append("m.is_active = 1.0")
    if category:
        where_clauses.append("m.category ILIKE :category")
        params["category"] = f"%{category}%"

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    result = await session.execute(
        text(
            f"SELECT {MARKET_COLS} FROM markets m {MARKET_JOIN} {where_sql} "
            "ORDER BY m.volume DESC NULLS LAST LIMIT :limit OFFSET :offset"
        ),
        params,
    )
    details = [_row_to_detail(row) for row in result.fetchall()]
    await _attach_cluster_ids_to_details(session, details)
    await _attach_distortion_scores_to_details(session, details)
    return details
