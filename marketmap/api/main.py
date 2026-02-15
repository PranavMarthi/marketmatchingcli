"""FastAPI application for MarketMap API."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from marketmap.api.schemas import (
    GraphLink,
    GraphResponse,
    HealthResponse,
    MarketDetail,
    MarketNode,
    MarketPriceHistory,
    PricePoint,
)
from marketmap.models.database import get_async_session

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

# Standard columns selected for market node/detail construction.
# Order: id, title, link, outcome_prices, volume, liquidity, category, close_time, event_id, description, is_active
MARKET_COLS = (
    "m.id, m.title, m.link, m.outcome_prices, m.volume, m.liquidity, "
    "m.category, m.close_time, m.event_id, m.description, m.is_active"
)


def _parse_prob(outcome_prices_str: str | None) -> float | None:
    if not outcome_prices_str:
        return None
    try:
        prices = json.loads(outcome_prices_str)
        return float(prices[0]) if prices else None
    except (json.JSONDecodeError, ValueError, TypeError, IndexError):
        return None


def _row_to_node(row) -> MarketNode:  # type: ignore[no-untyped-def]
    """Convert a DB row (MARKET_COLS order) to MarketNode."""
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
    )


def _row_to_detail(row) -> MarketDetail:  # type: ignore[no-untyped-def]
    """Convert a DB row (MARKET_COLS order) to MarketDetail."""
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
    )


def _edge_row_to_link(row) -> GraphLink:  # type: ignore[no-untyped-def]
    """Convert a market_edges row to GraphLink."""
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


# --- Health ---


@app.get("/health", response_model=HealthResponse)
async def health_check(session: AsyncSession = Depends(get_async_session)) -> HealthResponse:
    """Health check with basic stats."""
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
    """Serve the semantic discovery map (GitHub-style exploration)."""
    params: dict[str, Any] = {"limit": limit}
    where_clauses = ["m.is_active = 1.0"]

    if topic:
        where_clauses.append("(m.category ILIKE :topic OR m.title ILIKE :topic)")
        params["topic"] = f"%{topic}%"

    where_sql = " AND ".join(where_clauses)

    result = await session.execute(
        text(f"SELECT {MARKET_COLS} FROM markets m WHERE {where_sql} "
             "ORDER BY m.volume DESC NULLS LAST LIMIT :limit"),
        params,
    )

    nodes: list[MarketNode] = []
    node_ids: set[str] = set()
    for row in result.fetchall():
        nodes.append(_row_to_node(row))
        node_ids.add(row[0])

    links: list[GraphLink] = []
    if node_ids:
        edge_result = await session.execute(
            text("""
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
            """),
            {"min_conf": min_conf, "node_ids": list(node_ids)},
        )
        for erow in edge_result.fetchall():
            links.append(_edge_row_to_link(erow))

    return GraphResponse(
        nodes=nodes, links=links,
        meta={"topic": topic, "min_conf": min_conf,
              "node_count": len(nodes), "edge_count": len(links)},
    )


# --- Hedge Graph (Related Markets) ---


@app.get("/market/{market_id}/related", response_model=GraphResponse)
async def get_related_markets(
    market_id: str,
    min_conf: float = Query(0.7, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(20, ge=1, le=100, description="Max related markets"),
    edge_types: str = Query("all", description="Comma-separated edge types or 'all'"),
    session: AsyncSession = Depends(get_async_session),
) -> GraphResponse:
    """Serve high-confidence related markets for hedging and dependency analysis."""
    focal_result = await session.execute(
        text(f"SELECT {MARKET_COLS} FROM markets m WHERE m.id = :market_id"),
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
        text(f"""
            SELECT source_id, target_id, confidence_score, edge_type,
                   semantic_score, stat_score, logical_score, propagation_score,
                   entity_overlap_score, template_penalty, explanation
            FROM market_edges
            WHERE (source_id = :market_id OR target_id = :market_id)
              AND confidence_score >= :min_conf {type_filter}
            ORDER BY confidence_score DESC
            LIMIT :limit
        """),
        params,
    )

    related_ids: set[str] = set()
    links: list[GraphLink] = []
    for erow in edge_result.fetchall():
        links.append(_edge_row_to_link(erow))
        if erow[0] != market_id:
            related_ids.add(erow[0])
        if erow[1] != market_id:
            related_ids.add(erow[1])

    nodes: list[MarketNode] = [focal_node]
    if related_ids:
        related_result = await session.execute(
            text(f"SELECT {MARKET_COLS} FROM markets m WHERE m.id = ANY(:ids)"),
            {"ids": list(related_ids)},
        )
        for row in related_result.fetchall():
            nodes.append(_row_to_node(row))

    return GraphResponse(
        nodes=nodes, links=links,
        meta={"focal_market": market_id, "min_conf": min_conf,
              "related_count": len(related_ids), "edge_count": len(links)},
    )


@app.get("/graph/hedge", response_model=GraphResponse)
async def get_hedge_graph(
    topic: str | None = Query(None, description="Filter by topic/category"),
    min_conf: float = Query(0.45, ge=0.0, le=1.0, description="Minimum confidence for hedge edges"),
    limit: int = Query(200, ge=1, le=1000, description="Max nodes to return"),
    session: AsyncSession = Depends(get_async_session),
) -> GraphResponse:
    """Serve hedge graph edges computed from statistical + logical dependencies."""
    params: dict[str, Any] = {"limit": limit}
    where_clauses = ["m.is_active = 1.0"]

    if topic:
        where_clauses.append("(m.category ILIKE :topic OR m.title ILIKE :topic)")
        params["topic"] = f"%{topic}%"

    where_sql = " AND ".join(where_clauses)

    result = await session.execute(
        text(
            f"SELECT {MARKET_COLS} FROM markets m WHERE {where_sql} "
            "ORDER BY m.volume DESC NULLS LAST LIMIT :limit"
        ),
        params,
    )

    nodes: list[MarketNode] = []
    node_ids: set[str] = set()
    for row in result.fetchall():
        nodes.append(_row_to_node(row))
        node_ids.add(row[0])

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
            {"min_conf": min_conf, "node_ids": list(node_ids)},
        )
        for erow in edge_result.fetchall():
            links.append(_edge_row_to_link(erow))

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
    """Get details for a single market."""
    result = await session.execute(
        text(f"SELECT {MARKET_COLS} FROM markets m WHERE m.id = :market_id"),
        {"market_id": market_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Market not found")
    return _row_to_detail(row)


@app.get("/market/{market_id}/prices", response_model=MarketPriceHistory)
async def get_market_prices(
    market_id: str,
    hours: int = Query(24, ge=1, le=720, description="Hours of history"),
    session: AsyncSession = Depends(get_async_session),
) -> MarketPriceHistory:
    """Get price history for a market."""
    result = await session.execute(
        text(f"""
            SELECT timestamp, probability, volume
            FROM market_prices
            WHERE market_id = :market_id
              AND timestamp >= NOW() - INTERVAL '{hours} hours'
            ORDER BY timestamp ASC
        """),
        {"market_id": market_id},
    )
    prices = [
        PricePoint(timestamp=row[0], probability=row[1], volume=row[2])
        for row in result.fetchall()
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
    """List markets with optional filtering."""
    where_clauses = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    if active_only:
        where_clauses.append("m.is_active = 1.0")
    if category:
        where_clauses.append("m.category ILIKE :category")
        params["category"] = f"%{category}%"

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    result = await session.execute(
        text(f"SELECT {MARKET_COLS} FROM markets m {where_sql} "
             "ORDER BY m.volume DESC NULLS LAST LIMIT :limit OFFSET :offset"),
        params,
    )
    return [_row_to_detail(row) for row in result.fetchall()]
