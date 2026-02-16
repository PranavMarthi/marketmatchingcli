"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Graph Data Contract (matches spec FrontendIntegration) ---


class MarketNode(BaseModel):
    """A market node in the graph."""

    id: str
    label: str
    link: str | None = None
    prob: float | None = None
    volume: float | None = None
    liquidity: float | None = None
    topic: str | None = None
    category: str | None = None
    close_time: datetime | None = None
    event_id: str | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None
    projection_version: str | None = None
    cluster_id: str | None = None
    distortion_score: float | None = None


class GraphLink(BaseModel):
    """An edge in the graph."""

    source: str
    target: str
    confidence: float
    type: str  # semantic, statistical, logical, propagation
    weight: float | None = None
    stat_score: float | None = None
    logical_score: float | None = None
    propagation_score: float | None = None
    semantic_score: float | None = None
    entity_overlap_score: float | None = None
    template_penalty: float | None = None
    explanation: dict[str, Any] | None = None


class GraphResponse(BaseModel):
    """Standard graph response matching the frontend data contract."""

    nodes: list[MarketNode]
    links: list[GraphLink]
    meta: dict[str, Any] = Field(default_factory=dict)


# --- Market detail ---


class MarketDetail(BaseModel):
    """Full market detail response."""

    id: str
    title: str
    link: str | None = None
    description: str | None = None
    category: str | None = None
    close_time: datetime | None = None
    liquidity: float | None = None
    volume: float | None = None
    probability: float | None = None
    event_id: str | None = None
    is_active: bool = True
    x: float | None = None
    y: float | None = None
    z: float | None = None
    projection_version: str | None = None
    cluster_id: str | None = None
    distortion_score: float | None = None


class PricePoint(BaseModel):
    """A single price/probability data point."""

    timestamp: datetime
    probability: float
    volume: float | None = None


class MarketPriceHistory(BaseModel):
    """Price history for a market."""

    market_id: str
    prices: list[PricePoint]


# --- Health ---


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    markets_count: int | None = None
    latest_snapshot: datetime | None = None
