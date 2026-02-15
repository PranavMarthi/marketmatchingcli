"""SQLAlchemy ORM models for MarketMap."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Market(Base):
    """Prediction market metadata from Polymarket."""

    __tablename__ = "markets"

    id = Column(String, primary_key=True)  # Polymarket condition_id
    polymarket_id = Column(String, unique=True, nullable=True)  # Gamma numeric ID
    slug = Column(String, nullable=True, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String, nullable=True, index=True)
    close_time = Column(DateTime(timezone=True), nullable=True)
    liquidity = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    outcome_prices = Column(String, nullable=True)  # JSON string "[0.63, 0.37]"
    clob_token_ids = Column(String, nullable=True)  # JSON string of CLOB token IDs
    event_id = Column(String, nullable=True, index=True)
    is_active = Column(Float, nullable=True)  # 1.0=active, 0.0=closed
    is_template = Column(Float, nullable=True)
    neg_risk = Column(Float, nullable=True)  # negRisk flag from Polymarket
    link = Column(Text, nullable=True)  # Polymarket URL
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_markets_category", "category"),
        Index("idx_markets_event_id", "event_id"),
        Index("idx_markets_active_volume", "is_active", "volume"),
    )


class MarketPrice(Base):
    """Time-series price snapshots. Will be converted to TimescaleDB hypertable."""

    __tablename__ = "market_prices"

    market_id = Column(String, primary_key=True)
    timestamp = Column(DateTime(timezone=True), primary_key=True)
    probability = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_market_prices_market_time", "market_id", timestamp.desc()),
    )


class MarketEmbedding(Base):
    """Vector embeddings for semantic similarity."""

    __tablename__ = "market_embeddings"

    market_id = Column(String, primary_key=True)
    embedding = Column(Vector(384), nullable=False)
    model_name = Column(String, nullable=True, default="all-MiniLM-L6-v2")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class MarketEntity(Base):
    """Named entities extracted from market text."""

    __tablename__ = "market_entities"

    id = Column(String, primary_key=True)  # market_id + entity_name + entity_type
    market_id = Column(String, nullable=False, index=True)
    entity_name = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)  # PERSON, ORG, GPE, EVENT, etc.
    confidence = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_market_entities_market", "market_id"),
        Index("idx_market_entities_entity", "entity_name", "entity_type"),
    )


class MarketEdge(Base):
    """Edges in the dual graph (discovery + hedge)."""

    __tablename__ = "market_edges"

    source_id = Column(String, primary_key=True)
    target_id = Column(String, primary_key=True)
    edge_type = Column(String, primary_key=True)  # discovery, hedge
    stat_score = Column(Float, nullable=True, default=0.0)
    logical_score = Column(Float, nullable=True, default=0.0)
    propagation_score = Column(Float, nullable=True, default=0.0)
    entity_overlap_score = Column(Float, nullable=True, default=0.0)
    semantic_score = Column(Float, nullable=True, default=0.0)
    template_penalty = Column(Float, nullable=True, default=0.0)
    confidence_score = Column(Float, nullable=False)
    explanation = Column(JSONB, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_edges_source_conf", "source_id", confidence_score.desc()),
        Index("idx_edges_target_conf", "target_id", confidence_score.desc()),
        Index("idx_edges_type_conf", "edge_type", confidence_score.desc()),
    )


class PolymarketEvent(Base):
    """Polymarket event (parent of markets). Useful for logical linkage."""

    __tablename__ = "polymarket_events"

    id = Column(String, primary_key=True)
    slug = Column(String, nullable=True)
    title = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    category = Column(String, nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    liquidity = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    neg_risk = Column(Float, nullable=True)
    is_active = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
