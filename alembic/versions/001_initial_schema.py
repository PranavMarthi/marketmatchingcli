"""Initial schema with all MarketMap tables.

Revision ID: 001
Revises: None
Create Date: 2026-02-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- markets ---
    op.create_table(
        "markets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("polymarket_id", sa.String(), unique=True, nullable=True),
        sa.Column("slug", sa.String(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(), nullable=True),
        sa.Column("close_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("liquidity", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("outcome_prices", sa.String(), nullable=True),
        sa.Column("clob_token_ids", sa.String(), nullable=True),
        sa.Column("event_id", sa.String(), nullable=True),
        sa.Column("is_active", sa.Float(), nullable=True),
        sa.Column("is_template", sa.Float(), nullable=True),
        sa.Column("neg_risk", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_markets_category", "markets", ["category"])
    op.create_index("idx_markets_event_id", "markets", ["event_id"])
    op.create_index("idx_markets_active_volume", "markets", ["is_active", "volume"])
    op.create_index("idx_markets_slug", "markets", ["slug"])

    # --- polymarket_events ---
    op.create_table(
        "polymarket_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("slug", sa.String(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(), nullable=True),
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("liquidity", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("neg_risk", sa.Float(), nullable=True),
        sa.Column("is_active", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- market_prices (TimescaleDB hypertable) ---
    op.create_table(
        "market_prices",
        sa.Column("market_id", sa.String(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("market_id", "timestamp"),
    )
    op.create_index(
        "idx_market_prices_market_time", "market_prices", ["market_id", sa.text("timestamp DESC")]
    )
    # Note: If TimescaleDB is available, convert to hypertable:
    #   SELECT create_hypertable('market_prices', 'timestamp',
    #     chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
    # Fallback: standard table with good indexes is sufficient for MVP scale.

    # --- market_embeddings (pgvector) ---
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("""
        CREATE TABLE market_embeddings (
            market_id TEXT PRIMARY KEY,
            embedding vector(1536) NOT NULL,
            model_name TEXT,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # --- market_entities ---
    op.create_table(
        "market_entities",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("market_id", sa.String(), nullable=False),
        sa.Column("entity_name", sa.String(), nullable=False),
        sa.Column("entity_type", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
    )
    op.create_index("idx_market_entities_market", "market_entities", ["market_id"])
    op.create_index(
        "idx_market_entities_entity", "market_entities", ["entity_name", "entity_type"]
    )

    # --- market_edges ---
    op.create_table(
        "market_edges",
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("target_id", sa.String(), nullable=False),
        sa.Column("edge_type", sa.String(), nullable=False),
        sa.Column("stat_score", sa.Float(), default=0.0),
        sa.Column("logical_score", sa.Float(), default=0.0),
        sa.Column("propagation_score", sa.Float(), default=0.0),
        sa.Column("entity_overlap_score", sa.Float(), default=0.0),
        sa.Column("semantic_score", sa.Float(), default=0.0),
        sa.Column("template_penalty", sa.Float(), default=0.0),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("explanation", JSONB, nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("source_id", "target_id"),
    )
    op.create_index(
        "idx_edges_source_conf",
        "market_edges",
        ["source_id", sa.text("confidence_score DESC")],
    )
    op.create_index(
        "idx_edges_target_conf",
        "market_edges",
        ["target_id", sa.text("confidence_score DESC")],
    )
    op.create_index(
        "idx_edges_type_conf",
        "market_edges",
        ["edge_type", sa.text("confidence_score DESC")],
    )


def downgrade() -> None:
    op.drop_table("market_edges")
    op.drop_table("market_entities")
    op.drop_table("market_embeddings")
    op.drop_table("market_prices")
    op.drop_table("polymarket_events")
    op.drop_table("markets")
