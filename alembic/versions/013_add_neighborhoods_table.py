"""Add neighborhoods table for hierarchical global anchors.

Revision ID: 013
Revises: 012
Create Date: 2026-02-16
"""

from typing import Sequence, Union

from alembic import op

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS neighborhoods (
          neighborhood_key TEXT PRIMARY KEY,
          label TEXT NOT NULL,
          market_count INT NOT NULL DEFAULT 0,
          anchor_x DOUBLE PRECISION NULL,
          anchor_y DOUBLE PRECISION NULL,
          anchor_z DOUBLE PRECISION NULL,
          scale DOUBLE PRECISION NOT NULL DEFAULT 1.0,
          centroid_vector VECTOR(1024) NULL,
          meta JSONB NOT NULL DEFAULT '{}'::jsonb,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_neighborhoods_market_count ON neighborhoods(market_count DESC);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_neighborhoods_market_count;")
    op.execute("DROP TABLE IF EXISTS neighborhoods;")
