"""Add 3D projection and optional tile cache tables.

Revision ID: 005
Revises: 004
Create Date: 2026-02-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "market_projection_3d",
        sa.Column("market_id", sa.Text(), primary_key=True),
        sa.Column("x", sa.Float(), nullable=False),
        sa.Column("y", sa.Float(), nullable=False),
        sa.Column("z", sa.Float(), nullable=False),
        sa.Column("projection_version", sa.Text(), nullable=False),
        sa.Column("embedding_version", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_projection_xyz", "market_projection_3d", ["x", "y", "z"])
    op.create_index(
        "idx_projection_version_market",
        "market_projection_3d",
        ["projection_version", "market_id"],
    )

    op.create_table(
        "discovery_tiles_cache",
        sa.Column("tile_key", sa.Text(), primary_key=True),
        sa.Column("projection_version", sa.Text(), nullable=False),
        sa.Column("payload_json", JSONB, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "idx_discovery_tiles_projection",
        "discovery_tiles_cache",
        ["projection_version"],
    )


def downgrade() -> None:
    op.drop_index("idx_discovery_tiles_projection", table_name="discovery_tiles_cache")
    op.drop_table("discovery_tiles_cache")

    op.drop_index("idx_projection_version_market", table_name="market_projection_3d")
    op.drop_index("idx_projection_xyz", table_name="market_projection_3d")
    op.drop_table("market_projection_3d")
