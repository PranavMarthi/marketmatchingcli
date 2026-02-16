"""Add per-market projection distortion metrics.

Revision ID: 007
Revises: 006
Create Date: 2026-02-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "market_projection_distortion",
        sa.Column("market_id", sa.Text(), nullable=False),
        sa.Column("projection_version", sa.Text(), nullable=False),
        sa.Column("distortion_version", sa.Text(), nullable=False),
        sa.Column("neighbor_k", sa.Integer(), nullable=False),
        sa.Column("distortion_score", sa.Float(), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint(
            "market_id", "projection_version", name="market_projection_distortion_pkey"
        ),
    )
    op.create_index(
        "idx_market_projection_distortion_proj",
        "market_projection_distortion",
        ["projection_version"],
    )
    op.create_index(
        "idx_market_projection_distortion_market_proj",
        "market_projection_distortion",
        ["market_id", "projection_version"],
    )
    op.create_index(
        "idx_market_projection_distortion_version",
        "market_projection_distortion",
        ["distortion_version"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_market_projection_distortion_version",
        table_name="market_projection_distortion",
    )
    op.drop_index(
        "idx_market_projection_distortion_market_proj",
        table_name="market_projection_distortion",
    )
    op.drop_index(
        "idx_market_projection_distortion_proj",
        table_name="market_projection_distortion",
    )
    op.drop_table("market_projection_distortion")
