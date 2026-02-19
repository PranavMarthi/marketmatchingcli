"""Add cached stage-1 manifold table for two-stage UMAP.

Revision ID: 010
Revises: 009
Create Date: 2026-02-16
"""

from typing import Sequence, Union

from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE market_projection_manifold (
            market_id TEXT PRIMARY KEY,
            manifold_version TEXT NOT NULL,
            embedding_version TEXT NOT NULL,
            manifold vector(15) NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute(
        """
        CREATE INDEX idx_projection_manifold_version_market
        ON market_projection_manifold (manifold_version, market_id);
        """
    )
    op.execute(
        """
        CREATE INDEX idx_projection_manifold_embedding
        ON market_projection_manifold (embedding_version);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS market_projection_manifold;")
