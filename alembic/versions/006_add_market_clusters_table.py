"""Add persisted discovery cluster assignments.

Revision ID: 006
Revises: 005
Create Date: 2026-02-15
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "market_clusters",
        sa.Column("market_id", sa.Text(), nullable=False),
        sa.Column("projection_version", sa.Text(), nullable=False),
        sa.Column("clustering_version", sa.Text(), nullable=False),
        sa.Column("cluster_id", sa.Text(), nullable=False),
        sa.Column("cluster_size", sa.Float(), nullable=False),
        sa.Column("algorithm", sa.Text(), nullable=False, server_default="louvain"),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("market_id", "projection_version", name="market_clusters_pkey"),
    )
    op.create_index(
        "idx_market_clusters_proj_cluster",
        "market_clusters",
        ["projection_version", "cluster_id"],
    )
    op.create_index(
        "idx_market_clusters_market_proj",
        "market_clusters",
        ["market_id", "projection_version"],
    )
    op.create_index(
        "idx_market_clusters_clustering_version",
        "market_clusters",
        ["clustering_version"],
    )


def downgrade() -> None:
    op.drop_index("idx_market_clusters_clustering_version", table_name="market_clusters")
    op.drop_index("idx_market_clusters_market_proj", table_name="market_clusters")
    op.drop_index("idx_market_clusters_proj_cluster", table_name="market_clusters")
    op.drop_table("market_clusters")
