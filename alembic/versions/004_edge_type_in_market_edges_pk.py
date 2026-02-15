"""Allow discovery + hedge edges for same pair.

Revision ID: 004
Revises: 003
Create Date: 2026-02-14
"""

from typing import Sequence, Union

from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint("market_edges_pkey", "market_edges", type_="primary")
    op.create_primary_key(
        "market_edges_pkey",
        "market_edges",
        ["source_id", "target_id", "edge_type"],
    )


def downgrade() -> None:
    op.drop_constraint("market_edges_pkey", "market_edges", type_="primary")
    op.create_primary_key("market_edges_pkey", "market_edges", ["source_id", "target_id"])
