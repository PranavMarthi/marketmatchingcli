"""Add link column to markets table.

Revision ID: 003
Revises: 002
Create Date: 2026-02-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("markets", sa.Column("link", sa.Text(), nullable=True))
    op.create_index("idx_markets_link", "markets", ["link"])


def downgrade() -> None:
    op.drop_index("idx_markets_link", table_name="markets")
    op.drop_column("markets", "link")
