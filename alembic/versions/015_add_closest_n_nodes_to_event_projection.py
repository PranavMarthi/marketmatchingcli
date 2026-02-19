"""Add closest_n_nodes to event projection table.

Revision ID: 015
Revises: 014
Create Date: 2026-02-19
"""

from typing import Sequence, Union

from alembic import op

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE event_projection_3d
        ADD COLUMN IF NOT EXISTS closest_n_nodes TEXT[] NOT NULL DEFAULT ARRAY[]::text[];
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_event_projection_closest_n_nodes_gin ON event_projection_3d USING GIN (closest_n_nodes);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_event_projection_closest_n_nodes_gin;")
    op.execute("ALTER TABLE event_projection_3d DROP COLUMN IF EXISTS closest_n_nodes;")
