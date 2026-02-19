"""Add polymarket event tags tables and market event link.

Revision ID: 011
Revises: 010
Create Date: 2026-02-16
"""

from typing import Sequence, Union

from alembic import op

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_tags (
            id BIGSERIAL PRIMARY KEY,
            tag TEXT NOT NULL UNIQUE
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_event_tags (
            event_id TEXT NOT NULL REFERENCES polymarket_events(id) ON DELETE CASCADE,
            tag_id BIGINT NOT NULL REFERENCES polymarket_tags(id) ON DELETE CASCADE,
            PRIMARY KEY (event_id, tag_id)
        );
        """
    )
    op.execute("ALTER TABLE markets ADD COLUMN IF NOT EXISTS polymarket_event_id TEXT NULL;")
    op.execute("ALTER TABLE polymarket_events ADD COLUMN IF NOT EXISTS raw JSONB NOT NULL DEFAULT '{}'::jsonb;")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_markets_polymarket_event_id ON markets(polymarket_event_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_event_tags_event_id ON polymarket_event_tags(event_id);"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_event_tags_tag_id ON polymarket_event_tags(tag_id);")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_event_tags_tag_id;")
    op.execute("DROP INDEX IF EXISTS idx_event_tags_event_id;")
    op.execute("DROP INDEX IF EXISTS idx_markets_polymarket_event_id;")
    op.execute("ALTER TABLE markets DROP COLUMN IF EXISTS polymarket_event_id;")
    op.execute("ALTER TABLE polymarket_events DROP COLUMN IF EXISTS raw;")
    op.execute("DROP TABLE IF EXISTS polymarket_event_tags;")
    op.execute("DROP TABLE IF EXISTS polymarket_tags;")
