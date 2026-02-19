"""Add event embedding and event projection tables.

Revision ID: 014
Revises: 013
Create Date: 2026-02-17
"""

from typing import Sequence, Union

from alembic import op

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE polymarket_events
          ADD COLUMN IF NOT EXISTS neighborhood_key TEXT NULL,
          ADD COLUMN IF NOT EXISTS neighborhood_label TEXT NULL,
          ADD COLUMN IF NOT EXISTS neighborhood_rank INT NULL;
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_polymarket_events_neighborhood_key ON polymarket_events(neighborhood_key);"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS event_embeddings (
          event_id TEXT PRIMARY KEY REFERENCES polymarket_events(id) ON DELETE CASCADE,
          embedding vector(1024) NOT NULL,
          model_name TEXT DEFAULT 'BAAI/bge-large-en-v1.5',
          updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_event_embeddings_hnsw ON event_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS event_projection_3d (
          event_id TEXT PRIMARY KEY REFERENCES polymarket_events(id) ON DELETE CASCADE,
          x DOUBLE PRECISION NOT NULL,
          y DOUBLE PRECISION NOT NULL,
          z DOUBLE PRECISION NOT NULL,
          projection_version TEXT NOT NULL,
          embedding_version TEXT NOT NULL,
          neighborhood_key TEXT NULL,
          neighborhood_label TEXT NULL,
          local_cluster_id INT NULL,
          local_distortion DOUBLE PRECISION NULL,
          stitch_distortion DOUBLE PRECISION NULL,
          updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_event_projection_version ON event_projection_3d(projection_version);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_event_projection_neighborhood ON event_projection_3d(neighborhood_key);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_event_projection_neighborhood;")
    op.execute("DROP INDEX IF EXISTS idx_event_projection_version;")
    op.execute("DROP TABLE IF EXISTS event_projection_3d;")
    op.execute("DROP INDEX IF EXISTS idx_event_embeddings_hnsw;")
    op.execute("DROP TABLE IF EXISTS event_embeddings;")
    op.execute("DROP INDEX IF EXISTS idx_polymarket_events_neighborhood_key;")
    op.execute(
        """
        ALTER TABLE polymarket_events
          DROP COLUMN IF EXISTS neighborhood_key,
          DROP COLUMN IF EXISTS neighborhood_label,
          DROP COLUMN IF EXISTS neighborhood_rank;
        """
    )
