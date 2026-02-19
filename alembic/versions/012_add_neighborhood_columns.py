"""Add neighborhood and hierarchical coordinate columns to markets.

Revision ID: 012
Revises: 011
Create Date: 2026-02-16
"""

from typing import Sequence, Union

from alembic import op

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE markets
          ADD COLUMN IF NOT EXISTS neighborhood_key TEXT NULL,
          ADD COLUMN IF NOT EXISTS neighborhood_label TEXT NULL,
          ADD COLUMN IF NOT EXISTS neighborhood_rank INT NULL,
          ADD COLUMN IF NOT EXISTS local_cluster_id INT NULL,
          ADD COLUMN IF NOT EXISTS local_x DOUBLE PRECISION NULL,
          ADD COLUMN IF NOT EXISTS local_y DOUBLE PRECISION NULL,
          ADD COLUMN IF NOT EXISTS local_z DOUBLE PRECISION NULL,
          ADD COLUMN IF NOT EXISTS global_x DOUBLE PRECISION NULL,
          ADD COLUMN IF NOT EXISTS global_y DOUBLE PRECISION NULL,
          ADD COLUMN IF NOT EXISTS global_z DOUBLE PRECISION NULL,
          ADD COLUMN IF NOT EXISTS local_distortion DOUBLE PRECISION NULL,
          ADD COLUMN IF NOT EXISTS stitch_distortion DOUBLE PRECISION NULL;
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_markets_neighborhood_key ON markets(neighborhood_key);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_markets_local_cluster_id ON markets(local_cluster_id);")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_markets_local_cluster_id;")
    op.execute("DROP INDEX IF EXISTS idx_markets_neighborhood_key;")
    op.execute(
        """
        ALTER TABLE markets
          DROP COLUMN IF EXISTS neighborhood_key,
          DROP COLUMN IF EXISTS neighborhood_label,
          DROP COLUMN IF EXISTS neighborhood_rank,
          DROP COLUMN IF EXISTS local_cluster_id,
          DROP COLUMN IF EXISTS local_x,
          DROP COLUMN IF EXISTS local_y,
          DROP COLUMN IF EXISTS local_z,
          DROP COLUMN IF EXISTS global_x,
          DROP COLUMN IF EXISTS global_y,
          DROP COLUMN IF EXISTS global_z,
          DROP COLUMN IF EXISTS local_distortion,
          DROP COLUMN IF EXISTS stitch_distortion;
        """
    )
