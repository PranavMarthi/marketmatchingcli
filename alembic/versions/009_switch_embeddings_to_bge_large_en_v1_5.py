"""Switch embedding storage to BGE-large dimensions.

Revision ID: 009
Revises: 008
Create Date: 2026-02-16
"""

from typing import Sequence, Union

from alembic import op

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS market_embeddings;")
    op.execute(
        """
        CREATE TABLE market_embeddings (
            market_id TEXT PRIMARY KEY,
            embedding vector(1024) NOT NULL,
            model_name TEXT DEFAULT 'BAAI/bge-large-en-v1.5',
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute(
        """
        CREATE INDEX idx_embeddings_hnsw ON market_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS market_embeddings;")
    op.execute(
        """
        CREATE TABLE market_embeddings (
            market_id TEXT PRIMARY KEY,
            embedding vector(64) NOT NULL,
            model_name TEXT DEFAULT 'minishlab/potion-base-2M',
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute(
        """
        CREATE INDEX idx_embeddings_hnsw ON market_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )
