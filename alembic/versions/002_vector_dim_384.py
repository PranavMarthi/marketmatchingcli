"""Change embedding dimension from 1536 to 384 for sentence-transformers.

Revision ID: 002
Revises: 001
Create Date: 2026-02-14
"""
from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop and recreate with new dimension (table is empty)
    op.execute("DROP TABLE IF EXISTS market_embeddings;")
    op.execute("""
        CREATE TABLE market_embeddings (
            market_id TEXT PRIMARY KEY,
            embedding vector(384) NOT NULL,
            model_name TEXT DEFAULT 'all-MiniLM-L6-v2',
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    # Create HNSW index for fast approximate nearest neighbor search
    op.execute("""
        CREATE INDEX idx_embeddings_hnsw ON market_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS market_embeddings;")
    op.execute("""
        CREATE TABLE market_embeddings (
            market_id TEXT PRIMARY KEY,
            embedding vector(1536) NOT NULL,
            model_name TEXT,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
