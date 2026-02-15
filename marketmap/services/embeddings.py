"""Embedding service using sentence-transformers (local, free, no API key)."""

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from marketmap.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model (cached singleton)."""
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def compute_embeddings(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    """Compute embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for encoding. Defaults to config setting.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.array([])

    model = get_model()
    bs = batch_size or settings.embedding_batch_size

    embeddings = model.encode(
        texts,
        batch_size=bs,
        show_progress_bar=len(texts) > 1000,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    return embeddings


def build_market_text(title: str, description: str | None = None, category: str | None = None) -> str:
    """Build the text representation for a market to embed.

    Combines title + truncated description + category for a rich embedding.
    """
    parts = [title]
    if description:
        # Truncate description to ~500 chars to keep embedding focused
        desc = description[:500].strip()
        if desc:
            parts.append(desc)
    if category:
        parts.append(f"[{category}]")
    return " ".join(parts)
