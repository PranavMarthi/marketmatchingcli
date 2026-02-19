"""Embedding service using sentence-transformers (local, free, no API key)."""

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from marketmap.config import settings

logger = logging.getLogger(__name__)

SPORTS_TERMS = {
    "nba",
    "nfl",
    "mlb",
    "nhl",
    "soccer",
    "olympic",
    "olympics",
    "world cup",
    "super bowl",
    "champions league",
    "finals",
    "playoffs",
}
POLITICS_TERMS = {
    "election",
    "president",
    "senate",
    "house",
    "congress",
    "governor",
    "trump",
    "biden",
    "democrat",
    "republican",
}
CRYPTO_TERMS = {
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "solana",
    "crypto",
    "token",
    "airdrop",
    "defi",
    "memecoin",
}
GEOPOLITICS_TERMS = {
    "russia",
    "ukraine",
    "china",
    "taiwan",
    "israel",
    "iran",
    "nato",
    "war",
    "ceasefire",
    "tariff",
    "sanction",
}


def _infer_domain_and_subdomain(title: str, description: str | None, category: str | None) -> tuple[str, str]:
    text_blob = " ".join(
        part.lower()
        for part in [title, description or "", category or ""]
        if part
    )

    if any(term in text_blob for term in SPORTS_TERMS):
        if "nba" in text_blob:
            return ("sports", "basketball")
        if "nfl" in text_blob or "super bowl" in text_blob:
            return ("sports", "american_football")
        if "mlb" in text_blob or "world series" in text_blob:
            return ("sports", "baseball")
        if "nhl" in text_blob or "stanley cup" in text_blob:
            return ("sports", "hockey")
        if "soccer" in text_blob or "world cup" in text_blob or "champions league" in text_blob:
            return ("sports", "soccer")
        if "olympic" in text_blob:
            return ("sports", "olympics")
        return ("sports", "general")

    if any(term in text_blob for term in POLITICS_TERMS):
        if "election" in text_blob:
            return ("politics", "elections")
        if "senate" in text_blob or "house" in text_blob or "congress" in text_blob:
            return ("politics", "legislative")
        return ("politics", "general")

    if any(term in text_blob for term in CRYPTO_TERMS):
        if "bitcoin" in text_blob or "btc" in text_blob:
            return ("crypto", "bitcoin")
        if "ethereum" in text_blob or "eth" in text_blob:
            return ("crypto", "ethereum")
        if "solana" in text_blob:
            return ("crypto", "solana")
        return ("crypto", "general")

    if any(term in text_blob for term in GEOPOLITICS_TERMS):
        return ("geopolitics", "general")

    return ("other", "general")


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
    """Build instruction-prefixed text for semantic clustering embeddings."""
    domain, subdomain = _infer_domain_and_subdomain(title, description, category)
    parts = [
        "Represent this prediction market question for semantic similarity and clustering:",
        f"domain={domain}; subdomain={subdomain};",
        title.strip(),
    ]
    if description:
        desc = description[:500].strip()
        if desc:
            parts.append(desc)
    if category:
        parts.append(f"category={category.strip()}")
    if not settings.embedding_use_domain_hints:
        parts = [
            "Represent this prediction market question for semantic similarity and clustering:",
            title.strip(),
        ] + ([description[:500].strip()] if description and description[:500].strip() else [])
    return " ".join(parts)
