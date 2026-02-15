"""Embedding worker: computes text embeddings for markets missing embeddings."""

import logging
from datetime import datetime, timezone

from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal
from marketmap.services.embeddings import build_market_text, compute_embeddings
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)


@app.task(bind=True, name="marketmap.workers.embedding_worker.compute_market_embeddings", max_retries=2)
def compute_market_embeddings(self, batch_limit: int = 5000) -> dict:  # type: ignore[type-arg]
    """Compute embeddings for markets that don't have one yet.

    Fetches active markets without embeddings, computes embeddings in batch,
    and upserts them into market_embeddings.
    """
    logger.info("Starting embedding computation...")
    start = datetime.now(timezone.utc)

    session = SyncSessionLocal()
    try:
        # Find active markets missing embeddings
        result = session.execute(
            text("""
                SELECT m.id, m.title, m.description, m.category
                FROM markets m
                LEFT JOIN market_embeddings me ON m.id = me.market_id
                WHERE m.is_active = 1.0
                  AND m.title IS NOT NULL
                  AND m.title != ''
                  AND me.market_id IS NULL
                ORDER BY m.volume DESC NULLS LAST
                LIMIT :limit
            """),
            {"limit": batch_limit},
        )
        rows = result.fetchall()
        logger.info(f"Found {len(rows)} markets needing embeddings")

        if not rows:
            return {"status": "success", "embedded": 0, "elapsed_seconds": 0}

        # Build texts
        market_ids = []
        texts = []
        for row in rows:
            market_id, title, description, category = row
            market_ids.append(market_id)
            texts.append(build_market_text(title, description, category))

        # Compute embeddings
        logger.info(f"Computing embeddings for {len(texts)} markets...")
        embeddings = compute_embeddings(texts)
        logger.info(f"Embeddings computed: shape={embeddings.shape}")

        # Upsert into DB in batches
        inserted = 0
        batch_size = 500
        for i in range(0, len(market_ids), batch_size):
            batch_ids = market_ids[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size]

            for mid, emb in zip(batch_ids, batch_embs):
                emb_str = "[" + ",".join(str(float(x)) for x in emb) + "]"
                session.execute(
                    text("""
                        INSERT INTO market_embeddings (market_id, embedding, model_name, updated_at)
                        VALUES (:market_id, :embedding, :model_name, NOW())
                        ON CONFLICT (market_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            model_name = EXCLUDED.model_name,
                            updated_at = NOW()
                    """),
                    {
                        "market_id": mid,
                        "embedding": emb_str,
                        "model_name": settings.embedding_model,
                    },
                )
                inserted += 1

            session.commit()
            logger.info(f"  Inserted batch {i // batch_size + 1}: {inserted} total")

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(f"Embedding computation complete: {inserted} markets in {elapsed:.1f}s")

        return {
            "status": "success",
            "embedded": inserted,
            "elapsed_seconds": elapsed,
        }

    except Exception as exc:
        session.rollback()
        logger.exception("Embedding computation failed")
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()


@app.task(bind=True, name="marketmap.workers.embedding_worker.recompute_all_embeddings", max_retries=1)
def recompute_all_embeddings(self) -> dict:  # type: ignore[type-arg]
    """Force recompute all embeddings (e.g., after model change)."""
    logger.info("Clearing all embeddings for recomputation...")
    session = SyncSessionLocal()
    try:
        session.execute(text("DELETE FROM market_embeddings"))
        session.commit()
    finally:
        session.close()

    return compute_market_embeddings(batch_limit=50000)
