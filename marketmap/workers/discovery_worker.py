"""Discovery graph worker: computes Top-K semantic neighbors and stores discovery edges."""

import logging
from datetime import datetime, timezone

from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)


@app.task(
    bind=True,
    name="marketmap.workers.discovery_worker.compute_discovery_edges",
    max_retries=2,
)
def compute_discovery_edges(self, batch_limit: int = 5000) -> dict:  # type: ignore[type-arg]
    """Compute Top-K semantic neighbors for each market and store as discovery edges.

    Uses pgvector's cosine distance operator (<=> for cosine) to find
    the K nearest neighbors for each embedded market.

    Strategy:
    - Process markets in batches
    - For each market, find top-K nearest neighbors using pgvector
    - Compute confidence = semantic_similarity (for discovery graph)
    - Store edges above min_similarity threshold
    - Delete stale edges for reprocessed markets
    """
    logger.info("Starting discovery edge computation...")
    start = datetime.now(timezone.utc)

    top_k = settings.discovery_top_k_neighbors
    min_sim = settings.discovery_min_similarity

    session = SyncSessionLocal()
    try:
        # Get all markets with embeddings, ordered by volume
        result = session.execute(
            text("""
                SELECT me.market_id
                FROM market_embeddings me
                JOIN markets m ON m.id = me.market_id
                WHERE m.is_active = 1.0
                ORDER BY m.volume DESC NULLS LAST
                LIMIT :limit
            """),
            {"limit": batch_limit},
        )
        market_ids = [row[0] for row in result.fetchall()]
        logger.info(f"Processing {len(market_ids)} markets for discovery edges")

        if not market_ids:
            return {"status": "success", "edges_created": 0, "elapsed_seconds": 0}

        # Delete existing discovery edges for markets we're about to reprocess
        # (do in batches to avoid huge transactions)
        for i in range(0, len(market_ids), 1000):
            batch = market_ids[i : i + 1000]
            session.execute(
                text("""
                    DELETE FROM market_edges
                    WHERE edge_type = 'discovery'
                      AND (source_id = ANY(:ids) OR target_id = ANY(:ids))
                """),
                {"ids": batch},
            )
        session.commit()

        # For each market, find top-K neighbors using pgvector
        edges_created = 0
        processed = 0

        for i in range(0, len(market_ids), 100):
            batch = market_ids[i : i + 100]

            for market_id in batch:
                # pgvector cosine distance: 1 - cosine_similarity
                # So similarity = 1 - distance
                result = session.execute(
                    text("""
                        SELECT
                            me2.market_id,
                            1 - (me1.embedding <=> me2.embedding) AS similarity
                        FROM market_embeddings me1
                        JOIN market_embeddings me2 ON me1.market_id != me2.market_id
                        JOIN markets m2 ON m2.id = me2.market_id AND m2.is_active = 1.0
                        WHERE me1.market_id = :market_id
                          AND 1 - (me1.embedding <=> me2.embedding) >= :min_sim
                        ORDER BY me1.embedding <=> me2.embedding ASC
                        LIMIT :top_k
                    """),
                    {"market_id": market_id, "min_sim": min_sim, "top_k": top_k},
                )
                neighbors = result.fetchall()

                for neighbor_id, similarity in neighbors:
                    # Ensure consistent edge direction (lower id -> higher id)
                    # to avoid duplicate edges
                    src = min(market_id, neighbor_id)
                    tgt = max(market_id, neighbor_id)

                    session.execute(
                        text("""
                            INSERT INTO market_edges
                                (source_id, target_id, edge_type, semantic_score,
                                 confidence_score, explanation, updated_at)
                            VALUES
                                (:src, :tgt, 'discovery', :sim,
                                 :conf, :explanation, NOW())
                            ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                                semantic_score = GREATEST(market_edges.semantic_score, EXCLUDED.semantic_score),
                                confidence_score = GREATEST(market_edges.confidence_score, EXCLUDED.confidence_score),
                                updated_at = NOW()
                        """),
                        {
                            "src": src,
                            "tgt": tgt,
                            "sim": float(similarity),
                            "conf": float(similarity),  # For discovery, confidence = similarity
                            "explanation": '{"type": "semantic", "model": "all-MiniLM-L6-v2"}',
                        },
                    )
                    edges_created += 1

                processed += 1

            session.commit()
            logger.info(
                f"  Batch {i // 100 + 1}: processed {processed} markets, "
                f"{edges_created} edges so far"
            )

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(
            f"Discovery edges complete: {edges_created} edges for "
            f"{processed} markets in {elapsed:.1f}s"
        )

        return {
            "status": "success",
            "markets_processed": processed,
            "edges_created": edges_created,
            "elapsed_seconds": elapsed,
        }

    except Exception as exc:
        session.rollback()
        logger.exception("Discovery edge computation failed")
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()
