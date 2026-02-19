"""Discovery graph worker: computes Top-K semantic neighbors and stores discovery edges."""

import logging
import json
from datetime import datetime, timezone

from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)
DISCOVERY_LOCK_KEY = 913_441


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
        locked = bool(
            session.execute(
                text("SELECT pg_try_advisory_lock(:key)"),
                {"key": DISCOVERY_LOCK_KEY},
            ).scalar()
        )
        if not locked:
            return {
                "status": "skipped",
                "reason": "discovery_edges_job_already_running",
            }

        # Materialize top active source markets for deterministic processing.
        session.execute(text("DROP TABLE IF EXISTS tmp_discovery_sources"))
        session.execute(
            text(
                """
                CREATE TEMP TABLE tmp_discovery_sources AS
                SELECT me.market_id,
                       me.embedding,
                       COALESCE(NULLIF(m.neighborhood_key, ''), 'misc::unknown') AS neighborhood_key
                FROM market_embeddings me
                JOIN markets m ON m.id = me.market_id
                WHERE m.is_active = 1.0
                ORDER BY m.volume DESC NULLS LAST, me.market_id ASC
                LIMIT :limit
                """
            ),
            {"limit": batch_limit},
        )
        session.execute(
            text("CREATE INDEX IF NOT EXISTS idx_tmp_discovery_sources_market_id ON tmp_discovery_sources (market_id)")
        )

        processed = int(
            session.execute(text("SELECT COUNT(*) FROM tmp_discovery_sources")).scalar() or 0
        )
        logger.info("Processing %s markets for discovery edges", processed)

        if processed == 0:
            session.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": DISCOVERY_LOCK_KEY})
            session.commit()
            return {"status": "success", "markets_processed": 0, "edges_created": 0, "elapsed_seconds": 0}

        session.execute(
            text(
                """
                DELETE FROM market_edges me
                USING tmp_discovery_sources s
                WHERE me.edge_type = 'discovery'
                  AND (me.source_id = s.market_id OR me.target_id = s.market_id)
                """
            )
        )

        session.execute(
            text(
                """
                WITH raw_neighbors AS (
                    SELECT
                        s.market_id AS source_id,
                        n.market_id AS target_id,
                        1 - (s.embedding <=> n.embedding) AS similarity
                    FROM tmp_discovery_sources s
                    CROSS JOIN LATERAL (
                        SELECT me2.market_id, me2.embedding
                        FROM market_embeddings me2
                        JOIN markets m2 ON m2.id = me2.market_id
                        WHERE m2.is_active = 1.0
                          AND COALESCE(NULLIF(m2.neighborhood_key, ''), 'misc::unknown') = s.neighborhood_key
                          AND me2.market_id != s.market_id
                        ORDER BY s.embedding <=> me2.embedding ASC
                        LIMIT :top_k
                    ) AS n
                    WHERE 1 - (s.embedding <=> n.embedding) >= :min_sim
                ),
                dedup AS (
                    SELECT
                        LEAST(source_id, target_id) AS src,
                        GREATEST(source_id, target_id) AS tgt,
                        MAX(similarity) AS similarity
                    FROM raw_neighbors
                    GROUP BY LEAST(source_id, target_id), GREATEST(source_id, target_id)
                )
                INSERT INTO market_edges
                    (source_id, target_id, edge_type, semantic_score, confidence_score, explanation, updated_at)
                SELECT
                    d.src,
                    d.tgt,
                    'discovery',
                    d.similarity,
                    d.similarity,
                    CAST(:explanation AS jsonb),
                    NOW()
                FROM dedup d
                ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                    semantic_score = GREATEST(market_edges.semantic_score, EXCLUDED.semantic_score),
                    confidence_score = GREATEST(market_edges.confidence_score, EXCLUDED.confidence_score),
                    explanation = EXCLUDED.explanation,
                    updated_at = NOW()
                """
            ),
            {
                "top_k": top_k,
                "min_sim": min_sim,
                "explanation": json.dumps(
                    {
                        "type": "semantic",
                        "scope": "within_neighborhood",
                        "model": settings.embedding_model,
                    },
                    separators=(",", ":"),
                ),
            },
        )

        if settings.discovery_cross_neighborhood_edges_enabled:
            session.execute(
                text(
                    """
                    WITH source_neighborhood AS (
                        SELECT market_id, neighborhood_key, embedding
                        FROM tmp_discovery_sources
                    ),
                    cross_neighbors AS (
                        SELECT
                            s.market_id AS source_id,
                            n.market_id AS target_id,
                            1 - (s.embedding <=> n.embedding) AS similarity
                        FROM source_neighborhood s
                        CROSS JOIN LATERAL (
                            SELECT me2.market_id, me2.embedding
                            FROM market_embeddings me2
                            JOIN markets m2 ON m2.id = me2.market_id
                            WHERE m2.is_active = 1.0
                              AND COALESCE(NULLIF(m2.neighborhood_key, ''), 'misc::unknown') != s.neighborhood_key
                            ORDER BY s.embedding <=> me2.embedding ASC
                            LIMIT :cross_top_k
                        ) n
                        WHERE 1 - (s.embedding <=> n.embedding) >= :cross_min_sim
                    ),
                    dedup AS (
                        SELECT
                            LEAST(source_id, target_id) AS src,
                            GREATEST(source_id, target_id) AS tgt,
                            MAX(similarity) AS similarity
                        FROM cross_neighbors
                        GROUP BY LEAST(source_id, target_id), GREATEST(source_id, target_id)
                    )
                    INSERT INTO market_edges
                        (source_id, target_id, edge_type, semantic_score, confidence_score, explanation, updated_at)
                    SELECT
                        d.src,
                        d.tgt,
                        'discovery',
                        d.similarity,
                        d.similarity,
                        CAST(:cross_explanation AS jsonb),
                        NOW()
                    FROM dedup d
                    ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                        semantic_score = GREATEST(market_edges.semantic_score, EXCLUDED.semantic_score),
                        confidence_score = GREATEST(market_edges.confidence_score, EXCLUDED.confidence_score),
                        explanation = EXCLUDED.explanation,
                        updated_at = NOW()
                    """
                ),
                {
                    "cross_top_k": settings.discovery_cross_neighborhood_top_k,
                    "cross_min_sim": settings.discovery_cross_neighborhood_min_similarity,
                    "cross_explanation": json.dumps(
                        {
                            "type": "semantic",
                            "scope": "cross_neighborhood",
                            "model": settings.embedding_model,
                        },
                        separators=(",", ":"),
                    ),
                },
            )

        edges_created = int(
            session.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM market_edges me
                    WHERE me.edge_type = 'discovery'
                      AND EXISTS (
                          SELECT 1
                          FROM tmp_discovery_sources s
                          WHERE me.source_id = s.market_id OR me.target_id = s.market_id
                      )
                    """
                )
            ).scalar()
            or 0
        )
        session.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": DISCOVERY_LOCK_KEY})
        session.commit()

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
        try:
            session.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": DISCOVERY_LOCK_KEY})
            session.commit()
        except Exception:
            session.rollback()
        session.rollback()
        logger.exception("Discovery edge computation failed")
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()
