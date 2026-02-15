"""Price snapshot worker: captures current probabilities for active markets."""

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import text

from marketmap.clients.polymarket import PolymarketClient
from marketmap.models.database import SyncSessionLocal
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)


@app.task(bind=True, name="marketmap.workers.price_snapshot.snapshot_prices", max_retries=3)
def snapshot_prices(self) -> dict:  # type: ignore[type-arg]
    """Fetch current prices for all active markets and insert snapshots.

    Strategy:
    1. Query DB for active markets with CLOB token IDs.
    2. For each market, extract the first CLOB token ID (YES outcome).
    3. Fetch midpoint price from CLOB API.
    4. Insert into market_prices time-series table.

    Note: For the MVP, we use the outcomePrices from the Gamma API during
    ingestion as a fast path, and also snapshot from CLOB midpoints for
    higher accuracy on a subset.
    """
    logger.info("Starting price snapshot...")
    start = datetime.now(timezone.utc)
    now = start

    session = SyncSessionLocal()
    try:
        # Get active markets with CLOB token IDs
        result = session.execute(
            text(
                "SELECT id, clob_token_ids, outcome_prices "
                "FROM markets "
                "WHERE is_active = 1.0 AND clob_token_ids IS NOT NULL "
                "ORDER BY volume DESC NULLS LAST "
                "LIMIT 2000"
            )
        )
        rows = result.fetchall()
        logger.info(f"Found {len(rows)} active markets for price snapshot")

        if not rows:
            return {"status": "success", "snapshots": 0, "elapsed_seconds": 0}

        # Strategy: Use outcomePrices from Gamma (already stored) for bulk snapshots,
        # and CLOB midpoints for high-volume markets.
        # For MVP: use outcomePrices as the fast path for all markets.
        snapshots_inserted = 0
        batch_values = []

        for row in rows:
            market_id = row[0]
            outcome_prices_str = row[2]

            if not outcome_prices_str:
                continue

            try:
                outcome_prices = json.loads(outcome_prices_str)
                if outcome_prices and len(outcome_prices) > 0:
                    probability = float(outcome_prices[0])
                    # Skip degenerate values
                    if 0.001 < probability < 0.999:
                        batch_values.append({
                            "market_id": market_id,
                            "timestamp": now,
                            "probability": probability,
                            "volume": None,
                        })
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Bulk insert snapshots
        if batch_values:
            # Use INSERT ... ON CONFLICT DO NOTHING to avoid duplicates
            # within the same timestamp
            for batch_start in range(0, len(batch_values), 500):
                batch = batch_values[batch_start : batch_start + 500]
                insert_sql = text(
                    "INSERT INTO market_prices (market_id, timestamp, probability, volume) "
                    "VALUES (:market_id, :timestamp, :probability, :volume) "
                    "ON CONFLICT (market_id, timestamp) DO UPDATE SET "
                    "probability = EXCLUDED.probability"
                )
                for val in batch:
                    session.execute(insert_sql, val)
                    snapshots_inserted += 1

            session.commit()

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(f"Price snapshot complete: {snapshots_inserted} snapshots in {elapsed:.1f}s")

        return {
            "status": "success",
            "snapshots": snapshots_inserted,
            "markets_checked": len(rows),
            "elapsed_seconds": elapsed,
        }

    except Exception as exc:
        session.rollback()
        logger.exception("Price snapshot failed")
        raise self.retry(exc=exc, countdown=60)
    finally:
        session.close()


@app.task(bind=True, name="marketmap.workers.price_snapshot.snapshot_clob_prices", max_retries=3)
def snapshot_clob_prices(self) -> dict:  # type: ignore[type-arg]
    """Fetch higher-fidelity prices from CLOB midpoints for top-volume markets.

    This is a supplementary task that fetches actual CLOB midpoint prices
    for the highest-volume markets, providing more accurate data than
    the Gamma outcomePrices snapshots.
    """
    logger.info("Starting CLOB price snapshot...")
    start = datetime.now(timezone.utc)
    now = start

    session = SyncSessionLocal()
    try:
        # Get top-volume active markets
        result = session.execute(
            text(
                "SELECT id, clob_token_ids "
                "FROM markets "
                "WHERE is_active = 1.0 AND clob_token_ids IS NOT NULL "
                "ORDER BY volume DESC NULLS LAST "
                "LIMIT 500"
            )
        )
        rows = result.fetchall()

        if not rows:
            return {"status": "success", "snapshots": 0}

        # Extract first CLOB token ID (YES outcome) for each market
        market_tokens: list[tuple[str, str]] = []
        for row in rows:
            market_id = row[0]
            try:
                token_ids = json.loads(row[1])
                if token_ids and len(token_ids) > 0:
                    market_tokens.append((market_id, token_ids[0]))
            except (json.JSONDecodeError, TypeError):
                continue

        # Fetch midpoints from CLOB
        snapshots_inserted = 0
        with PolymarketClient() as client:
            for market_id, token_id in market_tokens:
                try:
                    prices = client.fetch_midpoint_prices([token_id])
                    if token_id in prices:
                        probability = prices[token_id]
                        if 0.001 < probability < 0.999:
                            session.execute(
                                text(
                                    "INSERT INTO market_prices (market_id, timestamp, probability, volume) "
                                    "VALUES (:market_id, :timestamp, :probability, :volume) "
                                    "ON CONFLICT (market_id, timestamp) DO UPDATE SET "
                                    "probability = EXCLUDED.probability"
                                ),
                                {
                                    "market_id": market_id,
                                    "timestamp": now,
                                    "probability": probability,
                                    "volume": None,
                                },
                            )
                            snapshots_inserted += 1
                except Exception:
                    logger.warning(f"Failed CLOB snapshot for market {market_id}")

        session.commit()
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(f"CLOB snapshot complete: {snapshots_inserted} snapshots in {elapsed:.1f}s")

        return {
            "status": "success",
            "snapshots": snapshots_inserted,
            "elapsed_seconds": elapsed,
        }

    except Exception as exc:
        session.rollback()
        logger.exception("CLOB price snapshot failed")
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()
