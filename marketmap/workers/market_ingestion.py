"""Market ingestion worker: fetches active markets from Polymarket and upserts to DB."""

import logging
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from marketmap.clients.polymarket import (
    PolymarketClient,
    extract_markets_from_events,
    parse_event_for_db,
    parse_market_for_db,
)
from marketmap.models.database import SyncSessionLocal
from marketmap.models.tables import Market, PolymarketEvent
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)


@app.task(bind=True, name="marketmap.workers.market_ingestion.ingest_markets", max_retries=3)
def ingest_markets(self) -> dict:  # type: ignore[type-arg]
    """Fetch all active events+markets from Polymarket and upsert into DB."""
    logger.info("Starting market ingestion...")
    start = datetime.now(timezone.utc)

    try:
        with PolymarketClient() as client:
            events = client.fetch_all_active_events()

        logger.info(f"Fetched {len(events)} events from Polymarket")

        # Parse events
        event_rows = [parse_event_for_db(e) for e in events]

        # Extract and parse markets from events
        raw_markets = extract_markets_from_events(events)
        market_rows = [parse_market_for_db(m) for m in raw_markets]

        logger.info(f"Parsed {len(market_rows)} markets from {len(event_rows)} events")

        # Upsert into DB
        session = SyncSessionLocal()
        try:
            # Upsert events
            if event_rows:
                _upsert_events(session, event_rows)

            # Upsert markets
            if market_rows:
                _upsert_markets(session, market_rows)

            session.commit()
            logger.info(
                f"Ingestion complete: {len(event_rows)} events, {len(market_rows)} markets"
            )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        return {
            "status": "success",
            "events_count": len(event_rows),
            "markets_count": len(market_rows),
            "elapsed_seconds": elapsed,
        }

    except Exception as exc:
        logger.exception("Market ingestion failed")
        raise self.retry(exc=exc, countdown=60)


def _upsert_events(session, event_rows: list[dict]) -> None:  # type: ignore[type-arg]
    """Bulk upsert events using PostgreSQL ON CONFLICT."""
    for row in event_rows:
        stmt = pg_insert(PolymarketEvent).values(**row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "title": stmt.excluded.title,
                "description": stmt.excluded.description,
                "category": stmt.excluded.category,
                "end_date": stmt.excluded.end_date,
                "liquidity": stmt.excluded.liquidity,
                "volume": stmt.excluded.volume,
                "neg_risk": stmt.excluded.neg_risk,
                "is_active": stmt.excluded.is_active,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        session.execute(stmt)


def _upsert_markets(session, market_rows: list[dict]) -> None:  # type: ignore[type-arg]
    """Bulk upsert markets using PostgreSQL ON CONFLICT."""
    for row in market_rows:
        # Remove internal fields not in DB schema
        db_row = {k: v for k, v in row.items() if not k.startswith("_")}
        stmt = pg_insert(Market).values(**db_row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "title": stmt.excluded.title,
                "description": stmt.excluded.description,
                "category": stmt.excluded.category,
                "close_time": stmt.excluded.close_time,
                "liquidity": stmt.excluded.liquidity,
                "volume": stmt.excluded.volume,
                "outcome_prices": stmt.excluded.outcome_prices,
                "clob_token_ids": stmt.excluded.clob_token_ids,
                "event_id": stmt.excluded.event_id,
                "is_active": stmt.excluded.is_active,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        session.execute(stmt)
