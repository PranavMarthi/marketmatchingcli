"""Load markets from allqslinks.txt into the database.

Parses each line as "question, https://polymarket.com/market/slug"
and creates market records. Uses the slug from the URL as the market ID.
Attempts to match with existing Polymarket data already ingested for enrichment.
"""

import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from marketmap.models.database import SyncSessionLocal


def parse_allqslinks(filepath: str) -> list[dict]:
    """Parse allqslinks.txt into list of {question, link, slug}."""
    records = []
    seen_slugs = set()

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx = line.rfind(", https://polymarket.com/")
            if idx == -1:
                continue
            question = line[:idx].strip()
            link = line[idx + 2:].strip()  # skip ", "

            # Extract slug from URL: https://polymarket.com/market/<slug>
            slug = link.split("/market/")[-1] if "/market/" in link else None
            if not slug:
                continue

            # Use slug as ID (dedup on slug)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            records.append({
                "id": slug,
                "slug": slug,
                "title": question,
                "link": link,
            })

    return records


def load_markets(records: list[dict]) -> None:
    """Insert markets from allqslinks.txt, enriching from existing Polymarket data."""
    session = SyncSessionLocal()
    try:
        # First, build a lookup of existing markets by slug for enrichment
        print("Building slug lookup from existing markets...")
        result = session.execute(text(
            "SELECT slug, category, close_time, liquidity, volume, "
            "outcome_prices, clob_token_ids, event_id, description "
            "FROM markets WHERE slug IS NOT NULL"
        ))
        slug_lookup = {}
        for row in result.fetchall():
            if row[0]:
                slug_lookup[row[0]] = {
                    "category": row[1],
                    "close_time": row[2],
                    "liquidity": row[3],
                    "volume": row[4],
                    "outcome_prices": row[5],
                    "clob_token_ids": row[6],
                    "event_id": row[7],
                    "description": row[8],
                }
        print(f"  Found {len(slug_lookup)} existing markets for enrichment")

        # Clear existing edges and embeddings/entities (will recompute)
        print("Clearing dependent tables...")
        session.execute(text("DELETE FROM market_edges"))
        session.execute(text("DELETE FROM market_embeddings"))
        session.execute(text("DELETE FROM market_entities"))
        session.execute(text("DELETE FROM market_prices"))

        # Clear and reload markets
        print("Clearing markets table...")
        session.execute(text("DELETE FROM markets"))
        session.commit()

        # Insert new markets
        print(f"Inserting {len(records)} markets from allqslinks.txt...")
        inserted = 0
        for rec in records:
            enrichment = slug_lookup.get(rec["slug"], {})

            session.execute(
                text("""
                    INSERT INTO markets (id, slug, title, link, description, category,
                        close_time, liquidity, volume, outcome_prices, clob_token_ids,
                        event_id, is_active, created_at, updated_at)
                    VALUES (:id, :slug, :title, :link, :description, :category,
                        :close_time, :liquidity, :volume, :outcome_prices, :clob_token_ids,
                        :event_id, 1.0, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        link = EXCLUDED.link,
                        updated_at = NOW()
                """),
                {
                    "id": rec["id"],
                    "slug": rec["slug"],
                    "title": rec["title"],
                    "link": rec["link"],
                    "description": enrichment.get("description"),
                    "category": enrichment.get("category", ""),
                    "close_time": enrichment.get("close_time"),
                    "liquidity": enrichment.get("liquidity"),
                    "volume": enrichment.get("volume"),
                    "outcome_prices": enrichment.get("outcome_prices"),
                    "clob_token_ids": enrichment.get("clob_token_ids"),
                    "event_id": enrichment.get("event_id"),
                },
            )
            inserted += 1
            if inserted % 2000 == 0:
                session.commit()
                print(f"  Inserted {inserted}...")

        session.commit()

        # Count enriched
        result = session.execute(text(
            "SELECT COUNT(*) FROM markets WHERE volume IS NOT NULL AND volume > 0"
        ))
        enriched = result.scalar()
        print(f"\nDone: {inserted} markets inserted, {enriched} enriched with Polymarket data")

    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "allqslinks.txt")
    records = parse_allqslinks(filepath)
    print(f"Parsed {len(records)} unique markets from allqslinks.txt")
    load_markets(records)
