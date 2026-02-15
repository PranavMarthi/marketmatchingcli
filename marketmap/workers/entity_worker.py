"""Entity extraction worker: extracts named entities from market text.

Uses a simple regex + heuristic approach for MVP. Can be upgraded to
spaCy NER or LLM-based extraction later.

Entities support:
- Template penalty computation (high semantic similarity + no entity overlap = penalty)
- Entity overlap scoring for hedge graph edges
- Logical linkage detection (same entity across markets)
"""

import hashlib
import logging
import re
from datetime import datetime, timezone

from sqlalchemy import text

from marketmap.models.database import SyncSessionLocal
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)

# Simple patterns for entity extraction
# These catch common prediction market entity patterns
PERSON_PATTERNS = [
    # "Will [Name] ..." patterns common in prediction markets
    r"(?:Will|Does|Has|Is|Can)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:win|lose|be|become|resign|run|get|make|sign|leave|join|announce|nominate|appoint)",
    # "[Name] to ..." patterns
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+(?:to\s+(?:win|be|become|resign|run))",
    # "next [role]: [Name]"
    r"(?:next|new)\s+\w+(?:\s+\w+)?:\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
]

ORG_PATTERNS = [
    # Known orgs in prediction markets
    r"\b((?:the\s+)?(?:Fed|Federal Reserve|SEC|FBI|CIA|NATO|EU|UN|WHO|IMF|ECB|DOJ|EPA|FTC|SCOTUS|Supreme Court|Congress|Senate|House))\b",
    # Sports teams (common patterns)
    r"\b((?:the\s+)?(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:win|defeat|beat|make|reach))",
    # Company names
    r"\b(Tesla|SpaceX|Apple|Google|Microsoft|Meta|Amazon|Netflix|Nvidia|OpenAI|Anthropic|Bitcoin|Ethereum|Solana)\b",
]

GPE_PATTERNS = [
    # Countries and major regions
    r"\b(United States|US|USA|China|Russia|Ukraine|Iran|Israel|North Korea|Taiwan|India|UK|Japan|Germany|France|Brazil|Mexico|Canada|Australia)\b",
    # US States
    r"\b(California|Texas|Florida|New York|Pennsylvania|Ohio|Georgia|Michigan|Arizona|Nevada|Wisconsin|Minnesota|Virginia|North Carolina)\b",
]

EVENT_PATTERNS = [
    # Elections and political events
    r"\b(\d{4}\s+(?:presidential|midterm|general|primary)\s+election)",
    r"\b(Super Bowl|World Cup|Olympics|World Series|Stanley Cup|NBA Finals|Champions League)\b",
    # Policy/economic events
    r"\b((?:FOMC|Fed)\s+(?:meeting|decision|rate\s+(?:cut|hike|decision)))\b",
    r"\b(government\s+shutdown|debt\s+ceiling|tariff)\b",
]


def extract_entities(title: str, description: str | None = None) -> list[dict[str, str]]:
    """Extract named entities from market text using pattern matching.

    Returns list of dicts with keys: entity_name, entity_type, confidence.
    """
    text_to_search = title
    if description:
        text_to_search += " " + description[:500]

    entities: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(name: str, etype: str, conf: float) -> None:
        # Normalize
        name = name.strip()
        if name.lower().startswith("the "):
            name = name[4:]
        name = name.strip()
        if len(name) < 2 or name.lower() in seen:
            return
        seen.add(name.lower())
        entities.append({
            "entity_name": name,
            "entity_type": etype,
            "confidence": str(conf),
        })

    for pattern in PERSON_PATTERNS:
        for match in re.finditer(pattern, text_to_search):
            _add(match.group(1), "PERSON", 0.7)

    for pattern in ORG_PATTERNS:
        for match in re.finditer(pattern, text_to_search):
            _add(match.group(1), "ORG", 0.8)

    for pattern in GPE_PATTERNS:
        for match in re.finditer(pattern, text_to_search):
            _add(match.group(1), "GPE", 0.9)

    for pattern in EVENT_PATTERNS:
        for match in re.finditer(pattern, text_to_search, re.IGNORECASE):
            _add(match.group(1), "EVENT", 0.8)

    return entities


@app.task(
    bind=True,
    name="marketmap.workers.entity_worker.extract_market_entities",
    max_retries=2,
)
def extract_market_entities(self, batch_limit: int = 10000) -> dict:  # type: ignore[type-arg]
    """Extract entities from markets that haven't been processed yet."""
    logger.info("Starting entity extraction...")
    start = datetime.now(timezone.utc)

    session = SyncSessionLocal()
    try:
        # Find active markets without entities
        result = session.execute(
            text("""
                SELECT m.id, m.title, m.description
                FROM markets m
                LEFT JOIN (
                    SELECT DISTINCT market_id FROM market_entities
                ) me ON m.id = me.market_id
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
        logger.info(f"Found {len(rows)} markets needing entity extraction")

        if not rows:
            return {"status": "success", "markets_processed": 0, "entities_created": 0}

        entities_created = 0
        markets_processed = 0

        for row in rows:
            market_id, title, description = row
            entities = extract_entities(title, description)

            for ent in entities:
                # Create deterministic ID
                ent_id = hashlib.md5(
                    f"{market_id}:{ent['entity_name']}:{ent['entity_type']}".encode()
                ).hexdigest()

                session.execute(
                    text("""
                        INSERT INTO market_entities (id, market_id, entity_name, entity_type, confidence)
                        VALUES (:id, :market_id, :entity_name, :entity_type, :confidence)
                        ON CONFLICT (id) DO UPDATE SET
                            confidence = EXCLUDED.confidence
                    """),
                    {
                        "id": ent_id,
                        "market_id": market_id,
                        "entity_name": ent["entity_name"],
                        "entity_type": ent["entity_type"],
                        "confidence": float(ent["confidence"]),
                    },
                )
                entities_created += 1

            # Insert a sentinel so we don't re-process markets with no entities
            if not entities:
                ent_id = hashlib.md5(f"{market_id}:__none__:NONE".encode()).hexdigest()
                session.execute(
                    text("""
                        INSERT INTO market_entities (id, market_id, entity_name, entity_type, confidence)
                        VALUES (:id, :market_id, '__none__', 'NONE', 0.0)
                        ON CONFLICT (id) DO NOTHING
                    """),
                    {"id": ent_id, "market_id": market_id},
                )

            markets_processed += 1

            if markets_processed % 1000 == 0:
                session.commit()
                logger.info(f"  Processed {markets_processed} markets, {entities_created} entities")

        session.commit()
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(
            f"Entity extraction complete: {entities_created} entities "
            f"from {markets_processed} markets in {elapsed:.1f}s"
        )

        return {
            "status": "success",
            "markets_processed": markets_processed,
            "entities_created": entities_created,
            "elapsed_seconds": elapsed,
        }

    except Exception as exc:
        session.rollback()
        logger.exception("Entity extraction failed")
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()
