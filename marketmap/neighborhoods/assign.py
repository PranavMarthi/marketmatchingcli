"""Deterministic neighborhood assignment from event tags."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from sqlalchemy import text

from marketmap.models.database import SyncSessionLocal

STOP_TAGS = {
    "markets",
    "market",
    "other",
    "featured",
    "popular",
    "all",
    "news",
    "hide-from-new",
    "recurring",
    "up-or-down",
    "5m",
    "15m",
    "1h",
    "new",
    "rewards-5-4pt5-50",
}

MACRO_PREFERENCES = {
    "sports": {
        "sports",
        "games",
        "nba",
        "nfl",
        "mlb",
        "nhl",
        "soccer",
        "olympics",
        "tennis",
        "golf",
        "hockey",
        "basketball",
        "ncaa",
        "esports",
    },
    "politics": {
        "politics",
        "election",
        "elections",
        "primary",
        "primaries",
        "trump",
        "biden",
        "senate",
        "house",
        "congress",
        "governor",
    },
    "crypto": {
        "crypto",
        "crypto-prices",
        "bitcoin",
        "ethereum",
        "solana",
        "xrp",
        "ripple",
        "defi",
        "memecoin",
    },
    "economy": {"economy", "finance", "fed", "inflation", "rates", "gdp", "equities"},
    "pop_culture": {"entertainment", "movies", "tv", "music", "celebrity"},
    "geopolitics": {"geopolitics", "war", "russia", "ukraine", "china", "israel"},
    "tech": {"technology", "tech", "ai", "openai", "apple", "google"},
    "science": {"science", "space", "climate", "research"},
    "weather": {"weather", "hurricane", "storm", "temperature"},
}

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _detect_macro_from_tag(tag: str, macro_by_tag: dict[str, str]) -> str | None:
    lower = tag.lower()
    if lower in macro_by_tag:
        return macro_by_tag[lower]
    tokens = TOKEN_RE.findall(lower)
    for token in tokens:
        if token in macro_by_tag:
            return macro_by_tag[token]

    # phrase/substr fallbacks for common polymarket tag shapes
    if "election" in lower or "primary" in lower:
        return "politics"
    if "crypto" in lower or "bitcoin" in lower or "ethereum" in lower or "solana" in lower:
        return "crypto"
    if (
        "sport" in lower
        or "game" in lower
        or "nba" in lower
        or "nfl" in lower
        or "soccer" in lower
        or "ncaa" in lower
    ):
        return "sports"
    if "geo" in lower or "war" in lower:
        return "geopolitics"
    if "econom" in lower or "finance" in lower or "equities" in lower:
        return "economy"
    if "tech" in lower or "ai" in lower:
        return "tech"
    return None


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def _infer_from_text(title: str, category: str) -> tuple[str, str, int]:
    blob = f"{title} {category}".lower()
    if any(t in blob for t in ("nba", "nfl", "mlb", "nhl", "soccer", "olympic")):
        return ("sports", "Sports", 10)
    if any(t in blob for t in ("election", "senate", "president", "trump", "biden")):
        return ("politics", "Politics", 20)
    if any(t in blob for t in ("bitcoin", "ethereum", "crypto", "solana", "token")):
        return ("crypto", "Crypto", 30)
    if any(t in blob for t in ("russia", "ukraine", "china", "israel", "iran", "war")):
        return ("geopolitics", "Geopolitics", 60)
    if any(t in blob for t in ("economy", "inflation", "fed", "rates", "gdp", "stock")):
        return ("economy", "Economy", 40)
    if any(t in blob for t in ("tech", "technology", "ai", "apple", "google")):
        return ("tech", "Tech", 70)
    if any(t in blob for t in ("movie", "music", "celebrity", "entertainment", "tv")):
        return ("pop_culture", "Pop Culture", 50)
    return ("misc", "Misc", 999)


def choose_neighborhood_from_tags(
    tags: list[str],
    title: str,
    category: str,
    doc_freq: Counter[str],
    total_docs: int,
) -> tuple[str, str, int]:
    macro_by_tag: dict[str, str] = {}
    for macro, terms in MACRO_PREFERENCES.items():
        for term in terms:
            macro_by_tag[term] = macro

    rank_by_macro = {
        "sports": 10,
        "politics": 20,
        "crypto": 30,
        "economy": 40,
        "pop_culture": 50,
        "geopolitics": 60,
        "tech": 70,
        "science": 80,
        "weather": 90,
        "misc": 999,
    }

    if not tags:
        return _infer_from_text(title, category)

    candidates: list[tuple[str, str | None, float]] = []
    for tag in tags:
        matched_macro = _detect_macro_from_tag(tag, macro_by_tag)
        idf = math.log((1.0 + total_docs) / (1.0 + doc_freq[tag])) + 1.0
        candidates.append((tag, matched_macro, idf))

    macro_candidates = [item for item in candidates if item[1] is not None]
    pool = macro_candidates if macro_candidates else candidates

    best_tag = None
    best_score = float("-inf")
    best_macro = None
    for tag, matched_macro, idf in pool:
        score = idf
        if score > best_score:
            best_score = score
            best_tag = tag
            best_macro = matched_macro

    macro = best_macro or "misc"
    if macro == "misc":
        inferred_key, inferred_label, inferred_rank = _infer_from_text(title, category)
        if inferred_key != "misc":
            return inferred_key, inferred_label, inferred_rank
    key = macro
    label = macro.replace("_", " ").title()
    rank = rank_by_macro.get(macro, 999)
    return (key, label, rank)


def assign_market_neighborhoods() -> dict[str, int | str]:
    session = SyncSessionLocal()
    try:
        rows = session.execute(
            text(
                """
                SELECT m.id, COALESCE(m.title,''), COALESCE(m.category,''), m.polymarket_event_id,
                       COALESCE(array_agg(DISTINCT pt.tag) FILTER (WHERE pt.tag IS NOT NULL), '{}') AS tags
                FROM markets m
                LEFT JOIN polymarket_event_tags pet ON pet.event_id = m.polymarket_event_id
                LEFT JOIN polymarket_tags pt ON pt.id = pet.tag_id
                WHERE m.is_active = 1.0
                GROUP BY m.id, m.title, m.category, m.polymarket_event_id
                """
            )
        ).fetchall()

        doc_freq: Counter[str] = Counter()
        tags_by_market: dict[str, list[str]] = {}
        for market_id, _, _, _, tags in rows:
            normalized = sorted(
                {
                    str(t).strip().lower()
                    for t in (tags or [])
                    if isinstance(t, str) and str(t).strip().lower() not in STOP_TAGS
                }
            )
            tags_by_market[market_id] = normalized
            for tag in normalized:
                doc_freq[tag] += 1

        total_docs = max(1, len(rows))

        updates = []
        for market_id, title, category, _, _ in rows:
            tags = tags_by_market.get(market_id, [])
            key, label, rank = choose_neighborhood_from_tags(
                tags=tags,
                title=title,
                category=category,
                doc_freq=doc_freq,
                total_docs=total_docs,
            )
            updates.append((market_id, key, label, rank))

        for market_id, key, label, rank in updates:
            session.execute(
                text(
                    """
                    UPDATE markets
                    SET neighborhood_key = :key,
                        neighborhood_label = :label,
                        neighborhood_rank = :rank,
                        updated_at = NOW()
                    WHERE id = :market_id
                    """
                ),
                {"market_id": market_id, "key": key, "label": label, "rank": rank},
            )

        session.commit()
        return {"status": "success", "assigned": len(updates)}
    finally:
        session.close()
