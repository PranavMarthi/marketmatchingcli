"""Ingest Polymarket event tags from a JSON snapshot file.

Usage:
    python -m marketmap.ingest.polymarket_event_tags --path all_active_polymarket_events.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal

SPACE_RE = re.compile(r"\s+")


def _normalize_tag(tag: str) -> str:
    cleaned = SPACE_RE.sub(" ", tag.strip().lower())
    return cleaned


def _extract_tags(event: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("tags", "tag", "topics", "topic", "category"):
        raw = event.get(key)
        if isinstance(raw, str):
            candidates.append(raw)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    candidates.append(item)
                elif isinstance(item, dict):
                    for nested_key in ("name", "slug", "tag", "label"):
                        nested = item.get(nested_key)
                        if isinstance(nested, str):
                            candidates.append(nested)
                            break

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        tag = _normalize_tag(candidate)
        if tag and tag not in seen:
            seen.add(tag)
            normalized.append(tag)
    return normalized


def _coerce_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _load_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("events", "data", "items"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
    return []


def ingest_event_tags(path: str | None = None) -> dict[str, int | str]:
    source_path = Path(path or settings.polymarket_events_json_path)
    if not source_path.exists():
        return {"status": "error", "message": f"file not found: {source_path}"}

    events = _load_events(source_path)
    session = SyncSessionLocal()
    try:
        event_rows = 0
        tag_rows = 0
        join_rows = 0
        market_links = 0
        market_links_by_condition = 0

        for event in events:
            event_id = str(event.get("id", "")).strip()
            if not event_id:
                continue

            session.execute(
                text(
                    """
                    INSERT INTO polymarket_events (id, slug, title, created_at, updated_at, raw)
                    VALUES (:id, :slug, :title, :created_at, NOW(), CAST(:raw AS jsonb))
                    ON CONFLICT (id) DO UPDATE SET
                      slug = EXCLUDED.slug,
                      title = EXCLUDED.title,
                      updated_at = NOW(),
                      raw = EXCLUDED.raw
                    """
                ),
                {
                    "id": event_id,
                    "slug": event.get("slug"),
                    "title": event.get("title") or "",
                    "created_at": _coerce_dt(event.get("createdAt") or event.get("created_at")),
                    "raw": json.dumps(event, separators=(",", ":")),
                },
            )
            event_rows += 1

            tags = _extract_tags(event)
            for tag in tags:
                session.execute(
                    text(
                        """
                        INSERT INTO polymarket_tags (tag)
                        VALUES (:tag)
                        ON CONFLICT (tag) DO NOTHING
                        """
                    ),
                    {"tag": tag},
                )
                tag_rows += 1

                session.execute(
                    text(
                        """
                        INSERT INTO polymarket_event_tags (event_id, tag_id)
                        SELECT :event_id, t.id
                        FROM polymarket_tags t
                        WHERE t.tag = :tag
                        ON CONFLICT (event_id, tag_id) DO NOTHING
                        """
                    ),
                    {"event_id": event_id, "tag": tag},
                )
                join_rows += 1

            # Primary linkage path: nested markets[*].conditionId -> markets.id
            nested_markets = event.get("markets")
            if isinstance(nested_markets, list):
                for market_obj in nested_markets:
                    if not isinstance(market_obj, dict):
                        continue
                    condition_id = str(market_obj.get("conditionId", "")).strip()
                    if not condition_id:
                        continue
                    updated_condition = session.execute(
                        text(
                            """
                            UPDATE markets
                            SET polymarket_event_id = :event_id,
                                updated_at = NOW()
                            WHERE id = :condition_id
                            """
                        ),
                        {"event_id": event_id, "condition_id": condition_id},
                    )
                    market_links_by_condition += int(getattr(updated_condition, "rowcount", 0) or 0)

            # Secondary fallback link path by event_id/slug
            slug = (event.get("slug") or "").strip()
            if slug:
                updated = session.execute(
                    text(
                        """
                        UPDATE markets
                        SET polymarket_event_id = :event_id,
                            updated_at = NOW()
                        WHERE polymarket_event_id IS NULL
                          AND (event_id = :event_id OR slug ILIKE :slug_like)
                        """
                    ),
                    {"event_id": event_id, "slug_like": f"%{slug}%"},
                )
            else:
                updated = session.execute(
                    text(
                        """
                        UPDATE markets
                        SET polymarket_event_id = :event_id,
                            updated_at = NOW()
                        WHERE polymarket_event_id IS NULL
                          AND event_id = :event_id
                        """
                    ),
                    {"event_id": event_id},
                )
            market_links += int(getattr(updated, "rowcount", 0) or 0)

        session.commit()
        return {
            "status": "success",
            "events": event_rows,
            "tags_seen": tag_rows,
            "event_tag_links": join_rows,
            "market_links_condition_id": market_links_by_condition,
            "market_links": market_links,
        }
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Polymarket event tags from JSON")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()
    print(ingest_event_tags(path=args.path))


if __name__ == "__main__":
    main()
