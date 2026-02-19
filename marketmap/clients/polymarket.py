"""Polymarket API client for Gamma (markets/events) and CLOB (prices) endpoints."""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from marketmap.config import settings

logger = logging.getLogger(__name__)

# Base URLs
GAMMA_BASE = settings.polymarket_gamma_base_url
CLOB_BASE = settings.polymarket_clob_base_url


@dataclass
class RateLimiter:
    """Simple token-bucket rate limiter."""

    rate: float  # requests per second
    _last_call: float = field(default=0.0, init=False)

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        min_interval = 1.0 / self.rate
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call = time.monotonic()


class PolymarketClient:
    """Synchronous Polymarket API client for use in Celery workers."""

    def __init__(self) -> None:
        self._http = httpx.Client(
            timeout=30.0,
            headers={"Accept": "application/json"},
        )
        self._limiter = RateLimiter(rate=settings.polymarket_requests_per_second)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "PolymarketClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # --- Gamma API: Events & Markets ---

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        self._limiter.wait()
        resp = self._http.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def fetch_events_page(
        self,
        offset: int = 0,
        limit: int = 100,
        closed: bool = False,
        active: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch a page of events with embedded markets."""
        params: dict[str, Any] = {
            "order": "id",
            "ascending": "false",
            "closed": str(closed).lower(),
            "active": str(active).lower(),
            "limit": limit,
            "offset": offset,
        }
        data = self._get(f"{GAMMA_BASE}/events", params=params)
        if isinstance(data, list):
            return data
        return []

    def fetch_all_active_events(self, max_pages: int | None = None) -> list[dict[str, Any]]:
        """Paginate through all active events."""
        all_events: list[dict[str, Any]] = []
        offset = 0
        limit = settings.polymarket_page_size
        max_pages = max_pages or settings.polymarket_max_pages

        for page in range(max_pages):
            events = self.fetch_events_page(offset=offset, limit=limit)
            if not events:
                break
            all_events.extend(events)
            offset += limit
            logger.info(f"Fetched events page {page + 1}: {len(events)} events (total: {len(all_events)})")

        return all_events

    def fetch_markets_page(
        self,
        offset: int = 0,
        limit: int = 100,
        closed: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch a page of markets directly."""
        params: dict[str, Any] = {
            "order": "id",
            "ascending": "false",
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }
        data = self._get(f"{GAMMA_BASE}/markets", params=params)
        if isinstance(data, list):
            return data
        return []

    def fetch_market_by_id(self, market_id: str) -> dict[str, Any] | None:
        """Fetch a single market by its Gamma ID."""
        try:
            data = self._get(f"{GAMMA_BASE}/markets/{market_id}")
            return data if isinstance(data, dict) else None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    # --- CLOB API: Prices ---

    def fetch_midpoint_prices(self, token_ids: list[str]) -> dict[str, float]:
        """Fetch midpoint prices for multiple CLOB token IDs.

        The CLOB /midpoints endpoint is not documented for batch,
        so we fetch one-by-one or use /midpoint for each.
        For efficiency, we'll use the CLOB /prices endpoint.
        """
        results: dict[str, float] = {}
        for token_id in token_ids:
            try:
                data = self._get(f"{CLOB_BASE}/midpoint", params={"token_id": token_id})
                if isinstance(data, dict) and "mid" in data:
                    results[token_id] = float(data["mid"])
            except Exception:
                logger.warning(f"Failed to fetch midpoint for token {token_id}")
        return results

    def fetch_price_history(
        self,
        token_id: str,
        interval: str = "1d",
        fidelity: int = 5,
    ) -> list[dict[str, float]]:
        """Fetch price history for a CLOB token.

        Args:
            token_id: The CLOB token ID.
            interval: Time interval - "1m", "1w", "1d", "6h", "1h", "max".
            fidelity: Resolution in minutes.

        Returns:
            List of {"t": timestamp, "p": price} dicts.
        """
        try:
            data = self._get(
                f"{CLOB_BASE}/prices-history",
                params={
                    "market": token_id,
                    "interval": interval,
                    "fidelity": fidelity,
                },
            )
            if isinstance(data, dict) and "history" in data:
                return data["history"]
            return []
        except Exception:
            logger.warning(f"Failed to fetch price history for token {token_id}")
            return []


def extract_markets_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract individual markets from event responses, tagging each with event metadata."""
    markets: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("id", ""))
        event_category = event.get("category") or ""
        event_tags = event.get("tags", [])
        event_neg_risk = event.get("negRisk", False)

        for market in event.get("markets", []):
            market["_event_id"] = event_id
            market["_event_category"] = event_category
            market["_event_tags"] = event_tags
            market["_event_neg_risk"] = event_neg_risk
            market["_event_title"] = event.get("title", "")
            market["_event_description"] = event.get("description", "")
            markets.append(market)

    return markets


def parse_market_for_db(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw Polymarket market dict to our DB schema fields."""
    condition_id = raw.get("conditionId", raw.get("id", ""))

    # Parse outcome prices
    outcome_prices = raw.get("outcomePrices")
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except (json.JSONDecodeError, TypeError):
            outcome_prices = None

    # Parse CLOB token IDs
    clob_token_ids = raw.get("clobTokenIds")
    if isinstance(clob_token_ids, str):
        try:
            clob_token_ids_parsed = json.loads(clob_token_ids)
        except (json.JSONDecodeError, TypeError):
            clob_token_ids_parsed = None
    else:
        clob_token_ids_parsed = clob_token_ids

    # First outcome probability as the primary probability
    probability = None
    if outcome_prices and isinstance(outcome_prices, list) and len(outcome_prices) > 0:
        try:
            probability = float(outcome_prices[0])
        except (ValueError, TypeError):
            pass

    # Category from event metadata or market-level
    category = raw.get("_event_category") or raw.get("category") or ""

    return {
        "id": condition_id,
        "polymarket_id": str(raw.get("id", "")),
        "slug": raw.get("slug"),
        "title": raw.get("question") or raw.get("groupItemTitle") or "",
        "description": raw.get("description") or "",
        "category": category,
        "close_time": raw.get("endDate") or raw.get("endDateIso"),
        "liquidity": raw.get("liquidityNum") or _safe_float(raw.get("liquidity")),
        "volume": raw.get("volumeNum") or _safe_float(raw.get("volume")),
        "outcome_prices": json.dumps(outcome_prices) if outcome_prices else None,
        "clob_token_ids": json.dumps(clob_token_ids_parsed) if clob_token_ids_parsed else None,
        "event_id": raw.get("_event_id") or None,
        "polymarket_event_id": raw.get("_event_id") or None,
        "is_active": 1.0 if raw.get("active") else 0.0,
        "is_template": 1.0 if raw.get("_event_neg_risk") else 0.0,
        "neg_risk": 1.0 if raw.get("_event_neg_risk") else 0.0,
        "_probability": probability,
        "_clob_token_ids_parsed": clob_token_ids_parsed,
    }


def parse_event_for_db(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw Polymarket event dict to our DB schema fields."""
    return {
        "id": str(raw.get("id", "")),
        "slug": raw.get("slug"),
        "title": raw.get("title") or "",
        "description": raw.get("description") or "",
        "category": raw.get("category") or "",
        "end_date": raw.get("endDate"),
        "liquidity": raw.get("liquidity"),
        "volume": raw.get("volume"),
        "neg_risk": 1.0 if raw.get("negRisk") else 0.0,
        "is_active": 1.0 if raw.get("active") else 0.0,
    }


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
