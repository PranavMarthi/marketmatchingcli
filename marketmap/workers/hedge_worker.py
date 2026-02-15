"""Hedge graph worker: computes confidence-gated dependency edges."""

import json
import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)

STOPWORDS = {
    "will",
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "by",
    "and",
    "or",
    "is",
    "be",
    "with",
    "than",
    "from",
}


def _title_tokens(title: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (title or "").lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2 and not t.isdigit()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1.0 - p))


def _returns_from_points(points: list[tuple[datetime, float]]) -> dict[datetime, float]:
    if len(points) < 2:
        return {}
    points = sorted(points, key=lambda x: x[0])
    out: dict[datetime, float] = {}
    prev = _logit(points[0][1])
    for ts, prob in points[1:]:
        cur = _logit(prob)
        out[ts] = cur - prev
        prev = cur
    return out


def _corr_from_returns(r1: dict[datetime, float], r2: dict[datetime, float], min_points: int) -> float:
    common_ts = sorted(set(r1.keys()) & set(r2.keys()))
    if len(common_ts) < min_points:
        return 0.0
    x = np.array([r1[t] for t in common_ts], dtype=np.float64)
    y = np.array([r2[t] for t in common_ts], dtype=np.float64)
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(max(-1.0, min(1.0, corr)))


def _lead_lag_score(r1: dict[datetime, float], r2: dict[datetime, float], min_points: int) -> float:
    common_ts = sorted(set(r1.keys()) & set(r2.keys()))
    if len(common_ts) < min_points + 2:
        return 0.0

    x = np.array([r1[t] for t in common_ts], dtype=np.float64)
    y = np.array([r2[t] for t in common_ts], dtype=np.float64)

    best = 0.0
    max_lag = 3
    for lag in range(1, max_lag + 1):
        if len(x) - lag < min_points:
            break
        corr_xy = np.corrcoef(x[:-lag], y[lag:])[0, 1]
        corr_yx = np.corrcoef(y[:-lag], x[lag:])[0, 1]
        if not np.isnan(corr_xy):
            best = max(best, abs(float(corr_xy)))
        if not np.isnan(corr_yx):
            best = max(best, abs(float(corr_yx)))
    return min(1.0, best)


def _threshold_info(title: str) -> tuple[str, str | None, float | None]:
    pattern = r"(less than|under|below|more than|over|at least|at most)\s+(\d[\d,]*(?:\.\d+)?)"
    match = re.search(pattern, title.lower())
    if not match:
        return title.lower(), None, None
    comparator = match.group(1)
    value = float(match.group(2).replace(",", ""))
    stem = re.sub(pattern, "", title.lower()).strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem, comparator, value


def _logical_score(
    m1: dict,
    m2: dict,
    entities1: set[str],
    entities2: set[str],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    title1 = (m1.get("title") or "").strip()
    title2 = (m2.get("title") or "").strip()

    if m1.get("event_id") and m1.get("event_id") == m2.get("event_id"):
        score += 0.55
        reasons.append("same_event_family")

    stem1, cmp1, val1 = _threshold_info(title1)
    stem2, cmp2, val2 = _threshold_info(title2)
    if stem1 and stem1 == stem2 and cmp1 and cmp2 and cmp1 != cmp2 and val1 is not None and val2 is not None:
        score += 0.35
        reasons.append("threshold_chain")

    win_re = r"^will\s+(.+?)\s+win\b"
    w1 = re.search(win_re, title1.lower())
    w2 = re.search(win_re, title2.lower())
    if w1 and w2 and w1.group(1) != w2.group(1):
        score += 0.30
        reasons.append("mutex_candidates")

    neg1 = bool(re.search(r"\b(not|no|without|fail|won't|will not)\b", title1.lower()))
    neg2 = bool(re.search(r"\b(not|no|without|fail|won't|will not)\b", title2.lower()))
    token_j = _jaccard(_title_tokens(title1), _title_tokens(title2))
    if token_j >= 0.7 and neg1 != neg2:
        score += 0.40
        reasons.append("complementary_wording")

    if (m1.get("category") or "") == (m2.get("category") or "") and entities1 & entities2:
        score += 0.15
        reasons.append("shared_entities")

    return min(1.0, score), reasons


def _template_penalty(
    m1: dict,
    m2: dict,
    semantic_score: float,
    entity_overlap_score: float,
) -> tuple[float, list[str]]:
    if semantic_score < 0.9 or entity_overlap_score >= 0.1:
        return 0.0, []

    token_j = _jaccard(_title_tokens(m1.get("title") or ""), _title_tokens(m2.get("title") or ""))
    penalty = 0.0
    reasons: list[str] = []

    if token_j >= 0.9:
        penalty += 0.20
        reasons.append("near_duplicate_template")
    elif token_j >= 0.8:
        penalty += 0.12
        reasons.append("high_template_similarity")

    if (m1.get("category") or "") == (m2.get("category") or "") and m1.get("event_id") != m2.get("event_id"):
        penalty += 0.05
        reasons.append("same_category_no_event_link")

    return min(0.30, penalty), reasons


@app.task(bind=True, name="marketmap.workers.hedge_worker.compute_hedge_edges", max_retries=2)
def compute_hedge_edges(self, batch_limit: int = 3000) -> dict:  # type: ignore[type-arg]
    """Compute hedge edges using statistical + logical + propagation signals."""
    logger.info("Starting hedge edge computation...")
    start = datetime.now(timezone.utc)

    session = SyncSessionLocal()
    try:
        result = session.execute(
            text(
                """
                SELECT id, title, category, event_id, close_time, volume
                FROM markets
                WHERE is_active = 1.0
                ORDER BY volume DESC NULLS LAST
                LIMIT :limit
                """
            ),
            {"limit": batch_limit},
        )
        focal_rows = result.fetchall()
        if not focal_rows:
            return {"status": "success", "markets_processed": 0, "edges_created": 0, "elapsed_seconds": 0}

        focal_ids = [r[0] for r in focal_rows]
        logger.info(f"Preparing hedge candidates for {len(focal_ids)} markets")

        meta_result = session.execute(
            text(
                """
                SELECT id, title, category, event_id, close_time, volume
                FROM markets
                WHERE is_active = 1.0
                """
            )
        )
        market_meta: dict[str, dict] = {}
        category_index: dict[str, list[str]] = defaultdict(list)
        for row in meta_result.fetchall():
            market_meta[row[0]] = {
                "id": row[0],
                "title": row[1] or "",
                "category": row[2],
                "event_id": row[3],
                "close_time": row[4],
                "volume": row[5] or 0.0,
            }
            if row[2]:
                category_index[row[2]].append(row[0])

        for cat in category_index:
            category_index[cat].sort(key=lambda mid: market_meta[mid]["volume"], reverse=True)

        entities_result = session.execute(
            text(
                """
                SELECT market_id, entity_name, entity_type
                FROM market_entities
                WHERE entity_type != 'NONE'
                """
            )
        )
        entity_map: dict[str, set[str]] = defaultdict(set)
        entity_to_markets: dict[str, set[str]] = defaultdict(set)
        for market_id, entity_name, entity_type in entities_result.fetchall():
            key = f"{entity_type}:{(entity_name or '').strip().lower()}"
            entity_map[market_id].add(key)
            entity_to_markets[key].add(market_id)

        discovery_result = session.execute(
            text(
                """
                SELECT source_id, target_id, confidence_score
                FROM market_edges
                WHERE edge_type = 'discovery'
                """
            )
        )
        discovery_adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        discovery_scores: dict[tuple[str, str], float] = {}
        for source_id, target_id, conf in discovery_result.fetchall():
            discovery_adj[source_id].append((target_id, float(conf)))
            discovery_adj[target_id].append((source_id, float(conf)))
            discovery_scores[(source_id, target_id)] = float(conf)

        for mid in discovery_adj:
            discovery_adj[mid].sort(key=lambda x: x[1], reverse=True)

        horizon = timedelta(days=settings.hedge_candidate_time_horizon_days)
        candidate_map: dict[str, list[str]] = {}
        all_needed_ids: set[str] = set(focal_ids)

        for market_id in focal_ids:
            m = market_meta.get(market_id)
            if not m:
                continue

            candidates: list[str] = []
            seen: set[str] = {market_id}

            for neighbor_id, _ in discovery_adj.get(market_id, [])[: settings.hedge_candidate_semantic_k]:
                if neighbor_id not in seen:
                    seen.add(neighbor_id)
                    candidates.append(neighbor_id)

            overlap_counts: dict[str, int] = defaultdict(int)
            for ent in entity_map.get(market_id, set()):
                for other_id in entity_to_markets.get(ent, set()):
                    if other_id != market_id:
                        overlap_counts[other_id] += 1
            top_entity = sorted(overlap_counts.items(), key=lambda x: x[1], reverse=True)
            for other_id, _ in top_entity[: settings.hedge_candidate_entity_k]:
                if other_id not in seen:
                    seen.add(other_id)
                    candidates.append(other_id)

            category = m.get("category")
            close_time = m.get("close_time")
            category_added = 0
            for other_id in category_index.get(category, []):
                if other_id in seen:
                    continue
                if close_time and market_meta.get(other_id, {}).get("close_time"):
                    other_close = market_meta[other_id]["close_time"]
                    if abs(other_close - close_time) > horizon:
                        continue
                seen.add(other_id)
                candidates.append(other_id)
                category_added += 1
                if category_added >= settings.hedge_candidate_category_k:
                    break

            candidate_map[market_id] = candidates[: settings.hedge_candidate_max_total]
            all_needed_ids.update(candidate_map[market_id])

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=settings.hedge_price_window_hours)
        prices_result = session.execute(
            text(
                """
                SELECT market_id, timestamp, probability
                FROM market_prices
                WHERE market_id = ANY(:ids)
                  AND timestamp >= :window_start
                ORDER BY market_id, timestamp ASC
                """
            ),
            {"ids": list(all_needed_ids), "window_start": window_start},
        )
        price_points: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        for market_id, ts, probability in prices_result.fetchall():
            if probability is None:
                continue
            p = float(probability)
            if 0.001 < p < 0.999:
                price_points[market_id].append((ts, p))

        returns_map: dict[str, dict[datetime, float]] = {}
        for market_id, points in price_points.items():
            returns_map[market_id] = _returns_from_points(points)

        for i in range(0, len(focal_ids), 1000):
            batch = focal_ids[i : i + 1000]
            session.execute(
                text(
                    """
                    DELETE FROM market_edges
                    WHERE edge_type = 'hedge'
                      AND (source_id = ANY(:ids) OR target_id = ANY(:ids))
                    """
                ),
                {"ids": batch},
            )
        session.commit()

        processed_pairs: set[tuple[str, str]] = set()
        edges_created = 0

        for idx, market_id in enumerate(focal_ids, start=1):
            m1 = market_meta.get(market_id)
            if not m1:
                continue

            for other_id in candidate_map.get(market_id, []):
                if other_id not in market_meta:
                    continue
                src, tgt = (market_id, other_id) if market_id < other_id else (other_id, market_id)
                if (src, tgt) in processed_pairs:
                    continue
                processed_pairs.add((src, tgt))

                m2 = market_meta[other_id]

                semantic_score = discovery_scores.get((src, tgt), 0.0)

                ent1 = entity_map.get(src, set())
                ent2 = entity_map.get(tgt, set())
                entity_overlap_score = _jaccard(ent1, ent2)

                stat_corr = _corr_from_returns(
                    returns_map.get(src, {}),
                    returns_map.get(tgt, {}),
                    min_points=settings.hedge_min_points,
                )
                stat_score = min(1.0, abs(stat_corr))

                propagation_score = _lead_lag_score(
                    returns_map.get(src, {}),
                    returns_map.get(tgt, {}),
                    min_points=settings.hedge_min_points,
                )

                logical_score, logical_reasons = _logical_score(m1, m2, ent1, ent2)
                template_penalty, template_reasons = _template_penalty(
                    m1,
                    m2,
                    semantic_score,
                    entity_overlap_score,
                )

                confidence = (
                    0.40 * stat_score
                    + 0.30 * logical_score
                    + 0.20 * propagation_score
                    + 0.08 * entity_overlap_score
                    + 0.02 * semantic_score
                    - template_penalty
                )
                confidence = max(0.0, min(1.0, confidence))

                if confidence < settings.hedge_min_confidence:
                    continue

                explanation = {
                    "type": "hedge",
                    "components": {
                        "stat": round(stat_score, 4),
                        "logical": round(logical_score, 4),
                        "propagation": round(propagation_score, 4),
                        "entity": round(entity_overlap_score, 4),
                        "semantic": round(semantic_score, 4),
                        "template_penalty": round(template_penalty, 4),
                    },
                    "logical_reasons": logical_reasons,
                    "template_penalty_reasons": template_reasons,
                }

                session.execute(
                    text(
                        """
                        INSERT INTO market_edges (
                            source_id,
                            target_id,
                            edge_type,
                            stat_score,
                            logical_score,
                            propagation_score,
                            entity_overlap_score,
                            semantic_score,
                            template_penalty,
                            confidence_score,
                            explanation,
                            updated_at
                        ) VALUES (
                            :src,
                            :tgt,
                            'hedge',
                            :stat_score,
                            :logical_score,
                            :propagation_score,
                            :entity_overlap_score,
                            :semantic_score,
                            :template_penalty,
                            :confidence_score,
                            :explanation,
                            NOW()
                        )
                        ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                            stat_score = EXCLUDED.stat_score,
                            logical_score = EXCLUDED.logical_score,
                            propagation_score = EXCLUDED.propagation_score,
                            entity_overlap_score = EXCLUDED.entity_overlap_score,
                            semantic_score = EXCLUDED.semantic_score,
                            template_penalty = EXCLUDED.template_penalty,
                            confidence_score = EXCLUDED.confidence_score,
                            explanation = EXCLUDED.explanation,
                            updated_at = NOW()
                        """
                    ),
                    {
                        "src": src,
                        "tgt": tgt,
                        "stat_score": stat_score,
                        "logical_score": logical_score,
                        "propagation_score": propagation_score,
                        "entity_overlap_score": entity_overlap_score,
                        "semantic_score": semantic_score,
                        "template_penalty": template_penalty,
                        "confidence_score": confidence,
                        "explanation": json.dumps(explanation),
                    },
                )
                edges_created += 1

            if idx % 200 == 0:
                session.commit()
                logger.info(
                    f"  Processed {idx}/{len(focal_ids)} markets, "
                    f"{edges_created} hedge edges so far"
                )

        session.commit()

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(
            f"Hedge edges complete: {edges_created} edges for "
            f"{len(focal_ids)} markets in {elapsed:.1f}s"
        )

        return {
            "status": "success",
            "markets_processed": len(focal_ids),
            "edges_created": edges_created,
            "elapsed_seconds": elapsed,
        }

    except Exception as exc:
        session.rollback()
        logger.exception("Hedge edge computation failed")
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()
