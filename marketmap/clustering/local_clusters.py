"""Local clustering inside neighborhoods."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal

STOP_TAGS = {
    "markets",
    "market",
    "other",
    "featured",
    "popular",
    "all",
    "news",
    "hide from new",
    "recurring",
    "up or down",
    "5m",
    "15m",
    "1h",
    "new",
}


def _stable_tag_label(tag: str) -> str:
    return tag.strip().lower()


def _best_micro_tag(tags: list[str], doc_freq: Counter[str], total_docs: int, macro: str) -> str | None:
    best_tag = None
    best_score = float("-inf")
    for tag in tags:
        t = _stable_tag_label(tag)
        if not t or t in STOP_TAGS:
            continue
        if macro not in {"misc", "other"} and macro not in t and "::" in t:
            pass
        idf = math.log((1.0 + total_docs) / (1.0 + doc_freq[t])) + 1.0
        score = idf
        if score > best_score:
            best_score = score
            best_tag = t
    return best_tag


def compute_local_clusters() -> dict[str, int | str]:
    session = SyncSessionLocal()
    try:
        rows = session.execute(
            text(
                """
                SELECT m.id, COALESCE(m.neighborhood_key,'misc') AS neighborhood_key,
                       m.global_x, m.global_y, m.global_z,
                       COALESCE(array_agg(DISTINCT pt.tag) FILTER (WHERE pt.tag IS NOT NULL), '{}') AS tags
                FROM markets m
                LEFT JOIN polymarket_event_tags pet ON pet.event_id = m.polymarket_event_id
                LEFT JOIN polymarket_tags pt ON pt.id = pet.tag_id
                WHERE m.is_active = 1.0
                  AND m.global_x IS NOT NULL
                  AND m.global_y IS NOT NULL
                  AND m.global_z IS NOT NULL
                GROUP BY m.id, m.neighborhood_key, m.global_x, m.global_y, m.global_z
                """
            )
        ).fetchall()

        by_key: dict[str, list[tuple[str, np.ndarray, list[str]]]] = defaultdict(list)
        tag_df_by_macro: dict[str, Counter[str]] = defaultdict(Counter)
        for market_id, key, x, y, z, tags in rows:
            norm_tags = [
                _stable_tag_label(str(t))
                for t in (tags or [])
                if isinstance(t, str) and _stable_tag_label(str(t)) not in STOP_TAGS
            ]
            macro = str(key)
            for tag in set(norm_tags):
                tag_df_by_macro[macro][tag] += 1
            by_key[str(key)].append((market_id, np.asarray([x, y, z], dtype=np.float32), norm_tags))

        assigned = 0
        for key in sorted(by_key.keys()):
            items = by_key[key]
            n = len(items)
            if n < settings.neighborhood_local_cluster_min_size:
                for market_id, _, _ in items:
                    session.execute(
                        text(
                            "UPDATE markets SET local_cluster_id = 0, updated_at = NOW() WHERE id = :market_id"
                        ),
                        {"market_id": market_id},
                    )
                    assigned += 1
                continue

            total_docs = max(1, n)
            micro_counts: Counter[str] = Counter()
            micro_by_market: dict[str, str | None] = {}
            for market_id, _, tags in items:
                micro = _best_micro_tag(
                    tags=tags,
                    doc_freq=tag_df_by_macro[key],
                    total_docs=total_docs,
                    macro=key,
                )
                micro_by_market[market_id] = micro
                if micro:
                    micro_counts[micro] += 1

            dominant_micro = [
                tag for tag, count in sorted(micro_counts.items(), key=lambda kv: (-kv[1], kv[0])) if count >= 120
            ]
            dominant_index = {tag: idx for idx, tag in enumerate(dominant_micro)}
            remaining: list[tuple[str, np.ndarray]] = []
            assigned_labels: dict[str, int] = {}
            for market_id, vec, _ in items:
                micro = micro_by_market.get(market_id)
                if micro in dominant_index:
                    assigned_labels[market_id] = dominant_index[micro]
                else:
                    remaining.append((market_id, vec))

            if remaining:
                k = max(2, min(12, int(np.sqrt(len(remaining) / 10))))
                matrix = np.vstack([vec for _, vec in remaining])
                km = KMeans(n_clusters=k, random_state=settings.discovery_cluster_seed, n_init="auto")
                labels = km.fit_predict(matrix)
                offset = len(dominant_micro)
                for i, (market_id, _) in enumerate(remaining):
                    assigned_labels[market_id] = int(labels[i]) + offset

                # Merge tiny local clusters into nearest non-tiny centroid.
                min_cluster_size = max(10, int(0.012 * n))
                counts = Counter(assigned_labels.values())
                tiny_ids = {cid for cid, c in counts.items() if c < min_cluster_size}
                if tiny_ids and len(tiny_ids) < len(counts):
                    by_cluster: dict[int, list[np.ndarray]] = defaultdict(list)
                    vec_by_market = {mid: vec for mid, vec in remaining}
                    for mid, cid in assigned_labels.items():
                        vec = vec_by_market.get(mid)
                        if vec is not None:
                            by_cluster[cid].append(vec)

                    centroid = {
                        cid: np.mean(np.vstack(vecs), axis=0)
                        for cid, vecs in by_cluster.items()
                        if vecs
                    }
                    stable_ids = [
                        cid for cid in sorted(counts.keys()) if cid not in tiny_ids and cid in centroid
                    ]
                    if not stable_ids:
                        # Nothing to merge into safely; keep existing assignments.
                        stable_ids = []
                    for mid, cid in list(assigned_labels.items()):
                        if cid not in tiny_ids:
                            continue
                        vec = vec_by_market.get(mid)
                        if vec is None or not stable_ids:
                            continue
                        nearest = min(
                            stable_ids,
                            key=lambda sid: float(np.linalg.norm(vec - centroid[sid])),
                        )
                        assigned_labels[mid] = nearest

            for market_id, _, _ in items:
                session.execute(
                    text(
                        "UPDATE markets SET local_cluster_id = :cluster_id, updated_at = NOW() WHERE id = :market_id"
                    ),
                    {"market_id": market_id, "cluster_id": int(assigned_labels.get(market_id, 0))},
                )
                assigned += 1

        # store cluster counts in neighborhoods.meta
        for key, items in by_key.items():
            counts = session.execute(
                text(
                    """
                    SELECT local_cluster_id, COUNT(*)
                    FROM markets
                    WHERE neighborhood_key = :key
                      AND is_active = 1.0
                    GROUP BY local_cluster_id
                    ORDER BY local_cluster_id ASC
                    """
                ),
                {"key": key},
            ).fetchall()
            summary = {str(row[0]): int(row[1]) for row in counts if row[0] is not None}
            session.execute(
                text(
                    """
                    UPDATE neighborhoods
                    SET meta = jsonb_set(COALESCE(meta, '{}'::jsonb), '{local_clusters}', CAST(:summary AS jsonb), true),
                        updated_at = NOW()
                    WHERE neighborhood_key = :key
                    """
                ),
                {"key": key, "summary": str(summary).replace("'", '"')},
            )

        session.commit()
        return {"status": "success", "assigned": assigned, "neighborhoods": len(by_key)}
    finally:
        session.close()
