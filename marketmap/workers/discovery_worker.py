"""Discovery graph worker with projection-aware locality filtering."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal
from marketmap.workers.celery_app import app

logger = logging.getLogger(__name__)
DISCOVERY_LOCK_KEY = 913_441
UNKNOWN_NEIGHBORHOOD = "misc::unknown"


def _normalize_neighborhood(value: str | None) -> str:
    if not value:
        return UNKNOWN_NEIGHBORHOOD
    stripped = value.strip()
    return stripped if stripped else UNKNOWN_NEIGHBORHOOD


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    qq = min(1.0, max(0.0, q))
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = qq * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    t = idx - lo
    return ordered[lo] * (1.0 - t) + ordered[hi] * t


def _chunks(items: list[str], size: int) -> list[list[str]]:
    out: list[list[str]] = []
    for i in range(0, len(items), size):
        out.append(items[i : i + size])
    return out


def _latest_projection_version(session) -> str | None:  # type: ignore[no-untyped-def]
    return session.execute(
        text(
            """
            SELECT projection_version
            FROM market_projection_3d
            ORDER BY updated_at DESC
            LIMIT 1
            """
        )
    ).scalar()


def _load_xyz_map(
    session,  # type: ignore[no-untyped-def]
    market_ids: list[str],
    projection_version: str,
) -> dict[str, tuple[float, float, float]]:
    xyz: dict[str, tuple[float, float, float]] = {}
    if not market_ids:
        return xyz
    for batch in _chunks(market_ids, 5000):
        rows = session.execute(
            text(
                """
                SELECT market_id, x, y, z
                FROM market_projection_3d
                WHERE projection_version = :projection_version
                  AND market_id = ANY(:market_ids)
                """
            ),
            {"projection_version": projection_version, "market_ids": batch},
        ).fetchall()
        for row in rows:
            xyz[str(row[0])] = (float(row[1]), float(row[2]), float(row[3]))
    return xyz


def _load_cluster_map(
    session,  # type: ignore[no-untyped-def]
    market_ids: list[str],
    projection_version: str,
) -> dict[str, str]:
    clusters: dict[str, str] = {}
    if not market_ids:
        return clusters
    for batch in _chunks(market_ids, 5000):
        rows = session.execute(
            text(
                """
                SELECT market_id, cluster_id
                FROM market_clusters
                WHERE projection_version = :projection_version
                  AND market_id = ANY(:market_ids)
                """
            ),
            {"projection_version": projection_version, "market_ids": batch},
        ).fetchall()
        for row in rows:
            clusters[str(row[0])] = str(row[1])
    return clusters


@app.task(
    bind=True,
    name="marketmap.workers.discovery_worker.compute_discovery_edges",
    max_retries=2,
)
def compute_discovery_edges(self, batch_limit: int = 5000) -> dict:  # type: ignore[type-arg]
    """Build discovery edges with semantic + manifold locality filtering."""
    logger.info("Starting discovery edge computation...")
    started = datetime.now(timezone.utc)

    candidate_k = max(1, settings.discovery_candidate_k)
    keep_k = max(1, settings.discovery_keep_k)
    min_similarity = settings.discovery_min_similarity
    quantile = settings.discovery_max_3d_distance_quantile
    alpha = settings.discovery_alpha_semantic
    beta = settings.discovery_beta_3d_penalty
    mutual_knn = settings.discovery_mutual_knn_enabled

    generic_penalty = settings.discovery_generic_title_penalty
    template_penalty = settings.discovery_template_penalty
    hub_penalty_weight = settings.discovery_hub_penalty_weight
    similarity_margin = settings.discovery_similarity_margin

    counters = {
        "low_similarity": 0,
        "missing_xyz_source": 0,
        "missing_xyz_target": 0,
        "too_far_d3": 0,
        "mutual_knn_failed": 0,
        "bridge_limit_pruned": 0,
    }

    session = SyncSessionLocal()
    try:
        locked = bool(
            session.execute(
                text("SELECT pg_try_advisory_lock(:key)"),
                {"key": DISCOVERY_LOCK_KEY},
            ).scalar()
        )
        if not locked:
            return {"status": "skipped", "reason": "discovery_edges_job_already_running"}

        projection_gating_active = settings.discovery_projection_gating_enabled
        projection_version: str | None = None
        if projection_gating_active:
            if (
                settings.discovery_projection_version_mode == "explicit"
                and settings.discovery_projection_version_explicit.strip()
            ):
                projection_version = settings.discovery_projection_version_explicit.strip()
            else:
                projection_version = _latest_projection_version(session)

            if not projection_version:
                projection_gating_active = False
                logger.warning("No market_projection_3d found; projection gating disabled for this run")

        if settings.discovery_force_disable_cross_neighborhood and settings.discovery_cross_neighborhood_edges_enabled:
            logger.warning(
                "discovery_force_disable_cross_neighborhood=True; ignoring cross-neighborhood edge generation"
            )

        session.execute(text("DROP TABLE IF EXISTS tmp_discovery_sources"))
        session.execute(
            text(
                """
                CREATE TEMP TABLE tmp_discovery_sources AS
                SELECT me.market_id,
                       me.embedding,
                       COALESCE(NULLIF(m.neighborhood_key, ''), :unknown) AS neighborhood_key,
                       m.title,
                       COALESCE(m.is_template, 0.0) AS is_template
                FROM market_embeddings me
                JOIN markets m ON m.id = me.market_id
                WHERE m.is_active = 1.0
                ORDER BY m.volume DESC NULLS LAST, me.market_id ASC
                LIMIT :limit
                """
            ),
            {"unknown": UNKNOWN_NEIGHBORHOOD, "limit": batch_limit},
        )
        session.execute(
            text("CREATE INDEX IF NOT EXISTS idx_tmp_discovery_sources_market_id ON tmp_discovery_sources (market_id)")
        )

        source_rows = session.execute(
            text(
                """
                SELECT market_id, neighborhood_key, title, is_template
                FROM tmp_discovery_sources
                ORDER BY market_id ASC
                """
            )
        ).fetchall()

        sources = [str(r[0]) for r in source_rows]
        source_neighborhood: dict[str, str] = {str(r[0]): _normalize_neighborhood(str(r[1])) for r in source_rows}
        market_title: dict[str, str] = {str(r[0]): str(r[2] or "") for r in source_rows}
        market_is_template: dict[str, bool] = {str(r[0]): float(r[3] or 0.0) >= 1.0 for r in source_rows}

        processed = len(sources)
        logger.info("Processing %s sources for discovery edges", processed)
        if processed == 0:
            session.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": DISCOVERY_LOCK_KEY})
            session.commit()
            return {
                "status": "success",
                "markets_processed": 0,
                "edges_created": 0,
                "elapsed_seconds": 0,
            }

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

        candidate_sql = text(
            """
            SELECT
                me2.market_id AS target_id,
                1 - (s.embedding <=> me2.embedding) AS similarity,
                m2.title AS target_title,
                COALESCE(m2.is_template, 0.0) AS target_is_template,
                COALESCE(NULLIF(m2.neighborhood_key, ''), :unknown) AS target_neighborhood
            FROM tmp_discovery_sources s
            JOIN market_embeddings me2 ON TRUE
            JOIN markets m2 ON m2.id = me2.market_id
            WHERE s.market_id = :source_id
              AND m2.is_active = 1.0
              AND COALESCE(NULLIF(m2.neighborhood_key, ''), :unknown) = :neighborhood_key
              AND me2.market_id != :source_id
            ORDER BY s.embedding <=> me2.embedding ASC
            LIMIT :candidate_k
            """
        )

        raw_by_source: dict[str, list[dict[str, Any]]] = {}
        candidate_set: dict[str, set[str]] = {}
        all_candidate_ids: set[str] = set()
        total_candidates = 0

        for source_id in sources:
            neigh = source_neighborhood[source_id]
            rows = session.execute(
                candidate_sql,
                {
                    "source_id": source_id,
                    "neighborhood_key": neigh,
                    "candidate_k": candidate_k,
                    "unknown": UNKNOWN_NEIGHBORHOOD,
                },
            ).fetchall()

            source_candidates: list[dict[str, Any]] = []
            cset: set[str] = set()
            for row in rows:
                target_id = str(row[0])
                sim = float(row[1] or 0.0)
                source_candidates.append(
                    {
                        "target": target_id,
                        "sim": sim,
                    }
                )
                cset.add(target_id)
                all_candidate_ids.add(target_id)

                if target_id not in market_title:
                    market_title[target_id] = str(row[2] or "")
                if target_id not in market_is_template:
                    market_is_template[target_id] = float(row[3] or 0.0) >= 1.0
                if target_id not in source_neighborhood:
                    source_neighborhood[target_id] = _normalize_neighborhood(str(row[4]))

            raw_by_source[source_id] = source_candidates
            candidate_set[source_id] = cset
            total_candidates += len(source_candidates)

        xyz_map: dict[str, tuple[float, float, float]] = {}
        d3_cutoff: float | None = None
        if projection_gating_active and projection_version:
            xyz_map = _load_xyz_map(session, list(set(sources) | all_candidate_ids), projection_version)
            d3_values: list[float] = []
            for source_id in sources:
                sxyz = xyz_map.get(source_id)
                if sxyz is None:
                    continue
                for cand in raw_by_source[source_id]:
                    txyz = xyz_map.get(str(cand["target"]))
                    if txyz is None:
                        continue
                    d3_values.append(
                        math.sqrt(
                            (sxyz[0] - txyz[0]) ** 2
                            + (sxyz[1] - txyz[1]) ** 2
                            + (sxyz[2] - txyz[2]) ** 2
                        )
                    )

            d3_cutoff = _quantile(d3_values, quantile)
            if d3_cutoff is None or d3_cutoff <= 0:
                projection_gating_active = False
                d3_cutoff = None
                logger.warning("Projection gating disabled for this run: insufficient 3D candidate distances")

        cluster_gating_active = False
        cluster_map: dict[str, str] = {}
        if settings.discovery_cluster_gating_enabled and projection_version:
            cluster_map = _load_cluster_map(session, list(set(sources) | all_candidate_ids), projection_version)
            cluster_gating_active = len(cluster_map) > 0

        generic_rx = re.compile(
            r"(^will\s|\bby\b|\bbefore\b|\bafter\b|\b20\d{2}\b|"
            r"\bjan\b|\bfeb\b|\bmar\b|\bapr\b|\bmay\b|\bjun\b|"
            r"\bjul\b|\baug\b|\bsep\b|\boct\b|\bnov\b|\bdec\b)",
            re.IGNORECASE,
        )

        target_frequency: dict[str, int] = {}
        for source_id in sources:
            for cand in raw_by_source[source_id]:
                tgt = str(cand["target"])
                target_frequency[tgt] = target_frequency.get(tgt, 0) + 1
        max_target_frequency = max(target_frequency.values(), default=1)

        selected_by_source: dict[str, list[dict[str, Any]]] = {}
        kept_directed = 0

        for source_id in sources:
            source_candidates = raw_by_source[source_id]
            if projection_gating_active and source_id not in xyz_map:
                counters["missing_xyz_source"] += len(source_candidates)
                selected_by_source[source_id] = []
                continue

            scored: list[dict[str, Any]] = []
            source_title = market_title.get(source_id, "")
            source_is_template = market_is_template.get(source_id, False)

            mean_similarity = 0.0
            if source_candidates:
                mean_similarity = sum(float(c["sim"]) for c in source_candidates) / float(len(source_candidates))

            for cand in source_candidates:
                target_id = str(cand["target"])
                sim = float(cand["sim"])

                if sim < min_similarity or sim < mean_similarity + similarity_margin:
                    counters["low_similarity"] += 1
                    continue

                d3: float | None = None
                d3_norm = 0.0
                if projection_gating_active and d3_cutoff is not None:
                    txyz = xyz_map.get(target_id)
                    if txyz is None:
                        counters["missing_xyz_target"] += 1
                        continue

                    sxyz = xyz_map[source_id]
                    d3 = math.sqrt(
                        (sxyz[0] - txyz[0]) ** 2 + (sxyz[1] - txyz[1]) ** 2 + (sxyz[2] - txyz[2]) ** 2
                    )
                    if d3 > d3_cutoff:
                        counters["too_far_d3"] += 1
                        continue
                    d3_norm = min(d3 / d3_cutoff, 1.0) if d3_cutoff > 0 else 0.0

                score = alpha * sim - beta * d3_norm

                if source_is_template:
                    score -= template_penalty
                if market_is_template.get(target_id, False):
                    score -= template_penalty
                if generic_rx.search(source_title):
                    score -= generic_penalty
                if generic_rx.search(market_title.get(target_id, "")):
                    score -= generic_penalty

                freq = target_frequency.get(target_id, 1)
                denom = math.log1p(max_target_frequency) if max_target_frequency > 1 else 1.0
                score -= hub_penalty_weight * (math.log1p(freq) / denom)

                scored.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "sim": sim,
                        "d3": d3,
                        "score": score,
                    }
                )

            scored.sort(key=lambda item: (-float(item["score"]), -float(item["sim"]), str(item["target"])))

            if cluster_gating_active:
                src_cluster = cluster_map.get(source_id)
                intra: list[dict[str, Any]] = []
                bridge: list[dict[str, Any]] = []
                for item in scored:
                    tgt_cluster = cluster_map.get(str(item["target"]))
                    if src_cluster and tgt_cluster and src_cluster == tgt_cluster:
                        item["edge_kind"] = "intra"
                        intra.append(item)
                    else:
                        item["edge_kind"] = "bridge"
                        bridge.append(item)

                chosen = intra[:keep_k]
                if len(chosen) < keep_k:
                    for item in bridge:
                        if float(item["sim"]) >= settings.discovery_bridge_min_similarity:
                            chosen.append(item)
                        if len(chosen) >= keep_k:
                            break
            else:
                chosen = scored[:keep_k]
                for item in chosen:
                    item["edge_kind"] = "intra"

            kept_directed += len(chosen)
            selected_by_source[source_id] = chosen

        if cluster_gating_active:
            bridge_cap = max(0, settings.discovery_bridge_edges_per_cluster_pair)
            all_selected: list[dict[str, Any]] = []
            for source_id, items in selected_by_source.items():
                for item in items:
                    all_selected.append({"source": source_id, **item})
            all_selected.sort(key=lambda item: (-float(item["score"]), -float(item["sim"]), str(item["target"])))

            bridge_pair_counts: dict[tuple[str, str], int] = {}
            next_selected: dict[str, list[dict[str, Any]]] = {source_id: [] for source_id in selected_by_source}

            for item in all_selected:
                src = str(item["source"])
                tgt = str(item["target"])
                if item.get("edge_kind") != "bridge" or bridge_cap <= 0:
                    next_selected[src].append(item)
                    continue

                c1 = cluster_map.get(src, "unknown")
                c2 = cluster_map.get(tgt, "unknown")
                key = (c1, c2) if c1 <= c2 else (c2, c1)
                used = bridge_pair_counts.get(key, 0)
                if used >= bridge_cap:
                    counters["bridge_limit_pruned"] += 1
                    continue
                bridge_pair_counts[key] = used + 1
                next_selected[src].append(item)

            selected_by_source = next_selected

        mutual_cache: dict[str, set[str]] = dict(candidate_set)

        def _load_candidate_set_for_market(market_id: str) -> set[str]:
            cached = mutual_cache.get(market_id)
            if cached is not None:
                return cached

            neigh = source_neighborhood.get(market_id)
            if neigh is None:
                row = session.execute(
                    text(
                        """
                        SELECT COALESCE(NULLIF(neighborhood_key, ''), :unknown)
                        FROM markets
                        WHERE id = :market_id
                        """
                    ),
                    {"market_id": market_id, "unknown": UNKNOWN_NEIGHBORHOOD},
                ).fetchone()
                neigh = _normalize_neighborhood(str(row[0])) if row and row[0] is not None else UNKNOWN_NEIGHBORHOOD
                source_neighborhood[market_id] = neigh

            rows = session.execute(
                text(
                    """
                    SELECT me2.market_id
                    FROM market_embeddings me_src
                    JOIN market_embeddings me2 ON me2.market_id != me_src.market_id
                    JOIN markets m2 ON m2.id = me2.market_id
                    WHERE me_src.market_id = :source_id
                      AND m2.is_active = 1.0
                      AND COALESCE(NULLIF(m2.neighborhood_key, ''), :unknown) = :neighborhood_key
                    ORDER BY me_src.embedding <=> me2.embedding ASC
                    LIMIT :candidate_k
                    """
                ),
                {
                    "source_id": market_id,
                    "neighborhood_key": neigh,
                    "candidate_k": candidate_k,
                    "unknown": UNKNOWN_NEIGHBORHOOD,
                },
            ).fetchall()

            loaded = {str(row[0]) for row in rows}
            mutual_cache[market_id] = loaded
            return loaded

        post_mutual: dict[str, list[dict[str, Any]]] = {source_id: [] for source_id in selected_by_source}
        for source_id, items in selected_by_source.items():
            for item in items:
                target_id = str(item["target"])
                if not mutual_knn:
                    post_mutual[source_id].append(item)
                    continue

                reverse_candidates = _load_candidate_set_for_market(target_id)
                if source_id in reverse_candidates:
                    post_mutual[source_id].append(item)
                else:
                    counters["mutual_knn_failed"] += 1

        edge_map: dict[tuple[str, str], dict[str, Any]] = {}
        for source_id, items in post_mutual.items():
            for item in items:
                target_id = str(item["target"])
                src = source_id if source_id <= target_id else target_id
                tgt = target_id if source_id <= target_id else source_id
                key = (src, tgt)
                existing = edge_map.get(key)
                if existing is None or float(item["score"]) > float(existing["score"]):
                    edge_map[key] = {
                        "src": src,
                        "tgt": tgt,
                        "sim": float(item["sim"]),
                        "d3": item.get("d3"),
                        "score": float(item["score"]),
                        "edge_kind": str(item.get("edge_kind", "intra")),
                    }

        upsert_rows: list[dict[str, Any]] = []
        for edge in edge_map.values():
            explanation = {
                "type": "semantic",
                "scope": "within_neighborhood",
                "model": settings.embedding_model,
                "projection_gating": projection_gating_active,
                "projection_version": projection_version,
                "d3_cutoff": d3_cutoff,
                "d3": edge["d3"],
                "hybrid_score": edge["score"],
                "cluster_gating": cluster_gating_active,
                "edge_kind": edge["edge_kind"],
            }
            upsert_rows.append(
                {
                    "source_id": edge["src"],
                    "target_id": edge["tgt"],
                    "semantic_score": edge["sim"],
                    "confidence_score": edge["sim"],
                    "template_penalty": 0.0,
                    "explanation": json.dumps(explanation, separators=(",", ":")),
                }
            )

        if upsert_rows:
            session.execute(
                text(
                    """
                    INSERT INTO market_edges
                        (source_id, target_id, edge_type, semantic_score, confidence_score,
                         template_penalty, explanation, updated_at)
                    VALUES
                        (:source_id, :target_id, 'discovery', :semantic_score, :confidence_score,
                         :template_penalty, CAST(:explanation AS jsonb), NOW())
                    ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                        semantic_score = GREATEST(market_edges.semantic_score, EXCLUDED.semantic_score),
                        confidence_score = GREATEST(market_edges.confidence_score, EXCLUDED.confidence_score),
                        explanation = EXCLUDED.explanation,
                        updated_at = NOW()
                    """
                ),
                upsert_rows,
            )

        session.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": DISCOVERY_LOCK_KEY})
        session.commit()

        edges_per_source_avg = (float(kept_directed) / float(processed)) if processed else 0.0
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        logger.info(
            "Discovery edges complete: sources=%s candidates=%s d3_cutoff=%s q=%.3f avg_kept=%.3f final_edges=%s",
            processed,
            total_candidates,
            f"{d3_cutoff:.6f}" if d3_cutoff is not None else "null",
            quantile,
            edges_per_source_avg,
            len(upsert_rows),
        )
        logger.info(
            "Discard counts: low_similarity=%s missing_xyz_source=%s missing_xyz_target=%s "
            "too_far_d3=%s mutual_knn_failed=%s bridge_limit_pruned=%s",
            counters["low_similarity"],
            counters["missing_xyz_source"],
            counters["missing_xyz_target"],
            counters["too_far_d3"],
            counters["mutual_knn_failed"],
            counters["bridge_limit_pruned"],
        )

        return {
            "status": "success",
            "markets_processed": processed,
            "candidate_rows": total_candidates,
            "edges_created": len(upsert_rows),
            "d3_cutoff": d3_cutoff,
            "projection_version": projection_version,
            "projection_gating": projection_gating_active,
            "cluster_gating": cluster_gating_active,
            "discarded": counters,
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
