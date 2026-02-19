"""Hierarchical neighborhood projection pipeline."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import umap
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text

from marketmap.config import settings


def _parse_embedding(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        txt = value.strip()
        if txt.startswith("[") and txt.endswith("]"):
            txt = txt[1:-1]
        return np.fromstring(txt, sep=",", dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _projection_version(embedding_hash: str) -> str:
    payload = {
        "embedding_hash": embedding_hash,
        "local_n_neighbors": settings.neighborhood_local_umap_n_neighbors,
        "local_min_dist": settings.neighborhood_local_umap_min_dist,
        "local_metric": settings.neighborhood_local_umap_metric,
        "local_seed": settings.neighborhood_local_umap_seed,
        "min_size": settings.neighborhood_min_size,
        "small_merge": settings.neighborhood_small_merge_threshold,
        "stitch_min": settings.neighborhood_stitch_scale_min,
        "stitch_max": settings.neighborhood_stitch_scale_max,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"hproj_{digest}"


def _local_distortion(emb: np.ndarray, local_xyz: np.ndarray, k: int) -> np.ndarray:
    n = emb.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=np.float32)
    kk = max(1, min(k, n - 1))
    emb_nn = NearestNeighbors(n_neighbors=kk + 1, metric="cosine", algorithm="brute")
    emb_nn.fit(emb)
    _, emb_ids = emb_nn.kneighbors(return_distance=True)
    loc_nn = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean", algorithm="auto")
    loc_nn.fit(local_xyz)
    _, loc_ids = loc_nn.kneighbors(return_distance=True)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = set(int(x) for x in emb_ids[i][1:])
        b = set(int(x) for x in loc_ids[i][1:])
        overlap = len(a.intersection(b)) / float(kk)
        out[i] = float(max(0.0, min(1.0, 1.0 - overlap)))
    return out


def run_hierarchical_projection(session: Any, batch_limit: int | None = None) -> dict[str, Any]:
    limit = batch_limit or settings.projection_batch_limit
    rows = session.execute(
        text(
            """
            SELECT m.id,
                   COALESCE(NULLIF(m.neighborhood_key, ''), 'misc::unknown') AS neighborhood_key,
                   COALESCE(NULLIF(m.neighborhood_label, ''), 'Misc / Unknown') AS neighborhood_label,
                   me.embedding::text
            FROM markets m
            JOIN market_embeddings me ON me.market_id = m.id
            WHERE m.is_active = 1.0
            ORDER BY m.id ASC
            LIMIT :limit
            """
        ),
        {"limit": limit},
    ).fetchall()

    if not rows:
        return {"status": "success", "projected": 0, "elapsed_seconds": 0.0}

    market_ids: list[str] = []
    keys: list[str] = []
    labels: list[str] = []
    vectors: list[np.ndarray] = []

    for market_id, key, label, emb in rows:
        vec = _parse_embedding(emb)
        if vec.shape[0] != settings.embedding_dim:
            continue
        market_ids.append(market_id)
        keys.append(str(key))
        labels.append(str(label))
        vectors.append(vec)

    emb_matrix = _normalize_rows(np.vstack(vectors).astype(np.float32))
    emb_hash = hashlib.sha1(emb_matrix.tobytes()).hexdigest()[:12]
    projection_version = _projection_version(emb_hash)

    by_key: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        by_key[key].append(idx)

    # For macro neighborhoods we keep deterministic keys as-is.
    # (Small-group handling now happens in local clustering, not macro anchors.)
    merged: dict[str, str] = {}
    for key, idxs in sorted(by_key.items(), key=lambda kv: kv[0]):
        merged[key] = key

    by_key_merged: dict[str, list[int]] = defaultdict(list)
    label_for_key: dict[str, str] = {}
    for idx, key in enumerate(keys):
        mk = merged.get(key, key)
        by_key_merged[mk].append(idx)
        if mk not in label_for_key:
            label_for_key[mk] = labels[idx]

    local_xyz = np.zeros((len(market_ids), 3), dtype=np.float32)
    local_dist = np.zeros(len(market_ids), dtype=np.float32)
    centroid_vectors: dict[str, np.ndarray] = {}
    local_radius: dict[str, float] = {}

    for nkey, idxs in by_key_merged.items():
        sub_emb = emb_matrix[idxs]
        if len(idxs) >= 8:
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(settings.neighborhood_local_umap_n_neighbors, len(idxs) - 1),
                min_dist=settings.neighborhood_local_umap_min_dist,
                metric=settings.neighborhood_local_umap_metric,
                random_state=settings.neighborhood_local_umap_seed,
                transform_seed=settings.neighborhood_local_umap_seed,
            )
            sub_xyz = np.asarray(reducer.fit_transform(sub_emb), dtype=np.float32)
        else:
            centered = sub_emb - np.mean(sub_emb, axis=0)
            u, s, _ = np.linalg.svd(centered, full_matrices=False)
            dims = min(3, u.shape[1])
            sub_xyz = np.zeros((len(idxs), 3), dtype=np.float32)
            sub_xyz[:, :dims] = (u[:, :dims] * s[:dims]).astype(np.float32)

        centroid = np.mean(sub_emb, axis=0)
        centroid_vectors[nkey] = centroid
        sub_center = np.mean(sub_xyz, axis=0)
        radius = float(np.percentile(np.linalg.norm(sub_xyz - sub_center, axis=1), 90))
        local_radius[nkey] = max(radius, 1e-6)
        sub_dist = _local_distortion(sub_emb, sub_xyz, k=min(10, max(2, len(idxs) - 1)))

        for j, global_idx in enumerate(idxs):
            local_xyz[global_idx] = sub_xyz[j]
            local_dist[global_idx] = sub_dist[j]

    neigh_keys = sorted(by_key_merged.keys())
    centroid_matrix = np.vstack([centroid_vectors[k] for k in neigh_keys]).astype(np.float32)
    if len(neigh_keys) == 1:
        anchors = np.zeros((1, 3), dtype=np.float32)
    else:
        global_reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(10, len(neigh_keys) - 1),
            min_dist=0.4,
            metric="cosine",
            random_state=settings.neighborhood_local_umap_seed,
            transform_seed=settings.neighborhood_local_umap_seed,
        )
        anchors = np.asarray(global_reducer.fit_transform(centroid_matrix), dtype=np.float32)

    # scale by nearest-neighbor anchor spacing
    if len(neigh_keys) > 1:
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(anchors)
        dists, _ = nn.kneighbors(return_distance=True)
        target_scale = float(np.median(dists[:, 1])) * 0.35
    else:
        target_scale = 1.0

    scale_by_key: dict[str, float] = {}
    for k in neigh_keys:
        raw_scale = target_scale / local_radius[k]
        scale_by_key[k] = float(
            min(settings.neighborhood_stitch_scale_max, max(settings.neighborhood_stitch_scale_min, raw_scale))
        )

    anchor_by_key = {k: anchors[i] for i, k in enumerate(neigh_keys)}
    global_xyz = np.zeros_like(local_xyz)
    stitch_dist = np.zeros(len(market_ids), dtype=np.float32)
    for idx, key in enumerate(keys):
        mkey = merged.get(key, key)
        anchor = anchor_by_key[mkey]
        scale = scale_by_key[mkey]
        global_xyz[idx] = anchor + scale * local_xyz[idx]
        stitch_dist[idx] = float(np.linalg.norm(global_xyz[idx] - anchor))

    # persist neighborhoods
    session.execute(text("DELETE FROM neighborhoods"))
    for key in neigh_keys:
        idxs = by_key_merged[key]
        label = label_for_key.get(key, key)
        meta = {
            "macro": key.split("::", 1)[0],
            "local_count": len(idxs),
        }
        centroid_text = "[" + ",".join(str(float(x)) for x in centroid_vectors[key]) + "]"
        session.execute(
            text(
                """
                INSERT INTO neighborhoods
                    (neighborhood_key, label, market_count, anchor_x, anchor_y, anchor_z,
                     scale, centroid_vector, meta, updated_at)
                VALUES
                    (:key, :label, :market_count, :ax, :ay, :az, :scale, :centroid, CAST(:meta AS jsonb), NOW())
                ON CONFLICT (neighborhood_key) DO UPDATE SET
                    label = EXCLUDED.label,
                    market_count = EXCLUDED.market_count,
                    anchor_x = EXCLUDED.anchor_x,
                    anchor_y = EXCLUDED.anchor_y,
                    anchor_z = EXCLUDED.anchor_z,
                    scale = EXCLUDED.scale,
                    centroid_vector = EXCLUDED.centroid_vector,
                    meta = EXCLUDED.meta,
                    updated_at = NOW()
                """
            ),
            {
                "key": key,
                "label": label,
                "market_count": len(idxs),
                "ax": float(anchor_by_key[key][0]),
                "ay": float(anchor_by_key[key][1]),
                "az": float(anchor_by_key[key][2]),
                "scale": scale_by_key[key],
                "centroid": centroid_text,
                "meta": json.dumps(meta, separators=(",", ":")),
            },
        )

    upsert_proj = text(
        """
        INSERT INTO market_projection_3d
            (market_id, x, y, z, projection_version, embedding_version, updated_at)
        VALUES
            (:market_id, :x, :y, :z, :projection_version, :embedding_version, NOW())
        ON CONFLICT (market_id) DO UPDATE SET
            x = EXCLUDED.x,
            y = EXCLUDED.y,
            z = EXCLUDED.z,
            projection_version = EXCLUDED.projection_version,
            embedding_version = EXCLUDED.embedding_version,
            updated_at = NOW()
        """
    )

    for i, market_id in enumerate(market_ids):
        key = merged.get(keys[i], keys[i])
        session.execute(
            text(
                """
                UPDATE markets
                SET neighborhood_key = :key,
                    neighborhood_label = :label,
                    local_x = :lx,
                    local_y = :ly,
                    local_z = :lz,
                    global_x = :gx,
                    global_y = :gy,
                    global_z = :gz,
                    local_distortion = :ld,
                    stitch_distortion = :sd,
                    updated_at = NOW()
                WHERE id = :market_id
                """
            ),
            {
                "market_id": market_id,
                "key": key,
                "label": label_for_key.get(key, key),
                "lx": float(local_xyz[i, 0]),
                "ly": float(local_xyz[i, 1]),
                "lz": float(local_xyz[i, 2]),
                "gx": float(global_xyz[i, 0]),
                "gy": float(global_xyz[i, 1]),
                "gz": float(global_xyz[i, 2]),
                "ld": float(local_dist[i]),
                "sd": float(stitch_dist[i]),
            },
        )
        session.execute(
            upsert_proj,
            {
                "market_id": market_id,
                "x": float(global_xyz[i, 0]),
                "y": float(global_xyz[i, 1]),
                "z": float(global_xyz[i, 2]),
                "projection_version": projection_version,
                "embedding_version": emb_hash,
            },
        )
        if i > 0 and i % 2000 == 0:
            session.commit()

    session.commit()
    return {
        "status": "success",
        "projected": len(market_ids),
        "projection_version": projection_version,
        "embedding_version": emb_hash,
        "neighborhoods": len(neigh_keys),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
