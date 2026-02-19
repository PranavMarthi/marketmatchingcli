"""Event-first neighborhood assignment, embedding, and projection."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import umap
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text

from marketmap.config import settings
from marketmap.neighborhoods.assign import STOP_TAGS, choose_neighborhood_from_tags
from marketmap.services.embeddings import compute_embeddings


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


def _norm_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _event_text(title: str, description: str, tags: list[str]) -> str:
    tags_text = ", ".join(sorted({t for t in tags if t and t not in STOP_TAGS}))
    return (
        "Represent this prediction market event for semantic similarity and clustering: "
        f"title={title.strip()} description={description[:700].strip()} tags={tags_text}"
    )


def assign_event_neighborhoods(session: Any) -> dict[str, int]:
    rows = session.execute(
        text(
            """
            SELECT e.id,
                   COALESCE(e.title, ''),
                   COALESCE(e.raw->>'description', ''),
                   COALESCE(array_agg(DISTINCT pt.tag) FILTER (WHERE pt.tag IS NOT NULL), '{}') AS tags
            FROM polymarket_events e
            LEFT JOIN polymarket_event_tags pet ON pet.event_id = e.id
            LEFT JOIN polymarket_tags pt ON pt.id = pet.tag_id
            GROUP BY e.id, e.title, e.raw
            ORDER BY e.id ASC
            """
        )
    ).fetchall()

    doc_freq: Counter[str] = Counter()
    tags_by_event: dict[str, list[str]] = {}
    for event_id, _, _, tags in rows:
        normalized = sorted(
            {
                str(t).strip().lower()
                for t in (tags or [])
                if isinstance(t, str) and str(t).strip().lower() not in STOP_TAGS
            }
        )
        tags_by_event[event_id] = normalized
        for tag in normalized:
            doc_freq[tag] += 1

    total_docs = max(1, len(rows))
    assigned = 0
    for event_id, title, description, _ in rows:
        key, label, rank = choose_neighborhood_from_tags(
            tags=tags_by_event.get(event_id, []),
            title=str(title),
            category=str(description),
            doc_freq=doc_freq,
            total_docs=total_docs,
        )
        session.execute(
            text(
                """
                UPDATE polymarket_events
                SET neighborhood_key = :key,
                    neighborhood_label = :label,
                    neighborhood_rank = :rank,
                    updated_at = NOW()
                WHERE id = :event_id
                """
            ),
            {"event_id": event_id, "key": key, "label": label, "rank": rank},
        )
        assigned += 1

    session.commit()
    return {"assigned_events": assigned}


def compute_event_embeddings(session: Any, force_reembed: bool = True) -> dict[str, int]:
    if force_reembed:
        session.execute(text("DELETE FROM event_embeddings"))
        session.commit()

    rows = session.execute(
        text(
            """
            SELECT e.id,
                   COALESCE(e.title, ''),
                   COALESCE(e.raw->>'description', ''),
                   COALESCE(array_agg(DISTINCT pt.tag) FILTER (WHERE pt.tag IS NOT NULL), '{}') AS tags
            FROM polymarket_events e
            LEFT JOIN polymarket_event_tags pet ON pet.event_id = e.id
            LEFT JOIN polymarket_tags pt ON pt.id = pet.tag_id
            LEFT JOIN event_embeddings ee ON ee.event_id = e.id
            WHERE ee.event_id IS NULL
            GROUP BY e.id, e.title, e.raw
            ORDER BY e.id ASC
            """
        )
    ).fetchall()

    if not rows:
        return {"embedded_events": 0}

    event_ids: list[str] = []
    texts: list[str] = []
    for event_id, title, description, tags in rows:
        event_ids.append(event_id)
        texts.append(_event_text(str(title), str(description), [str(t).lower() for t in (tags or []) if isinstance(t, str)]))

    vectors = compute_embeddings(texts)

    for i, event_id in enumerate(event_ids):
        vec = vectors[i]
        vec_text = "[" + ",".join(str(float(x)) for x in vec) + "]"
        session.execute(
            text(
                """
                INSERT INTO event_embeddings (event_id, embedding, model_name, updated_at)
                VALUES (:event_id, :embedding, :model_name, NOW())
                ON CONFLICT (event_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    model_name = EXCLUDED.model_name,
                    updated_at = NOW()
                """
            ),
            {
                "event_id": event_id,
                "embedding": vec_text,
                "model_name": settings.embedding_model,
            },
        )
        if i > 0 and i % 1000 == 0:
            session.commit()

    session.commit()
    return {"embedded_events": len(event_ids)}


def _local_distortion(emb: np.ndarray, xyz: np.ndarray, k: int = 10) -> np.ndarray:
    n = emb.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=np.float32)
    kk = max(1, min(k, n - 1))
    # sklearn requires n_neighbors < n_samples_fit for this query path.
    query_k = max(1, min(kk + 1, n - 1))
    emb_nn = NearestNeighbors(n_neighbors=query_k, metric="cosine", algorithm="brute")
    emb_nn.fit(emb)
    _, emb_ids = emb_nn.kneighbors(return_distance=True)
    xyz_nn = NearestNeighbors(n_neighbors=query_k, metric="euclidean", algorithm="auto")
    xyz_nn.fit(xyz)
    _, xyz_ids = xyz_nn.kneighbors(return_distance=True)
    out = np.zeros(n, dtype=np.float32)
    overlap_k = max(1, emb_ids.shape[1] - 1)
    for i in range(n):
        a = set(int(x) for x in emb_ids[i][1:])
        b = set(int(x) for x in xyz_ids[i][1:])
        out[i] = float(1.0 - (len(a.intersection(b)) / float(overlap_k)))
    return np.clip(out, 0.0, 1.0)


def _smooth_local_layout(emb: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce tiny random islands by pulling high-distortion points to semantic neighbors."""
    n = emb.shape[0]
    if n < 12:
        return xyz, _local_distortion(emb, xyz, k=min(5, max(2, n - 1)))

    smoothed = xyz.copy()
    k = min(30, n - 1)
    emb_nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    emb_nn.fit(emb)
    _, emb_ids = emb_nn.kneighbors(return_distance=True)

    for _ in range(2):
        distortion = _local_distortion(emb, smoothed, k=min(12, n - 1))
        high_idxs = np.where(distortion >= 0.65)[0]
        if high_idxs.size == 0:
            break
        nxt = smoothed.copy()
        for idx in high_idxs:
            neighbors = emb_ids[idx][1:]
            nbr_center = np.mean(smoothed[neighbors], axis=0)
            nxt[idx] = (0.7 * smoothed[idx]) + (0.3 * nbr_center)
        smoothed = nxt

    # Final outlier guard: pull extreme-radius points toward semantic neighborhood center
    distortion = _local_distortion(emb, smoothed, k=min(12, n - 1))
    center = np.mean(smoothed, axis=0)
    radii = np.linalg.norm(smoothed - center, axis=1)
    cutoff = float(np.quantile(radii, 0.985))
    outlier_idxs = np.where((radii >= cutoff) & (distortion >= 0.5))[0]
    if outlier_idxs.size > 0:
        nxt = smoothed.copy()
        for idx in outlier_idxs:
            neighbors = emb_ids[idx][1:11]
            nbr_center = np.mean(smoothed[neighbors], axis=0)
            nxt[idx] = (0.5 * smoothed[idx]) + (0.5 * nbr_center)
        smoothed = nxt

    final_distortion = _local_distortion(emb, smoothed, k=min(12, n - 1))
    return smoothed, final_distortion


def _reshape_local_layout_for_readability(xyz: np.ndarray) -> np.ndarray:
    """Keep subclusters visible while suppressing tiny far-away islands."""
    if xyz.shape[0] < 6:
        return xyz

    out = xyz.copy()
    center = np.mean(out, axis=0)
    shifted = out - center
    radii = np.linalg.norm(shifted, axis=1)
    q90 = float(np.quantile(radii, 0.90))
    q97 = float(np.quantile(radii, 0.97))
    safe_q90 = max(q90, 1e-6)
    max_radius = max(q97 * 1.2, safe_q90 * 1.6)

    # Compress extreme tails to reduce disconnected islands.
    for i in range(out.shape[0]):
        r = radii[i]
        if r > max_radius:
            direction = shifted[i] / max(r, 1e-6)
            shifted[i] = direction * max_radius

    # Expand internal structure so semantic subclusters are easier to inspect.
    target_q90 = 3.6
    scale = target_q90 / safe_q90
    shifted *= scale
    return shifted + center


def _absorb_tiny_islands_within_macro(xyz: np.ndarray) -> np.ndarray:
    """Merge tiny disconnected groups into nearest stable local groups."""
    n = xyz.shape[0]
    if n < 80:
        return xyz

    k = max(5, min(20, int(np.sqrt(n) / 2)))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(xyz)
    centers = km.cluster_centers_

    counts = Counter(int(x) for x in labels)
    tiny_threshold = max(12, int(0.012 * n))
    tiny_ids = {cid for cid, c in counts.items() if c < tiny_threshold}
    stable_ids = [cid for cid in sorted(counts.keys()) if cid not in tiny_ids]
    if not tiny_ids or not stable_ids:
        return xyz

    adjusted = xyz.copy()
    for i in range(n):
        cid = int(labels[i])
        if cid not in tiny_ids:
            continue
        point = adjusted[i]
        nearest = min(
            stable_ids,
            key=lambda sid: float(np.linalg.norm(point - centers[sid])),
        )
        target = centers[nearest]
        # Pull tiny-island points toward nearest stable group but preserve local shape.
        adjusted[i] = (0.45 * point) + (0.55 * target)

    return adjusted


def project_events_hierarchical(session: Any) -> dict[str, Any]:
    rows = session.execute(
        text(
            """
            SELECT e.id,
                   COALESCE(NULLIF(e.neighborhood_key,''), 'misc') AS neighborhood_key,
                   COALESCE(NULLIF(e.neighborhood_label,''), 'Misc') AS neighborhood_label,
                   ee.embedding::text
            FROM polymarket_events e
            JOIN event_embeddings ee ON ee.event_id = e.id
            ORDER BY e.id ASC
            """
        )
    ).fetchall()
    if not rows:
        return {"projected_events": 0}

    event_ids: list[str] = []
    keys: list[str] = []
    labels: list[str] = []
    vectors: list[np.ndarray] = []
    for event_id, key, label, emb in rows:
        vec = _parse_embedding(emb)
        if vec.shape[0] != settings.embedding_dim:
            continue
        event_ids.append(event_id)
        keys.append(str(key))
        labels.append(str(label))
        vectors.append(vec)

    emb_matrix = _norm_rows(np.vstack(vectors).astype(np.float32))
    embedding_hash = hashlib.sha1(emb_matrix.tobytes()).hexdigest()[:12]
    projection_version = f"events_hproj_{embedding_hash}"
    if len(event_ids) >= 8:
        hdbscan_min_cluster = max(6, int(math.sqrt(len(event_ids)) * 0.9))
        hdbscan_min_samples = max(2, min(8, hdbscan_min_cluster // 3))
        hdbscan = HDBSCAN(
            min_cluster_size=hdbscan_min_cluster,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
            algorithm="auto",
        )
        hdbscan_labels = np.asarray(hdbscan.fit_predict(emb_matrix), dtype=np.int32)
    else:
        hdbscan_labels = np.zeros(len(event_ids), dtype=np.int32)

    # Ensure every node belongs to a concrete cluster.
    assigned_mask = hdbscan_labels >= 0
    if not np.any(assigned_mask):
        hdbscan_labels = np.zeros(len(event_ids), dtype=np.int32)
    elif np.any(~assigned_mask):
        known_vecs = emb_matrix[assigned_mask]
        known_labels = hdbscan_labels[assigned_mask]
        unique_labels = np.unique(known_labels)
        centroids = np.vstack([np.mean(known_vecs[known_labels == cid], axis=0) for cid in unique_labels])
        centroids = _norm_rows(centroids.astype(np.float32))
        noise_idxs = np.where(~assigned_mask)[0]
        sims = emb_matrix[noise_idxs] @ centroids.T
        nearest_cluster_idx = np.argmax(sims, axis=1)
        hdbscan_labels[noise_idxs] = unique_labels[nearest_cluster_idx]

    # Re-index labels to compact non-negative ids for storage/readability.
    unique_labels = sorted(int(x) for x in np.unique(hdbscan_labels))
    remap = {old: new for new, old in enumerate(unique_labels)}
    hdbscan_labels = np.asarray([remap[int(x)] for x in hdbscan_labels], dtype=np.int32)

    by_key: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        by_key[key].append(i)

    local_xyz = np.zeros((len(event_ids), 3), dtype=np.float32)
    local_dist = np.zeros(len(event_ids), dtype=np.float32)
    local_cluster = np.zeros(len(event_ids), dtype=np.int32)
    centroid_by_key: dict[str, np.ndarray] = {}
    radius_by_key: dict[str, float] = {}

    for key in sorted(by_key.keys()):
        idxs = by_key[key]
        sub = emb_matrix[idxs]
        if len(idxs) >= 6:
            local_neighbors = min(max(18, int(np.sqrt(len(idxs)) * 1.6)), len(idxs) - 1)
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=local_neighbors,
                min_dist=0.16,
                metric="cosine",
                random_state=42,
                transform_seed=42,
            )
            sub_xyz = np.asarray(reducer.fit_transform(sub), dtype=np.float32)
        else:
            centered = sub - np.mean(sub, axis=0)
            u, s, _ = np.linalg.svd(centered, full_matrices=False)
            sub_xyz = np.zeros((len(idxs), 3), dtype=np.float32)
            dims = min(3, u.shape[1])
            sub_xyz[:, :dims] = (u[:, :dims] * s[:dims]).astype(np.float32)

        sub_xyz, sub_dist = _smooth_local_layout(sub, sub_xyz)
        sub_xyz = _reshape_local_layout_for_readability(sub_xyz)
        sub_xyz = _absorb_tiny_islands_within_macro(sub_xyz)
        sub_dist = _local_distortion(sub, sub_xyz, k=min(12, max(2, len(idxs) - 1)))
        centroid_by_key[key] = np.mean(sub, axis=0)
        center = np.mean(sub_xyz, axis=0)
        radius_by_key[key] = float(max(1e-6, np.percentile(np.linalg.norm(sub_xyz - center, axis=1), 90)))

        for j, gidx in enumerate(idxs):
            local_xyz[gidx] = sub_xyz[j]
            local_dist[gidx] = sub_dist[j]
            local_cluster[gidx] = int(hdbscan_labels[gidx])

    macro_keys = sorted(centroid_by_key.keys())
    centroid_matrix = np.vstack([centroid_by_key[k] for k in macro_keys]).astype(np.float32)
    if len(macro_keys) == 1:
        anchors = np.zeros((1, 3), dtype=np.float32)
    else:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(10, len(macro_keys) - 1),
            min_dist=0.5,
            metric="cosine",
            random_state=42,
            transform_seed=42,
        )
        anchors = np.asarray(reducer.fit_transform(centroid_matrix), dtype=np.float32)

    if len(macro_keys) > 1:
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(anchors)
        dists, _ = nn.kneighbors(return_distance=True)
        target_scale = float(np.median(dists[:, 1])) * 0.75
    else:
        target_scale = 1.0

    # Expand macro anchor spacing for clearer macro separations in UI
    anchors = anchors * 1.60

    scale_by_key = {
        k: float(
            min(4.6, max(1.25, target_scale / radius_by_key[k]))
        )
        for k in macro_keys
    }
    anchor_by_key = {k: anchors[i] for i, k in enumerate(macro_keys)}

    global_xyz = np.zeros_like(local_xyz)
    stitch_dist = np.zeros(len(event_ids), dtype=np.float32)
    for i, key in enumerate(keys):
        a = anchor_by_key[key]
        s = scale_by_key[key]
        global_xyz[i] = a + s * local_xyz[i]
        stitch_dist[i] = float(np.linalg.norm(global_xyz[i] - a))

    closest_n_by_event: dict[str, list[str]] = {event_id: [] for event_id in event_ids}

    session.execute(text("DELETE FROM event_projection_3d"))
    session.execute(text("DELETE FROM neighborhoods"))

    for key in macro_keys:
        idxs = by_key[key]
        label = labels[idxs[0]] if idxs else key.title()
        meta = {"scope": "events", "macro": key, "event_count": len(idxs)}
        centroid_text = "[" + ",".join(str(float(x)) for x in centroid_by_key[key]) + "]"
        session.execute(
            text(
                """
                INSERT INTO neighborhoods
                    (neighborhood_key, label, market_count, anchor_x, anchor_y, anchor_z, scale, centroid_vector, meta, updated_at)
                VALUES
                    (:key, :label, :count, :ax, :ay, :az, :scale, :centroid, CAST(:meta AS jsonb), NOW())
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
                "count": len(idxs),
                "ax": float(anchor_by_key[key][0]),
                "ay": float(anchor_by_key[key][1]),
                "az": float(anchor_by_key[key][2]),
                "scale": scale_by_key[key],
                "centroid": centroid_text,
                "meta": json.dumps(meta, separators=(",", ":")),
            },
        )

    for i, event_id in enumerate(event_ids):
        session.execute(
            text(
                """
                INSERT INTO event_projection_3d
                    (event_id, x, y, z, projection_version, embedding_version,
                     neighborhood_key, neighborhood_label, local_cluster_id,
                     local_distortion, stitch_distortion, closest_n_nodes, updated_at)
                VALUES
                    (:event_id, :x, :y, :z, :projection_version, :embedding_version,
                     :neighborhood_key, :neighborhood_label, :local_cluster_id,
                     :local_distortion, :stitch_distortion, :closest_n_nodes, NOW())
                ON CONFLICT (event_id) DO UPDATE SET
                    x = EXCLUDED.x,
                    y = EXCLUDED.y,
                    z = EXCLUDED.z,
                    projection_version = EXCLUDED.projection_version,
                    embedding_version = EXCLUDED.embedding_version,
                    neighborhood_key = EXCLUDED.neighborhood_key,
                    neighborhood_label = EXCLUDED.neighborhood_label,
                    local_cluster_id = EXCLUDED.local_cluster_id,
                    local_distortion = EXCLUDED.local_distortion,
                    stitch_distortion = EXCLUDED.stitch_distortion,
                    closest_n_nodes = EXCLUDED.closest_n_nodes,
                    updated_at = NOW()
                """
            ),
            {
                "event_id": event_id,
                "x": float(global_xyz[i, 0]),
                "y": float(global_xyz[i, 1]),
                "z": float(global_xyz[i, 2]),
                "projection_version": projection_version,
                "embedding_version": embedding_hash,
                "neighborhood_key": keys[i],
                "neighborhood_label": labels[i],
                "local_cluster_id": int(local_cluster[i]),
                "local_distortion": float(local_dist[i]),
                "stitch_distortion": float(stitch_dist[i]),
                "closest_n_nodes": closest_n_by_event.get(event_id, []),
            },
        )
        if i > 0 and i % 1000 == 0:
            session.commit()

    session.commit()
    return {
        "projected_events": len(event_ids),
        "projection_version": projection_version,
        "embedding_version": embedding_hash,
        "macro_neighborhoods": len(macro_keys),
    }
