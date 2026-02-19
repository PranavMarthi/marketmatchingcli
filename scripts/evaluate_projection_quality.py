"""Evaluate projection quality for latest or specified projection version.

Computes trustworthiness and continuity at multiple neighborhood sizes, plus
simple category-level compactness and separation diagnostics.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import text

from marketmap.config import settings
from marketmap.models.database import SyncSessionLocal


SPORTS_TERMS = {
    "nba",
    "nfl",
    "mlb",
    "nhl",
    "soccer",
    "olympic",
    "olympics",
    "super bowl",
    "champions league",
    "world cup",
}
POLITICS_TERMS = {
    "election",
    "president",
    "senate",
    "house",
    "trump",
    "biden",
    "governor",
    "democrat",
    "republican",
}
CRYPTO_TERMS = {
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "solana",
    "crypto",
    "token",
    "dogecoin",
}
GEOPOLITICS_TERMS = {
    "russia",
    "ukraine",
    "china",
    "taiwan",
    "israel",
    "iran",
    "nato",
    "war",
    "ceasefire",
    "tariff",
}


def _infer_domain(title: str, category: str) -> str:
    text_blob = f"{title} {category}".lower()
    if any(term in text_blob for term in SPORTS_TERMS):
        return "sports"
    if any(term in text_blob for term in POLITICS_TERMS):
        return "politics"
    if any(term in text_blob for term in CRYPTO_TERMS):
        return "crypto"
    if any(term in text_blob for term in GEOPOLITICS_TERMS):
        return "geopolitics"
    return "other"


def _parse_embedding(value) -> np.ndarray:  # type: ignore[no-untyped-def]
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


def _continuity(high: np.ndarray, low: np.ndarray, k: int) -> float:
    """Compute continuity as mean overlap ratio of top-k neighborhoods."""
    n = high.shape[0]
    if n <= k:
        return 0.0

    high_nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    high_nn.fit(high)
    _, high_idx = high_nn.kneighbors(return_distance=True)

    low_nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    low_nn.fit(low)
    _, low_idx = low_nn.kneighbors(return_distance=True)

    overlaps: list[float] = []
    for i in range(n):
        high_set = set(int(x) for x in high_idx[i][1:])
        low_set = set(int(x) for x in low_idx[i][1:])
        overlaps.append(len(high_set.intersection(low_set)) / k)
    return float(np.mean(overlaps))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-version", default="", help="Projection version to evaluate")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    session = SyncSessionLocal()
    try:
        projection_version = args.projection_version
        if not projection_version:
            projection_version = session.execute(
                text(
                    """
                    SELECT projection_version
                    FROM market_projection_3d
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """
                )
            ).scalar() or ""

        if not projection_version:
            print({"status": "error", "reason": "no_projection"})
            return

        rows = session.execute(
            text(
                """
                SELECT p.market_id, me.embedding::text, p.x, p.y, p.z,
                       COALESCE(m.category, '') AS category,
                       COALESCE(m.title, '') AS title
                FROM market_projection_3d p
                JOIN market_embeddings me ON me.market_id = p.market_id
                JOIN markets m ON m.id = p.market_id
                WHERE p.projection_version = :projection_version
                  AND m.is_active = 1.0
                  AND p.x IS NOT NULL
                  AND p.y IS NOT NULL
                  AND p.z IS NOT NULL
                ORDER BY p.market_id ASC
                """
            ),
            {"projection_version": projection_version},
        ).fetchall()

        embeddings: list[np.ndarray] = []
        coords: list[tuple[float, float, float]] = []
        categories: list[str] = []
        for _, embedding, x, y, z, category, title in rows:
            vec = _parse_embedding(embedding)
            if vec.shape[0] != settings.embedding_dim:
                continue
            embeddings.append(vec)
            coords.append((float(x), float(y), float(z)))
            categories.append(_infer_domain(str(title), str(category)))

        n = len(coords)
        if n < 200:
            print({"status": "error", "reason": "not_enough_rows", "rows": n})
            return

        sample_size = min(args.sample_size, n)
        random.seed(args.seed)
        sample_idx = sorted(random.sample(range(n), sample_size))

        high = _normalize_rows(np.vstack([embeddings[i] for i in sample_idx]).astype(np.float32))
        low = np.asarray([coords[i] for i in sample_idx], dtype=np.float32)
        sample_categories = [categories[i] for i in sample_idx]

        ks = [15, 50, 150]
        metrics: dict[str, float] = {}
        for k in ks:
            kk = min(k, sample_size - 1)
            if kk < 2:
                continue
            metrics[f"trustworthiness@{kk}"] = float(trustworthiness(high, low, n_neighbors=kk))
            metrics[f"continuity@{kk}"] = _continuity(high, low, kk)

        cat_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        for i, cat in enumerate(sample_categories):
            if cat:
                cat_vectors[cat].append(low[i])

        centroids: dict[str, np.ndarray] = {}
        compactness: dict[str, float] = {}
        for cat, vecs in cat_vectors.items():
            if len(vecs) < 50:
                continue
            arr = np.asarray(vecs, dtype=np.float32)
            centroid = np.mean(arr, axis=0)
            centroids[cat] = centroid
            compactness[cat] = float(np.mean(np.linalg.norm(arr - centroid, axis=1)))

        centroid_distances: list[float] = []
        centroid_keys = sorted(centroids.keys())
        for i in range(len(centroid_keys)):
            for j in range(i + 1, len(centroid_keys)):
                a = centroids[centroid_keys[i]]
                b = centroids[centroid_keys[j]]
                centroid_distances.append(float(np.linalg.norm(a - b)))

        category_counts = Counter(sample_categories)
        print(
            {
                "status": "ok",
                "projection_version": projection_version,
                "sample_size": sample_size,
                "metrics": metrics,
                "category_counts_top10": category_counts.most_common(10),
                "category_compactness_top10": sorted(
                    compactness.items(), key=lambda item: item[1]
                )[:10],
                "mean_centroid_separation": float(np.mean(centroid_distances))
                if centroid_distances
                else None,
            }
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
