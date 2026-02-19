#!/usr/bin/env python3
"""Validate discovery edge locality metrics."""

from __future__ import annotations

import math
import sys
from collections import Counter

from sqlalchemy import create_engine, text

from marketmap.config import settings


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    qq = min(1.0, max(0.0, q))
    ordered = sorted(values)
    idx = int(math.ceil(qq * len(ordered))) - 1
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def main() -> int:
    engine = create_engine(settings.database_url_sync, future=True)
    with engine.connect() as conn:
        projection_version = conn.execute(
            text(
                """
                SELECT projection_version
                FROM market_projection_3d
                ORDER BY updated_at DESC
                LIMIT 1
                """
            )
        ).scalar()

        if not projection_version:
            print("FAIL: no projection_version available")
            return 1

        edge_rows = conn.execute(
            text(
                """
                SELECT source_id, target_id, semantic_score
                FROM market_edges
                WHERE edge_type = 'discovery'
                """
            )
        ).fetchall()

        edge_count = len(edge_rows)
        if edge_count == 0:
            print("FAIL: edge_count=0")
            return 1

        market_ids = {str(r[0]) for r in edge_rows} | {str(r[1]) for r in edge_rows}
        xyz_rows = conn.execute(
            text(
                """
                SELECT market_id, x, y, z
                FROM market_projection_3d
                WHERE projection_version = :projection_version
                  AND market_id = ANY(:market_ids)
                """
            ),
            {"projection_version": projection_version, "market_ids": list(market_ids)},
        ).fetchall()
        xyz = {str(r[0]): (float(r[1]), float(r[2]), float(r[3])) for r in xyz_rows}

        d3_values: list[float] = []
        missing_xyz = 0
        for s, t, _ in edge_rows:
            a = xyz.get(str(s))
            b = xyz.get(str(t))
            if a is None or b is None:
                missing_xyz += 1
                continue
            d3_values.append(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))

        if not d3_values:
            print("FAIL: no edge distances computed (missing projection xyz)")
            return 1

        p50 = quantile(d3_values, 0.50)
        p90 = quantile(d3_values, 0.90)
        p95 = quantile(d3_values, 0.95)
        p99 = quantile(d3_values, 0.99)
        long_edge_rate = sum(1 for d in d3_values if d > p95) / float(len(d3_values))

        cluster_rows = conn.execute(
            text(
                """
                SELECT market_id, cluster_id
                FROM market_clusters
                WHERE projection_version = :projection_version
                  AND market_id = ANY(:market_ids)
                """
            ),
            {"projection_version": projection_version, "market_ids": list(market_ids)},
        ).fetchall()
        clusters = {str(r[0]): str(r[1]) for r in cluster_rows}
        has_cluster_data = len(clusters) > 0

        intra_cluster_rate = None
        top_bridge_pairs: list[tuple[str, int]] = []
        if has_cluster_data:
            intra = 0
            considered = 0
            bridge_counts: Counter[str] = Counter()
            for s, t, _ in edge_rows:
                cs = clusters.get(str(s))
                ct = clusters.get(str(t))
                if cs is None or ct is None:
                    continue
                considered += 1
                if cs == ct:
                    intra += 1
                else:
                    key = f"{cs}|{ct}" if cs <= ct else f"{ct}|{cs}"
                    bridge_counts[key] += 1
            intra_cluster_rate = (intra / float(considered)) if considered else 0.0
            top_bridge_pairs = bridge_counts.most_common(10)

    print("=== Discovery Edge Validation ===")
    print(f"projection_version: {projection_version}")
    print(f"edge_count: {edge_count}")
    print(f"missing_xyz_edges: {missing_xyz}")
    print(
        "d3 stats: "
        f"min={min(d3_values):.6f} median={p50:.6f} p90={p90:.6f} "
        f"p95={p95:.6f} p99={p99:.6f} max={max(d3_values):.6f}"
    )
    print(f"long_edge_rate(>p95): {long_edge_rate:.4f}")

    if has_cluster_data:
        assert intra_cluster_rate is not None
        print(f"intra_cluster_rate: {intra_cluster_rate:.4f}")
        print("top_bridge_cluster_pairs:")
        for key, count in top_bridge_pairs:
            print(f"  {key}: {count}")
    else:
        print("intra_cluster_rate: N/A (no cluster data for projection_version)")

    failed = False
    reasons: list[str] = []
    if edge_count == 0:
        failed = True
        reasons.append("edge_count == 0")
    if long_edge_rate > 0.0500001:
        failed = True
        reasons.append(f"long_edge_rate {long_edge_rate:.4f} > 0.05")
    if has_cluster_data and intra_cluster_rate is not None and intra_cluster_rate < 0.80:
        failed = True
        reasons.append(f"intra_cluster_rate {intra_cluster_rate:.4f} < 0.80")

    if failed:
        print("FAIL: " + "; ".join(reasons))
        return 1

    print("PASS: discovery edges satisfy locality thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
