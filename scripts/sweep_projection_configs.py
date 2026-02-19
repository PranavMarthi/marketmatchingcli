"""Run A/B projection sweeps and optionally apply best config.

This script avoids re-embedding and only reprojects existing embeddings.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionProfile:
    name: str
    stage1_neighbors: int
    stage1_min_dist: float
    stage2_neighbors: int
    stage2_min_dist: float
    stage2_metric: str


PROFILES: tuple[ProjectionProfile, ...] = (
    ProjectionProfile(
        name="balanced_global_local",
        stage1_neighbors=80,
        stage1_min_dist=0.03,
        stage2_neighbors=45,
        stage2_min_dist=0.10,
        stage2_metric="euclidean",
    ),
    ProjectionProfile(
        name="strong_global_context",
        stage1_neighbors=120,
        stage1_min_dist=0.05,
        stage2_neighbors=50,
        stage2_min_dist=0.12,
        stage2_metric="euclidean",
    ),
    ProjectionProfile(
        name="high_separation",
        stage1_neighbors=100,
        stage1_min_dist=0.04,
        stage2_neighbors=40,
        stage2_min_dist=0.15,
        stage2_metric="euclidean",
    ),
)


def _run_python(code: str, env_overrides: dict[str, str]) -> str:
    env = os.environ.copy()
    env.update(env_overrides)
    completed = subprocess.run(
        ["python", "-c", code],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _run_cmd(args: list[str], env_overrides: dict[str, str]) -> str:
    env = os.environ.copy()
    env.update(env_overrides)
    completed = subprocess.run(
        args,
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _score(metrics: dict[str, float]) -> float:
    tw15 = metrics.get("trustworthiness@15", 0.0)
    tw50 = metrics.get("trustworthiness@50", 0.0)
    tw150 = metrics.get("trustworthiness@150", 0.0)
    c15 = metrics.get("continuity@15", 0.0)
    c50 = metrics.get("continuity@50", 0.0)
    c150 = metrics.get("continuity@150", 0.0)
    return 0.25 * tw15 + 0.2 * tw50 + 0.15 * tw150 + 0.2 * c15 + 0.12 * c50 + 0.08 * c150


def _env_for_profile(profile: ProjectionProfile) -> dict[str, str]:
    return {
        "PROJECTION_STAGE1_N_NEIGHBORS": str(profile.stage1_neighbors),
        "PROJECTION_STAGE1_MIN_DIST": str(profile.stage1_min_dist),
        "PROJECTION_STAGE2_N_NEIGHBORS": str(profile.stage2_neighbors),
        "PROJECTION_STAGE2_MIN_DIST": str(profile.stage2_min_dist),
        "PROJECTION_STAGE2_METRIC": profile.stage2_metric,
    }


def _projection_and_eval(env_overrides: dict[str, str], sample_size: int) -> tuple[dict, dict]:
    projection_out = _run_python(
        (
            "from marketmap.workers.projection_worker import compute_market_projection_3d; "
            "print(compute_market_projection_3d.apply(kwargs={'batch_limit':50000}).get())"
        ),
        env_overrides,
    )
    projection_result = ast.literal_eval(projection_out.splitlines()[-1])
    projection_version = projection_result.get("projection_version", "")

    eval_out = _run_cmd(
        [
            "python",
            "scripts/evaluate_projection_quality.py",
            "--sample-size",
            str(sample_size),
            "--projection-version",
            str(projection_version),
        ],
        env_overrides,
    )
    eval_result = ast.literal_eval(eval_out.splitlines()[-1])
    return projection_result, eval_result


def _refresh_downstream(env_overrides: dict[str, str]) -> None:
    _run_python(
        "from marketmap.workers.discovery_worker import compute_discovery_edges; print(compute_discovery_edges.apply(kwargs={'batch_limit':50000}).get())",
        env_overrides,
    )
    _run_python(
        "from marketmap.workers.clustering_worker import compute_discovery_clusters; print(compute_discovery_clusters.apply().get())",
        env_overrides,
    )
    _run_python(
        "from marketmap.workers.projection_distortion_worker import compute_projection_distortion_scores; print(compute_projection_distortion_scores.apply().get())",
        env_overrides,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--apply-best", action="store_true")
    args = parser.parse_args()

    run_results: list[dict[str, object]] = []
    best_profile: ProjectionProfile | None = None
    best_score = -1.0
    best_env: dict[str, str] = {}

    for profile in PROFILES:
        env = _env_for_profile(profile)
        projection_result, eval_result = _projection_and_eval(env, args.sample_size)
        metrics = eval_result.get("metrics", {}) if isinstance(eval_result, dict) else {}
        score = _score(metrics if isinstance(metrics, dict) else {})
        run_results.append(
            {
                "profile": profile.name,
                "score": score,
                "projection_version": projection_result.get("projection_version"),
                "metrics": metrics,
            }
        )
        if score > best_score:
            best_score = score
            best_profile = profile
            best_env = env

    output = {
        "status": "ok",
        "runs": run_results,
        "best_profile": best_profile.name if best_profile else None,
        "best_score": best_score,
        "best_env_overrides": best_env,
    }
    print(output)

    if args.apply_best and best_profile is not None:
        _projection_and_eval(best_env, args.sample_size)
        _refresh_downstream(best_env)
        print({"status": "applied", "best_profile": best_profile.name})


if __name__ == "__main__":
    main()
