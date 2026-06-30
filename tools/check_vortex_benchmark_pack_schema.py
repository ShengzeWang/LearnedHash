#!/usr/bin/env python3
"""Fixture-based schema check for the Vortex SIFT benchmark pack."""

from __future__ import annotations

import ast
import json
import struct
import tempfile
from pathlib import Path

from vortex_benchmark_pack.cli import build_arg_parser
from vortex_benchmark_pack.common import PROFILE_PATHS, profile_manifest_info
from vortex_benchmark_pack.dominant_cost import dominant_cost
from vortex_benchmark_pack.metric_flattening import flatten_metrics
from vortex_benchmark_pack.selector_summary import summarize_selector
from vortex_benchmark_pack.reporting import write_report, write_summary_csv
from vortex_benchmark_pack.selector_args import binary_paths, selector_args


EXPECTED_KEYS = {
    "profile",
    "preset",
    "summary_schema_version",
    "legacy_metric_aliases",
    "git_commit",
    "git_dirty",
    "base_count",
    "query_count",
    "dim",
    "dataset_manifest_available",
    "dataset_manifest_path",
    "dataset_manifest_sha256",
    "dataset_source_family",
    "dataset_raw_validation_ok",
    "dataset_conversion_available",
    "dataset_conversion_validated",
    "dataset_base_manifest_sha256",
    "dataset_query_manifest_sha256",
    "dataset_groundtruth_manifest_sha256",
    "selector_wall_seconds",
    "selector_peak_rss_kib",
    "selector_assignment_sample_limit",
    "selector_assignment_sample_policy",
    "selector_graph_centroid_order",
    "selector_eval_nearest_cache_inflight_bypasses",
    "selector_eval_nearest_cache_prewarm_requests",
    "selector_eval_nearest_cache_prewarm_ready_models",
    "selector_eval_nearest_cache_prewarm_skipped",
    "selector_eval_nearest_cache_prewarm_failures",
    "selector_eval_nearest_cache_prewarm_threads",
    "selector_eval_nearest_cache_prewarm_ms",
    "selector_eval_nearest_cache_sampled_wait_ms",
    "selector_eval_nearest_cache_calibration_wait_ms",
    "selector_eval_nearest_cache_sampled_build_ms",
    "selector_eval_nearest_cache_calibration_build_ms",
    "selector_centroid_order_policy",
    "candidate_count",
    "candidate_ok_count",
    "best_recall_at_k_in_window",
    "materialized_role_requested",
    "materialized_role",
    "materialized_matches_best",
    "train_threads",
    "train_threads_fallback",
    "model_file_bytes",
    "codegen_model_bin_bytes",
    "codegen_centroid_index_bytes",
    "codegen_package_bytes",
    "codegen_cold_load_time_ms",
    "codegen_cold_library_status",
    "codegen_warm_load_time_ms",
    "eval_recall_at_k_in_window",
    "eval_truth_groundtruth_load_time_ms",
    "materialized_overlay_match_score",
    "eval_overlay_match_score",
    "eval_node_locality_score",
    "materialized_selector_vs_eval_recall_delta",
    "materialized_selector_vs_eval_recall_abs_delta",
    "materialized_recall_validation_tolerance",
    "materialized_recall_validation_status",
    "hash_latency_avg_ms",
    "hash_latency_p95_ms",
    "hash_latency_p99_ms",
    "dominant_cost",
    "dominant_cost_wall_seconds",
}


def fixture_manifest() -> dict:
    manifest = {
        "summary_schema_version": 2,
        "legacy_metric_aliases": False,
        "profile": "siftsmall",
        "preset": "smoke",
        "materialized_role_requested": "auto",
        "materialized_role": "peak_recall",
        "git": {"commit": "fixture", "dirty": False},
        "dataset": {
            "base": {"count": 10_000, "dim": 128, "bytes": 5_120_000},
            "query": {"count": 100, "dim": 128, "bytes": 51_200},
            "manifest": {
                "available": True,
                "required": False,
                "path": "vector_datasets/siftsmall/manifest.json",
                "sha256": "fixture-manifest-sha",
                "profile": "siftsmall",
                "dataset": "SIFT-Small",
                "metric": "L2",
                "source_family": "texmex",
                "base_split": "official_texmex_base_file",
                "query_split": "official_texmex_query_file",
                "groundtruth_scope": "official_exact_top100_for_official_base",
                "groundtruth_exact": True,
                "raw_validation_ok": True,
                "conversion_available": False,
                "conversion_validated": None,
                "files": {
                    "base": {
                        "sha256": "base-sha",
                        "bytes": 5_120_000,
                        "count": 10_000,
                        "dim": 128,
                    },
                    "query": {
                        "sha256": "query-sha",
                        "bytes": 51_200,
                        "count": 100,
                        "dim": 128,
                    },
                    "groundtruth": {
                        "sha256": "gt-sha",
                        "bytes": 40_400,
                        "count": 100,
                        "dim": 100,
                    },
                },
            },
        },
        "commands": {
            "selector": {"wall_seconds": 1.25, "peak_rss_kib": 100_000},
            "train_codegen": {"wall_seconds": 0.75, "peak_rss_kib": 50_000},
            "codegen_cold_build_load": {"wall_seconds": 0.50, "peak_rss_kib": 25_000},
            "vortex_eval": {"wall_seconds": 0.25, "peak_rss_kib": 20_000},
        },
        "selector_summary": {
            "candidate_count": 4,
            "candidate_ok_count": 4,
            "selector_assignment_sample_limit": 0,
            "selector_assignment_sample_policy": "exact",
            "selector_graph_centroid_order": True,
            "selector_eval_nearest_cache_inflight_bypasses": 2,
            "selector_eval_nearest_cache_prewarm_requests": 3,
            "selector_eval_nearest_cache_prewarm_ready_models": 3,
            "selector_eval_nearest_cache_prewarm_skipped": 0,
            "selector_eval_nearest_cache_prewarm_failures": 0,
            "selector_eval_nearest_cache_prewarm_threads": 8,
            "selector_eval_nearest_cache_prewarm_ms": 10_000.0,
            "selector_eval_nearest_cache_wait_ms": 122_000.0,
            "selector_eval_nearest_cache_sampled_wait_ms": 5_000.0,
            "selector_eval_nearest_cache_calibration_wait_ms": 117_000.0,
            "selector_eval_nearest_cache_build_ms": 45_000.0,
            "selector_eval_nearest_cache_sampled_build_ms": 6_000.0,
            "selector_eval_nearest_cache_calibration_build_ms": 39_000.0,
            "selector_centroid_order_policy": "euclidean_mst_2opt_plus_nsw_graph_affinity",
            "best_recall_at_k_in_window": 0.875,
            "best_overlay_match_score": 0.42,
            "materialized_matches_best": False,
            "materialized_matches_peak_recall": True,
            "materialized_recall_at_k_in_window": 0.875,
            "materialized_overlay_match_score": 0.42,
        },
        "materialized_model": {
            "train_threads": 4,
            "train_threads_fallback": False,
            "model_file_bytes": 2048,
            "codegen_model_bin_bytes": 1024,
            "codegen_centroid_index_bytes": 512,
            "codegen_package_bytes": 4096,
        },
        "codegen_eval": {
            "codegen_cold_build_load": {
                "parsed_output": {
                    "load_time_ms": 11.0,
                    "codegen_library_status": "built",
                },
            },
            "codegen_warm_load": {
                "parsed_output": {
                    "load_time_ms": 3.0,
                    "codegen_library_status": "cache_hit",
                },
            },
            "vortex_eval": {
                "parsed_output": {
                    "load_time_ms": 2.0,
                    "codegen_library_status": "loaded",
                    "truth_cache": "groundtruth",
                    "truth_groundtruth_load_time_ms": 0.3,
                    "truth_search_time_ms": 0.0,
                    "recall@k_in_window": 0.86,
                    "mean_rank_distance_norm": 0.02,
                    "overlay_match_score": 0.41,
                    "overlay_match_p05": 0.20,
                    "overlay_match_p50": 0.40,
                    "overlay_match_p95": 0.70,
                    "node_locality_score": 0.71,
                    "latency_avg_ms": 0.0012,
                    "latency_p95_ms": 0.0020,
                    "latency_p99_ms": 0.0030,
                },
            },
        },
    }
    metrics = flatten_metrics(manifest)
    manifest["dominant_cost"] = dominant_cost(metrics)
    return manifest


def check_no_internal_metrics_wrapper_imports() -> None:
    def uses_metrics_wrapper(path: Path) -> bool:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name == "vortex_benchmark_pack.metrics"
                    for alias in node.names
                ):
                    return True
            elif isinstance(node, ast.ImportFrom):
                if node.level == 1:
                    if node.module == "metrics":
                        return True
                    if node.module is None and any(
                        alias.name == "metrics" for alias in node.names
                    ):
                        return True
                elif node.level == 0:
                    if node.module == "vortex_benchmark_pack.metrics":
                        return True
                    if node.module == "vortex_benchmark_pack" and any(
                        alias.name == "metrics" for alias in node.names
                    ):
                        return True
        return False

    package_dir = Path(__file__).resolve().parent / "vortex_benchmark_pack"
    offenders: list[str] = []
    for path in sorted(package_dir.glob("*.py")):
        if path.name == "metrics.py":
            continue
        if uses_metrics_wrapper(path):
            offenders.append(path.relative_to(package_dir.parent).as_posix())
    this_path = Path(__file__).resolve()
    if uses_metrics_wrapper(this_path):
        offenders.append(this_path.name)
    if offenders:
        raise AssertionError(
            "internal benchmark-pack code should import focused metric modules, "
            f"not the compatibility metrics wrapper: {offenders}"
        )


def check_parser_accepts_existing_cli() -> None:
    parser = build_arg_parser()
    expected_profiles = {
        "siftsmall",
        "sift",
        "gist1m",
        "sift10m",
        "deep10m_l2",
        "spacev10m",
    }
    if set(PROFILE_PATHS) != expected_profiles:
        raise AssertionError(f"unexpected benchmark profiles: {set(PROFILE_PATHS)}")
    args = parser.parse_args(
        [
            "--profile",
            "siftsmall",
            "--preset",
            "smoke",
            "--skip-build",
            "--eval-cache-root",
            "none",
            "--codegen-build-cache-root",
            "none",
            "--materialize-role",
            "auto",
            "--legacy-metric-aliases",
            "--node-count-values",
            "32,64,128",
            "--K-values",
            "1024,2048",
            "--knn-values",
            "64,128",
            "--cdf-branches",
            "64,128",
            "--cdf-models",
            "linear,linear;cubic,linear",
            "--two-opt-iters",
            "8,12",
            "--disable-full-dataset-assignments",
            "--disable-graph-centroid-order",
            "--assignment-sample-limit",
            "12345",
        ]
    )
    assert args.profile == "siftsmall"
    assert args.preset == "smoke"
    assert args.skip_build
    assert args.legacy_metric_aliases
    paths = binary_paths(Path("build"))
    assert {"vortex_v1_cli", "vortex_model_selector", "vortex_eval"} <= paths.keys()
    selector_cmd = selector_args(
        args,
        paths,
        {
            "base": Path("base.fvecs"),
            "query": Path("query.fvecs"),
            "hnsw_index": Path("index.bin"),
        },
        Path("run"),
    )
    assert "--dataset" in selector_cmd
    assert "--output" in selector_cmd
    assert "--selector-json-mode" not in selector_cmd
    assert "--disable-full-dataset-assignments" in selector_cmd
    assert "--disable-graph-centroid-order" in selector_cmd
    node_index = selector_cmd.index("--node-count-values")
    assert selector_cmd[node_index + 1] == "32,64,128"
    k_index = selector_cmd.index("--K-values")
    assert selector_cmd[k_index + 1] == "1024,2048"
    knn_index = selector_cmd.index("--knn-values")
    assert selector_cmd[knn_index + 1] == "64,128"
    branch_index = selector_cmd.index("--cdf-branches")
    assert selector_cmd[branch_index + 1] == "64,128"
    cdf_index = selector_cmd.index("--cdf-models")
    assert selector_cmd[cdf_index + 1] == "linear,linear;cubic,linear"
    two_opt_index = selector_cmd.index("--two-opt-iters")
    assert selector_cmd[two_opt_index + 1] == "8,12"
    limit_index = selector_cmd.index("--assignment-sample-limit")
    assert selector_cmd[limit_index + 1] == "12345"

    sift_args = parser.parse_args(
        [
            "--profile",
            "sift",
            "--preset",
            "baseline",
            "--skip-build",
        ]
    )
    sift_selector_cmd = selector_args(
        sift_args,
        paths,
        {
            "base": Path("base.fvecs"),
            "query": Path("query.fvecs"),
            "hnsw_index": Path("index.bin"),
        },
        Path("run"),
    )
    assert "--target-skeleton-percentages" not in sift_selector_cmd
    assert "--K-values" not in sift_selector_cmd
    assert "--knn-values" not in sift_selector_cmd
    assert "--cdf-branches" not in sift_selector_cmd

    ten_m_args = parser.parse_args(
        [
            "--profile",
            "sift10m",
            "--preset",
            "smoke",
            "--skip-build",
        ]
    )
    ten_m_selector_cmd = selector_args(
        ten_m_args,
        paths,
        {
            "base": Path("base.fvecs"),
            "query": Path("query.fvecs"),
            "hnsw_index": Path("index.bin"),
        },
        Path("run"),
    )
    assert "--dataset" in ten_m_selector_cmd
    assert "--target-skeleton-percentages" in ten_m_selector_cmd

    ten_m_baseline_args = parser.parse_args(
        [
            "--profile",
            "sift10m",
            "--preset",
            "baseline",
            "--budget-seconds",
            "1200",
            "--skip-build",
        ]
    )
    ten_m_baseline_selector_cmd = selector_args(
        ten_m_baseline_args,
        paths,
        {
            "base": Path("base.fvecs"),
            "query": Path("query.fvecs"),
            "hnsw_index": Path("index.bin"),
        },
        Path("run"),
    )
    assert (
        ten_m_baseline_selector_cmd[
            ten_m_baseline_selector_cmd.index("--max-search-seconds") + 1
        ]
        == "1200"
    )
    assert (
        ten_m_baseline_selector_cmd[
            ten_m_baseline_selector_cmd.index("--max-model-trains") + 1
        ]
        == "360"
    )
    assert (
        ten_m_baseline_selector_cmd[
            ten_m_baseline_selector_cmd.index("--max-stagnant-contexts") + 1
        ]
        == "2"
    )
    assert (
        ten_m_baseline_selector_cmd[
            ten_m_baseline_selector_cmd.index("--min-recall-improvement") + 1
        ]
        == "0.0005"
    )

    ten_m_deep_args = parser.parse_args(
        [
            "--profile",
            "sift10m",
            "--preset",
            "deep",
            "--budget-seconds",
            "3600",
            "--skip-build",
        ]
    )
    ten_m_deep_selector_cmd = selector_args(
        ten_m_deep_args,
        paths,
        {
            "base": Path("base.fvecs"),
            "query": Path("query.fvecs"),
            "hnsw_index": Path("index.bin"),
        },
        Path("run"),
    )
    assert (
        ten_m_deep_selector_cmd[ten_m_deep_selector_cmd.index("--max-model-trains") + 1]
        == "720"
    )
    assert (
        ten_m_deep_selector_cmd[ten_m_deep_selector_cmd.index("--phase1-keep") + 1]
        == "32"
    )
    assert (
        ten_m_deep_selector_cmd[
            ten_m_deep_selector_cmd.index("--final-calibration-count") + 1
        ]
        == "32"
    )

    override_args = parser.parse_args(
        [
            "--profile",
            "sift",
            "--preset",
            "baseline",
            "--skip-build",
            "--target-skeleton-percentages",
            "1,10,100",
        ]
    )
    override_selector_cmd = selector_args(
        override_args,
        paths,
        {
            "base": Path("base.fvecs"),
            "query": Path("query.fvecs"),
            "hnsw_index": Path("index.bin"),
        },
        Path("run"),
    )
    override_index = override_selector_cmd.index("--target-skeleton-percentages")
    assert override_selector_cmd[override_index + 1] == "1,10,100"


def check_flattened_schema() -> None:
    manifest = fixture_manifest()
    metrics = flatten_metrics(manifest)
    missing = sorted(EXPECTED_KEYS - metrics.keys())
    if missing:
        raise AssertionError(f"missing benchmark-pack metric keys: {missing}")
    if metrics["dominant_cost"] != "selector_search":
        raise AssertionError(f"unexpected dominant cost: {metrics['dominant_cost']}")
    if metrics["selector_eval_nearest_cache_calibration_wait_ms"] != 117_000.0:
        raise AssertionError("fixture should expose calibration nearest-cache wait")
    if (
        manifest["dominant_cost"].get("largest_selector_substage")
        != "nearest_centroid_cache_calibration_wait"
    ):
        raise AssertionError(
            "dominant-cost analysis should use split nearest-cache wait"
        )
    if metrics["materialized_recall_validation_status"] != "mismatch":
        raise AssertionError("fixture should expose selector/eval recall mismatch")
    if abs(metrics["materialized_selector_vs_eval_recall_delta"] + 0.015) > 1.0e-12:
        raise AssertionError("unexpected materialized recall validation delta")

    rounded_manifest = fixture_manifest()
    rounded_manifest["selector_summary"][
        "materialized_recall_at_k_in_window"
    ] = 0.3072265625
    rounded_manifest["codegen_eval"]["vortex_eval"]["parsed_output"][
        "recall@k_in_window"
    ] = 0.307227
    rounded_metrics = flatten_metrics(rounded_manifest)
    if rounded_metrics["materialized_recall_validation_status"] != "matched":
        raise AssertionError(
            "six-decimal vortex_eval recall rounding should validate as matched"
        )

    one_hit_manifest = fixture_manifest()
    one_hit_eval = one_hit_manifest["codegen_eval"]["vortex_eval"]["parsed_output"]
    one_hit_eval["query_count"] = 1000
    one_hit_eval["k"] = 10
    one_hit_manifest["selector_summary"]["materialized_recall_at_k_in_window"] = 0.2503
    one_hit_eval["recall@k_in_window"] = 0.2504
    one_hit_metrics = flatten_metrics(one_hit_manifest)
    if one_hit_metrics["materialized_recall_validation_status"] != "matched":
        raise AssertionError(
            "one-hit selector/eval recall granularity should validate as matched"
        )
    if (
        abs(one_hit_metrics["materialized_recall_validation_tolerance"] - 0.0001)
        > 1.0e-12
    ):
        raise AssertionError("unexpected one-hit recall validation tolerance")

    two_hit_manifest = fixture_manifest()
    two_hit_eval = two_hit_manifest["codegen_eval"]["vortex_eval"]["parsed_output"]
    two_hit_eval["query_count"] = 1000
    two_hit_eval["k"] = 10
    two_hit_manifest["selector_summary"]["materialized_recall_at_k_in_window"] = 0.2503
    two_hit_eval["recall@k_in_window"] = 0.2505
    two_hit_metrics = flatten_metrics(two_hit_manifest)
    if two_hit_metrics["materialized_recall_validation_status"] != "mismatch":
        raise AssertionError(
            "multi-hit selector/eval recall deltas should still be flagged"
        )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        write_summary_csv(run_dir / "summary.csv", metrics)
        write_report(run_dir, manifest, metrics)
        for name in ("summary.csv", "report.html", "README.md"):
            path = run_dir / name
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"missing or empty report artifact: {name}")
        report_html = (run_dir / "report.html").read_text(encoding="utf-8")
        for expected in (
            "Quality Metrics",
            "Cost and Search Metrics",
            "Artifacts and Runtime Metrics",
            "All Summary Metrics",
            "Materialized recall validation",
        ):
            if expected not in report_html:
                raise AssertionError(f"benchmark report missing section: {expected}")
        if "radial-gradient" in report_html:
            raise AssertionError(
                "benchmark report should avoid decorative gradient noise"
            )
        if "Dataset Provenance" not in report_html:
            raise AssertionError("benchmark report should expose dataset provenance")


def write_vecs(path: Path, *, count: int, dim: int, kind: str) -> None:
    payload = bytearray()
    for row in range(count):
        payload.extend(struct.pack("<i", dim))
        if kind == "fvecs":
            payload.extend(
                struct.pack(f"<{dim}f", *[float(row + col) for col in range(dim)])
            )
        elif kind == "ivecs":
            payload.extend(struct.pack(f"<{dim}i", *range(dim)))
        else:
            raise AssertionError(f"unsupported vecs kind: {kind}")
    path.write_bytes(payload)


def check_profile_manifest_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = root / "base.fvecs"
        query = root / "query.fvecs"
        groundtruth = root / "groundtruth.ivecs"
        manifest_path = root / "manifest.json"
        write_vecs(base, count=3, dim=2, kind="fvecs")
        write_vecs(query, count=2, dim=2, kind="fvecs")
        write_vecs(groundtruth, count=2, dim=2, kind="ivecs")
        manifest = {
            "profile": "gist1m",
            "dataset": "GIST1M",
            "role": "test",
            "metric": "L2",
            "base_split": "official_texmex_base_file",
            "query_split": "official_texmex_query_file",
            "groundtruth_scope": "official_exact_top100_for_official_base",
            "groundtruth_exact": True,
            "slice_based": False,
            "validation": {"ok": True},
            "files": [
                {
                    "path": str(base),
                    "kind": "fvecs",
                    "count": 3,
                    "dim": 2,
                    "expected_bytes": base.stat().st_size,
                    "sha256": "base-sha",
                },
                {
                    "path": str(query),
                    "kind": "fvecs",
                    "count": 2,
                    "dim": 2,
                    "expected_bytes": query.stat().st_size,
                    "sha256": "query-sha",
                },
                {
                    "path": str(groundtruth),
                    "kind": "ivecs",
                    "count": 2,
                    "dim": 2,
                    "expected_bytes": groundtruth.stat().st_size,
                    "sha256": "gt-sha",
                },
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        info = profile_manifest_info(
            "gist1m",
            {
                "base": base,
                "query": query,
                "groundtruth": groundtruth,
                "manifest": manifest_path,
                "hnsw_index": root / "index.bin",
            },
        )
        if not info["available"] or not info["required"]:
            raise AssertionError("required manifest was not accepted")
        if info["files"]["base"]["count"] != 3:
            raise AssertionError(
                "base count was not preserved from manifest validation"
            )

        manifest["files"][0]["expected_bytes"] += 4
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            profile_manifest_info(
                "gist1m",
                {
                    "base": base,
                    "query": query,
                    "groundtruth": groundtruth,
                    "manifest": manifest_path,
                    "hnsw_index": root / "index.bin",
                },
            )
        except RuntimeError as exc:
            if "byte size" not in str(exc):
                raise AssertionError(f"unexpected stale-manifest error: {exc}") from exc
        else:
            raise AssertionError("stale manifest byte size was accepted")


def check_legacy_alias_policy() -> None:
    best = {
        "ok": True,
        "phase": "best",
        "config": {"target_skeleton": 10_000, "K": 128},
        "recall_at_k_in_window": 0.875,
        "eval_hash_base_ms": 2.0,
    }
    selector_json = {
        "best": best,
        "selection": {},
        "candidates": [
            best,
            {
                "ok": True,
                "phase": "phase1",
                "config": {"target_skeleton": 10_000, "K": 64},
                "eval_hash_base_ms": 1.0,
            },
        ],
    }
    summary = summarize_selector(selector_json, "best", best)
    if summary.get("selector_candidate_eval_hash_base_ms_sum") != 3.0:
        raise AssertionError("candidate timing total missing from canonical key")
    if "selector_eval_hash_base_ms_sum" in summary:
        raise AssertionError("legacy candidate timing alias present by default")

    legacy_summary = summarize_selector(
        selector_json,
        "best",
        best,
        legacy_metric_aliases=True,
    )
    if legacy_summary.get("selector_eval_hash_base_ms_sum") != 3.0:
        raise AssertionError("legacy candidate timing alias missing in opt-in mode")

    cost = dominant_cost(
        {
            "selector_wall_seconds": 10.0,
            "selector_candidate_eval_hash_base_ms_sum": 3_000.0,
        }
    )
    if cost.get("largest_selector_substage") != "base_hashing":
        raise AssertionError(
            "dominant-cost analysis did not use canonical candidate timing key"
        )


def main() -> int:
    check_no_internal_metrics_wrapper_imports()
    check_parser_accepts_existing_cli()
    check_flattened_schema()
    check_profile_manifest_validation()
    check_legacy_alias_policy()
    print("vortex benchmark-pack schema fixture: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
