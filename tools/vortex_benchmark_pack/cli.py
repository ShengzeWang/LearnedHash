"""Run a reproducible Vortex v1 benchmark pack on SIFT-style datasets.

The pack is intentionally orchestration-only: it records selector, training,
codegen load, hash latency, model-size, and memory metrics without changing
selector search heuristics or hash algorithms.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from path_safety import PathSafetyError, validate_component, validate_output_path

from .command import run_command
from .common import (
    DEFAULT_CODEGEN_BUILD_CACHE_ROOT,
    DEFAULT_EVAL_CACHE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MATERIALIZATION_ROLES,
    PROFILE_PATHS,
    ROOT,
    cmake_cache,
    cpu_count,
    fvecs_info,
    git_value,
    profile_manifest_info,
    utc_now_slug,
)
from .evaluation import evaluate_codegen
from .materialization import build_materialized_model, select_materialization_candidate
from .dominant_cost import dominant_cost
from .metric_flattening import flatten_metrics
from .selector_summary import summarize_selector
from .reporting import write_report, write_summary_csv
from .selector_args import ensure_build, selector_args


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_PATHS), default="sift")
    parser.add_argument(
        "--preset", choices=["smoke", "baseline", "deep"], default="baseline"
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--eval-cache-root",
        type=Path,
        default=DEFAULT_EVAL_CACHE_ROOT,
        help="Directory for reusable exact-truth caches; use 'none' to disable.",
    )
    parser.add_argument(
        "--codegen-build-cache-root",
        type=Path,
        default=DEFAULT_CODEGEN_BUILD_CACHE_ROOT,
        help="Directory for generated-library build cache; use 'none' to disable.",
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing run directory before writing new results.",
    )
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--faiss-provider", default="bundled")
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--selector-parallelism", type=int, default=0)
    parser.add_argument("--candidate-threads", type=int, default=1)
    parser.add_argument(
        "--train-threads",
        type=int,
        default=0,
        help="Threads for final materialized-role training/codegen; 0 uses --jobs.",
    )
    parser.add_argument(
        "--budget-seconds", type=int, help="Override selector --max-search-seconds."
    )
    parser.add_argument(
        "--target-skeleton-percentages",
        help="Override profile/preset skeleton percentages, for example 1,3,5,10.",
    )
    parser.add_argument(
        "--K-values",
        dest="k_values",
        help="Override selector K ladder, for example 1024,2048,4096.",
    )
    parser.add_argument(
        "--knn-values",
        help="Override selector centroid_knn ladder, for example 32,64,128.",
    )
    parser.add_argument(
        "--cdf-branches",
        help="Override selector CDF branch ladder, for example 32,64,128.",
    )
    parser.add_argument(
        "--cdf-models",
        help="Override selector CDF model pairs, for example linear,linear;cubic,linear.",
    )
    parser.add_argument(
        "--two-opt-iters",
        help="Override selector 2-opt iteration ladder, for example 4,8,12.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument(
        "--window-values",
        help="Optional selector recall-window ladder, for example 250,500,1000,2000,5000.",
    )
    parser.add_argument(
        "--node-count-values",
        help="Optional selector overlay node counts for match/locality scoring, "
        "for example 32,64,128,256,512,1024.",
    )
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument(
        "--final-calibration-count",
        type=int,
        help="Override profile default calibration count; 0 disables calibration.",
    )
    parser.add_argument(
        "--final-calibration-full-count",
        type=int,
        help="Override the number of candidates promoted to full-base calibration; "
        "0/omit uses the selector's dataset-aware default.",
    )
    parser.add_argument(
        "--disable-final-calibration-screen",
        action="store_true",
        help="Disable the selector's full-base/prefix-query calibration screen rung.",
    )
    parser.add_argument(
        "--final-calibration-screen-candidate-count",
        type=int,
        help="Override the selector's calibration screen candidate count; "
        "omit for dataset-aware auto.",
    )
    parser.add_argument(
        "--final-calibration-screen-query-limit",
        type=int,
        help="Override the selector's calibration screen prefix-query count; "
        "omit for dataset-aware auto.",
    )
    parser.add_argument("--final-calibration-query-limit", type=int, default=0)
    parser.add_argument(
        "--post-exploration-candidates",
        type=int,
        help="Override selector post-exploration candidate count; 0/omit = auto.",
    )
    parser.add_argument(
        "--disable-post-exploration",
        action="store_true",
        help="Disable selector post-exploration exploitation.",
    )
    parser.add_argument(
        "--disable-attribution-guided-compaction",
        action="store_true",
        help="Disable M51 high-K beam/frontier candidate compaction for deeper searches.",
    )
    parser.add_argument(
        "--disable-full-dataset-assignments",
        action="store_true",
        help="Use legacy skeleton-only mass/CDF assignment during selector and "
        "materialized training.",
    )
    parser.add_argument(
        "--disable-graph-centroid-order",
        action="store_true",
        help="Use legacy Euclidean-only centroid ordering during selector and "
        "materialized training.",
    )
    parser.add_argument(
        "--assignment-sample-limit",
        type=int,
        help="Cap full-dataset mass/CDF assignment samples; 0 forces exact.",
    )
    parser.add_argument(
        "--materialize-role",
        choices=("auto", *MATERIALIZATION_ROLES),
        default="auto",
        help="Selector role to train/codegen/evaluate. auto uses best, whose "
        "quality score is overlay-match first.",
    )
    parser.add_argument(
        "--selector-memory-budget-bytes",
        type=int,
        help="Override selector memory-aware parallelism budget.",
    )
    parser.add_argument("--bench-count", type=int, default=10000)
    parser.add_argument("--bench-warmup", type=int, default=200)
    parser.add_argument(
        "--bench-breakdown",
        action="store_true",
        help="Record vortex_eval per-stage hash/centroid/ANN latency metrics.",
    )
    parser.add_argument("--centroid-index-pivots", type=int, default=16)
    parser.add_argument(
        "--legacy-metric-aliases",
        action="store_true",
        help="Also write pre-M60 selector_<field>_sum aliases for "
        "candidate-phase timing totals. New consumers should "
        "use selector_candidate_<field>_sum.",
    )
    parser.add_argument(
        "--check-profile-only",
        action="store_true",
        help="Validate profile paths/manifests and exit before build, "
        "selector, training, or evaluation work.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.build_dir = (
        args.build_dir if args.build_dir.is_absolute() else ROOT / args.build_dir
    )
    args.output_root = (
        args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    )
    if (
        isinstance(args.eval_cache_root, Path)
        and str(args.eval_cache_root).lower() == "none"
    ):
        args.eval_cache_root = None
    elif args.eval_cache_root is not None:
        args.eval_cache_root = (
            args.eval_cache_root
            if args.eval_cache_root.is_absolute()
            else ROOT / args.eval_cache_root
        )
    if (
        isinstance(args.codegen_build_cache_root, Path)
        and str(args.codegen_build_cache_root).lower() == "none"
    ):
        args.codegen_build_cache_root = None
    elif args.codegen_build_cache_root is not None:
        args.codegen_build_cache_root = (
            args.codegen_build_cache_root
            if args.codegen_build_cache_root.is_absolute()
            else ROOT / args.codegen_build_cache_root
        )
    try:
        validate_output_path(args.output_root, root=ROOT, label="--output-root")
        if args.eval_cache_root is not None:
            validate_output_path(
                args.eval_cache_root,
                root=ROOT,
                label="--eval-cache-root",
            )
        if args.codegen_build_cache_root is not None:
            validate_output_path(
                args.codegen_build_cache_root,
                root=ROOT,
                label="--codegen-build-cache-root",
            )
    except PathSafetyError as exc:
        parser.error(str(exc))
    args.jobs = args.jobs if args.jobs > 0 else cpu_count()
    if args.candidate_threads <= 0:
        parser.error("--candidate-threads must be > 0")
    if args.train_threads < 0:
        parser.error("--train-threads must be >= 0")
    if args.final_calibration_count is not None and args.final_calibration_count < 0:
        parser.error("--final-calibration-count must be >= 0")
    if (
        args.final_calibration_full_count is not None
        and args.final_calibration_full_count < 0
    ):
        parser.error("--final-calibration-full-count must be >= 0")
    if (
        args.final_calibration_screen_candidate_count is not None
        and args.final_calibration_screen_candidate_count < 0
    ):
        parser.error("--final-calibration-screen-candidate-count must be >= 0")
    if (
        args.final_calibration_screen_query_limit is not None
        and args.final_calibration_screen_query_limit < 0
    ):
        parser.error("--final-calibration-screen-query-limit must be >= 0")
    if args.final_calibration_query_limit < 0:
        parser.error("--final-calibration-query-limit must be >= 0")
    if (
        args.post_exploration_candidates is not None
        and args.post_exploration_candidates < 0
    ):
        parser.error("--post-exploration-candidates must be >= 0")
    if (
        args.selector_memory_budget_bytes is not None
        and args.selector_memory_budget_bytes < 0
    ):
        parser.error("--selector-memory-budget-bytes must be >= 0")
    if args.assignment_sample_limit is not None and args.assignment_sample_limit < 0:
        parser.error("--assignment-sample-limit must be >= 0")

    run_name = args.run_name or f"{args.profile}_{args.preset}_{utc_now_slug()}"
    try:
        run_name = validate_component(run_name, label="--run-name")
        run_dir = validate_output_path(
            args.output_root / run_name,
            root=ROOT,
            label="benchmark run directory",
        )
    except PathSafetyError as exc:
        parser.error(str(exc))

    profile = {key: Path(value) for key, value in PROFILE_PATHS[args.profile].items()}
    for label, path in profile.items():
        if label not in {"hnsw_index", "manifest", "groundtruth"} and not path.exists():
            raise RuntimeError(f"Missing {label} dataset path: {path}")
    dataset_manifest = profile_manifest_info(args.profile, profile)
    if args.check_profile_only:
        print(
            json.dumps(
                {
                    "profile": args.profile,
                    "base": fvecs_info(profile["base"]),
                    "query": fvecs_info(profile["query"]),
                    "manifest": dataset_manifest,
                    "ok": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(
                f"Run directory already exists; use --overwrite: {run_dir}"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = ensure_build(args, run_dir)
    for name, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"Missing required binary after build: {name}={path}")

    commands: dict[str, Any] = {}
    if not profile["hnsw_index"].exists() or args.force_build:
        hnsw_cmd = [
            str(paths["vortex_v1_cli"]),
            "build_hnsw",
            "--dataset",
            str(profile["base"]),
            "--M",
            "32",
            "--efConstruction",
            "200",
            "--output",
            str(profile["hnsw_index"]),
        ]
        commands["build_hnsw"] = run_command("build_hnsw", hnsw_cmd, run_dir / "logs")

    selector_cmd = selector_args(args, paths, profile, run_dir)
    commands["selector"] = run_command(
        "selector",
        selector_cmd,
        run_dir / "logs",
        timeout=(args.budget_seconds + 600) if args.budget_seconds else None,
    )
    selector_json = json.loads((run_dir / "selector.json").read_text())
    materialized_role_requested, materialized_role, materialized_candidate = (
        select_materialization_candidate(args, selector_json)
    )
    materialized_config = materialized_candidate["config"]
    selector_summary = summarize_selector(
        selector_json,
        materialized_role,
        materialized_candidate,
        legacy_metric_aliases=args.legacy_metric_aliases,
    )

    materialized_model = build_materialized_model(
        args,
        paths,
        profile,
        run_dir,
        selector_json,
        materialized_config,
        materialized_role,
    )
    commands["extract_materialized_nsw"] = materialized_model["extract_nsw"]
    commands["train_codegen"] = materialized_model["train_codegen"]

    codegen_eval = evaluate_codegen(
        args, paths, profile, run_dir, Path(materialized_model["codegen_dir"])
    )
    commands["codegen_cold_build_load"] = codegen_eval["codegen_cold_build_load"]
    commands["codegen_warm_load"] = codegen_eval["codegen_warm_load"]
    commands["vortex_eval"] = codegen_eval["vortex_eval"]

    manifest: dict[str, Any] = {
        "summary_schema_version": 2,
        "legacy_metric_aliases": args.legacy_metric_aliases,
        "profile": args.profile,
        "preset": args.preset,
        "materialized_role_requested": materialized_role_requested,
        "materialized_role": materialized_role,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "run_dir": str(run_dir),
        "git": {
            "commit": git_value(["rev-parse", "HEAD"], "unknown"),
            "branch": git_value(["branch", "--show-current"], "unknown"),
            "dirty": bool(git_value(["status", "--porcelain"], "")),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "cpu_count": cpu_count(),
        },
        "build": {
            "build_dir": str(args.build_dir),
            "jobs": args.jobs,
            "cmake_cache": cmake_cache(args.build_dir),
            "binaries": {name: str(path) for name, path in paths.items()},
        },
        "dataset": {
            "base": fvecs_info(profile["base"]),
            "query": fvecs_info(profile["query"]),
            "manifest": dataset_manifest,
            "hnsw_index": {
                "path": str(profile["hnsw_index"]),
                "bytes": (
                    profile["hnsw_index"].stat().st_size
                    if profile["hnsw_index"].exists()
                    else None
                ),
            },
        },
        "requested": {
            "seed": args.seed,
            "hash_bits": args.hash_bits,
            "selector_parallelism": args.selector_parallelism,
            "candidate_threads": args.candidate_threads,
            "train_threads": args.train_threads,
            "budget_seconds": args.budget_seconds,
            "eval_k": args.eval_k,
            "window": args.window,
            "max_queries": args.max_queries,
            "bench_count": args.bench_count,
            "bench_warmup": args.bench_warmup,
            "bench_breakdown": args.bench_breakdown,
            "centroid_index_pivots": args.centroid_index_pivots,
            "materialize_role": args.materialize_role,
            "resolved_materialize_role": materialized_role,
            "legacy_metric_aliases": args.legacy_metric_aliases,
            "eval_cache_root": (
                str(args.eval_cache_root) if args.eval_cache_root else None
            ),
            "codegen_build_cache_root": (
                str(args.codegen_build_cache_root)
                if args.codegen_build_cache_root
                else None
            ),
            "final_calibration_count": (selector_json.get("selection") or {}).get(
                "final_calibration_count"
            ),
            "final_calibration_full_count": (selector_json.get("selection") or {}).get(
                "final_calibration_full_count"
            ),
            "effective_final_calibration_full_count": (
                selector_json.get("selection") or {}
            ).get("effective_final_calibration_full_count"),
            "effective_final_calibration_screen_candidate_count": (
                selector_json.get("selection") or {}
            ).get("effective_final_calibration_screen_candidate_count"),
            "final_calibration_reserve_seconds": (
                selector_json.get("selection") or {}
            ).get("final_calibration_reserve_seconds"),
            "final_calibration_reserve_hit": (selector_json.get("selection") or {}).get(
                "final_calibration_reserve_hit"
            ),
            "training_base_cache_hits": (selector_json.get("selection") or {}).get(
                "training_base_cache_hits"
            ),
            "training_base_cache_misses": (selector_json.get("selection") or {}).get(
                "training_base_cache_misses"
            ),
            "training_order_cache_hits": (selector_json.get("selection") or {}).get(
                "training_order_cache_hits"
            ),
            "training_order_cache_misses": (selector_json.get("selection") or {}).get(
                "training_order_cache_misses"
            ),
            "training_base_cache_build_ms": (selector_json.get("selection") or {}).get(
                "training_base_cache_build_ms"
            ),
            "training_order_cache_build_ms": (selector_json.get("selection") or {}).get(
                "training_order_cache_build_ms"
            ),
            "training_base_cache_memory_bytes": (
                selector_json.get("selection") or {}
            ).get("training_base_cache_memory_bytes"),
            "training_order_cache_memory_bytes": (
                selector_json.get("selection") or {}
            ).get("training_order_cache_memory_bytes"),
            "scheduler_ready_cached_candidates": (
                selector_json.get("selection") or {}
            ).get("scheduler_ready_cached_candidates"),
            "scheduler_in_flight_candidates": (
                selector_json.get("selection") or {}
            ).get("scheduler_in_flight_candidates"),
            "scheduler_missing_candidates": (selector_json.get("selection") or {}).get(
                "scheduler_missing_candidates"
            ),
            "scheduler_single_admission_batches": (
                selector_json.get("selection") or {}
            ).get("scheduler_single_admission_batches"),
            "scheduler_reuse_prioritized_candidates": (
                selector_json.get("selection") or {}
            ).get("scheduler_reuse_prioritized_candidates"),
            "scheduler_fresh_base_candidates": (
                selector_json.get("selection") or {}
            ).get("scheduler_fresh_base_candidates"),
            "final_calibration_screening": (selector_json.get("selection") or {}).get(
                "final_calibration_screening"
            ),
            "final_calibration_screen_candidate_count": (
                selector_json.get("selection") or {}
            ).get("final_calibration_screen_candidate_count"),
            "final_calibration_screen_query_limit": (
                selector_json.get("selection") or {}
            ).get("final_calibration_screen_query_limit"),
            "effective_final_calibration_screen_query_limit": (
                selector_json.get("selection") or {}
            ).get("effective_final_calibration_screen_query_limit"),
            "final_calibration_screen_evaluated": (
                selector_json.get("selection") or {}
            ).get("final_calibration_screen_evaluated"),
            "final_calibration_screen_promoted": (
                selector_json.get("selection") or {}
            ).get("final_calibration_screen_promoted"),
            "final_calibration_role_promoted": (
                selector_json.get("selection") or {}
            ).get("final_calibration_role_promoted"),
            "final_calibration_recall_promoted": (
                selector_json.get("selection") or {}
            ).get("final_calibration_recall_promoted"),
            "final_calibration_screen_status": (
                selector_json.get("selection") or {}
            ).get("final_calibration_screen_status"),
            "final_calibration_query_limit": (selector_json.get("selection") or {}).get(
                "final_calibration_query_limit"
            ),
            "final_calibration_reused": (selector_json.get("selection") or {}).get(
                "final_calibration_reused"
            ),
            "post_exploration_candidates": (selector_json.get("selection") or {}).get(
                "post_exploration_candidates"
            ),
            "post_exploration_evaluated": (selector_json.get("selection") or {}).get(
                "post_exploration_evaluated"
            ),
            "eval_nearest_cache_hits": (selector_json.get("selection") or {}).get(
                "eval_nearest_cache_hits"
            ),
            "eval_nearest_cache_misses": (selector_json.get("selection") or {}).get(
                "eval_nearest_cache_misses"
            ),
            "eval_nearest_cache_inflight_bypasses": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_inflight_bypasses"),
            "eval_nearest_cache_prewarm_requests": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_prewarm_requests"),
            "eval_nearest_cache_prewarm_ready_models": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_prewarm_ready_models"),
            "eval_nearest_cache_prewarm_skipped": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_prewarm_skipped"),
            "eval_nearest_cache_prewarm_failures": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_prewarm_failures"),
            "eval_nearest_cache_prewarm_threads": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_prewarm_threads"),
            "eval_nearest_cache_prewarm_ms": (selector_json.get("selection") or {}).get(
                "eval_nearest_cache_prewarm_ms"
            ),
            "eval_nearest_cache_sampled_wait_ms": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_sampled_wait_ms"),
            "eval_nearest_cache_calibration_wait_ms": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_calibration_wait_ms"),
            "eval_nearest_cache_sampled_build_ms": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_sampled_build_ms"),
            "eval_nearest_cache_calibration_build_ms": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_calibration_build_ms"),
            "eval_nearest_cache_entries": (selector_json.get("selection") or {}).get(
                "eval_nearest_cache_entries"
            ),
            "eval_nearest_cache_memory_bytes": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_memory_bytes"),
            "eval_nearest_cache_sampled_budget_bytes": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_sampled_budget_bytes"),
            "eval_nearest_cache_sampled_memory_bytes": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_sampled_memory_bytes"),
            "eval_nearest_cache_calibration_memory_bytes": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_calibration_memory_bytes"),
            "eval_nearest_cache_evictions": (selector_json.get("selection") or {}).get(
                "eval_nearest_cache_evictions"
            ),
            "eval_nearest_cache_evicted_bytes": (
                selector_json.get("selection") or {}
            ).get("eval_nearest_cache_evicted_bytes"),
            "eval_base_rank_cache_hits": (selector_json.get("selection") or {}).get(
                "eval_base_rank_cache_hits"
            ),
            "eval_base_rank_cache_misses": (selector_json.get("selection") or {}).get(
                "eval_base_rank_cache_misses"
            ),
            "eval_base_rank_cache_wait_ms": (selector_json.get("selection") or {}).get(
                "eval_base_rank_cache_wait_ms"
            ),
            "eval_base_rank_cache_build_ms": (selector_json.get("selection") or {}).get(
                "eval_base_rank_cache_build_ms"
            ),
            "eval_base_rank_cache_nearest_ms": (
                selector_json.get("selection") or {}
            ).get("eval_base_rank_cache_nearest_ms"),
            "eval_base_rank_cache_hash_ms": (selector_json.get("selection") or {}).get(
                "eval_base_rank_cache_hash_ms"
            ),
            "eval_base_rank_cache_sort_ms": (selector_json.get("selection") or {}).get(
                "eval_base_rank_cache_sort_ms"
            ),
            "eval_base_rank_cache_rank_index_ms": (
                selector_json.get("selection") or {}
            ).get("eval_base_rank_cache_rank_index_ms"),
            "eval_base_rank_cache_overhead_ms": (
                selector_json.get("selection") or {}
            ).get("eval_base_rank_cache_overhead_ms"),
            "eval_base_rank_cache_entries": (selector_json.get("selection") or {}).get(
                "eval_base_rank_cache_entries"
            ),
            "eval_base_rank_cache_memory_bytes": (
                selector_json.get("selection") or {}
            ).get("eval_base_rank_cache_memory_bytes"),
            "eval_base_rank_cache_sampled_budget_bytes": (
                selector_json.get("selection") or {}
            ).get("eval_base_rank_cache_sampled_budget_bytes"),
            "eval_base_rank_cache_sampled_memory_bytes": (
                selector_json.get("selection") or {}
            ).get("eval_base_rank_cache_sampled_memory_bytes"),
            "eval_base_rank_cache_calibration_memory_bytes": (
                selector_json.get("selection") or {}
            ).get("eval_base_rank_cache_calibration_memory_bytes"),
            "eval_base_rank_cache_evictions": (
                selector_json.get("selection") or {}
            ).get("eval_base_rank_cache_evictions"),
            "eval_base_rank_cache_evicted_bytes": (
                selector_json.get("selection") or {}
            ).get("eval_base_rank_cache_evicted_bytes"),
            "selector_memory_budget_bytes": (selector_json.get("selection") or {}).get(
                "selector_memory_budget_bytes"
            ),
            "effective_selector_memory_budget_bytes": (
                selector_json.get("selection") or {}
            ).get("effective_selector_memory_budget_bytes"),
        },
        "selector_summary": selector_summary,
        "materialized_selector_candidate": {
            "role": materialized_role,
            "candidate": materialized_candidate,
        },
        "materialized_model": {
            key: value
            for key, value in materialized_model.items()
            if key not in {"extract_nsw", "train_codegen"}
        },
        "best_model": {
            key: value
            for key, value in materialized_model.items()
            if key not in {"extract_nsw", "train_codegen"}
        },
        "codegen_eval": codegen_eval,
        "commands": commands,
    }
    metrics = flatten_metrics(manifest)
    manifest["dominant_cost"] = dominant_cost(metrics)
    metrics = flatten_metrics(manifest)

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    write_summary_csv(run_dir / "summary.csv", metrics)
    write_report(run_dir, manifest, metrics)

    print(f"benchmark_run_dir: {run_dir}")
    print(f"summary_csv: {run_dir / 'summary.csv'}")
    print(f"report_html: {run_dir / 'report.html'}")
    print(f"materialized_role: {materialized_role}")
    print(f"dominant_cost: {manifest['dominant_cost']['name']}")
    print(f"recommendation: {manifest['dominant_cost']['recommendation']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
