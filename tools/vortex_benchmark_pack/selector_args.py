"""Build and selector command-line construction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .common import ROOT
from .command import run_command


LARGE_SCALE_PROFILES = {"sift10m", "deep10m_l2", "spacev10m"}


def is_large_scale_profile(profile_name: str) -> bool:
    return profile_name in LARGE_SCALE_PROFILES


def binary_paths(build_dir: Path) -> dict[str, Path]:
    suffix = ".exe" if os.name == "nt" else ""
    tool_dir = build_dir / "hash_functions_core" / "Vortex"
    return {
        "vortex_v1_cli": tool_dir / f"vortex_v1_cli{suffix}",
        "vortex_model_selector": tool_dir / f"vortex_model_selector{suffix}",
        "vortex_eval": tool_dir / f"vortex_eval{suffix}",
    }

def ensure_build(args: argparse.Namespace, run_dir: Path) -> dict[str, Path]:
    paths = binary_paths(args.build_dir)
    missing = [name for name, path in paths.items() if not path.exists()]
    if args.skip_build and missing:
        raise RuntimeError(
            "Missing required binaries with --skip-build: " +
            ", ".join(f"{name}={path}" for name, path in paths.items() if not path.exists())
        )
    if args.force_build or (missing and not args.skip_build):
        configure = [
            "cmake", "-S", str(ROOT), "-B", str(args.build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DLEARNEDHASH_FAISS_PROVIDER={args.faiss_provider}",
        ]
        for item in args.cmake_arg:
            configure.append(item)
        run_command("cmake_configure", configure, run_dir / "logs")
        build = [
            "cmake", "--build", str(args.build_dir), "--config", "Release",
            "-j", str(args.jobs),
        ]
        run_command("cmake_build", build, run_dir / "logs")
    return paths

def selector_args(args: argparse.Namespace,
                  paths: dict[str, Path],
                  profile: dict[str, Path],
                  run_dir: Path) -> list[str]:
    large_profile = is_large_scale_profile(args.profile)
    max_stagnant_contexts = "0"
    min_recall_improvement = "0.001"
    if args.preset == "smoke":
        target_percentages = "20,100" if args.profile == "siftsmall" else "1,3"
        k_values = "32,64,128,256,512" if args.profile == "siftsmall" else "512,1024,2048"
        knn_values = "16,32,64,128" if args.profile == "siftsmall" else "64,128,256"
        branches = "32,64,128"
        cdf_models = "cubic,linear;cubic,cubic;linear,cubic"
        phase1_keep = "3" if args.profile == "siftsmall" else "4"
        max_candidates = "6" if args.profile == "siftsmall" else "12"
        max_model_trains = "8" if args.profile == "siftsmall" else "16"
        coarse_base = "1000"
        coarse_query = "16"
        eval_base = "1500" if args.profile == "siftsmall" else "4000"
        eval_query = "16"
        latency_iters = "32"
        max_search_seconds = args.budget_seconds or 60
        recall_rounds = "1"
    elif args.preset == "deep":
        target_percentages = "10,20,30,40,50,60,80,100" if args.profile == "siftsmall" else None
        k_values = "32,64,128,256,512,1024,2048,4096,8192" if args.profile == "siftsmall" else None
        knn_values = "32,64,128,256,512,1024" if args.profile == "siftsmall" else None
        branches = "32,64,128,256,512" if args.profile == "siftsmall" else None
        cdf_models = "cubic,cubic;cubic,linear;linear,cubic;linear,linear"
        phase1_keep = "20"
        max_candidates = "80" if args.profile == "siftsmall" else "160"
        max_model_trains = "160" if args.profile == "siftsmall" else "320"
        coarse_base = "7000" if args.profile == "siftsmall" else "30000"
        coarse_query = "96" if args.profile == "siftsmall" else "192"
        eval_base = "10000" if args.profile == "siftsmall" else "100000"
        eval_query = "100" if args.profile == "siftsmall" else "768"
        latency_iters = "1024" if args.profile == "siftsmall" else "1536"
        max_search_seconds = args.budget_seconds or (900 if args.profile == "siftsmall" else 3600)
        recall_rounds = "4"
        if large_profile and max_search_seconds >= 1800:
            # Long 10M-scale searches need a larger train/candidate budget, but
            # should still stop when extra skeleton contexts stop improving.
            phase1_keep = "32"
            max_candidates = "320"
            max_model_trains = "720"
            coarse_base = "40000"
            coarse_query = "256"
            eval_base = "150000"
            eval_query = "1024"
            recall_rounds = "6"
            max_stagnant_contexts = "2"
            min_recall_improvement = "0.0005"
    else:
        target_percentages = "10,20,40,60,80,100" if args.profile == "siftsmall" else None
        k_values = "32,64,128,256,512,1024,2048,4096" if args.profile == "siftsmall" else None
        knn_values = "32,64,128,256,512" if args.profile == "siftsmall" else None
        branches = "32,64,128,256" if args.profile == "siftsmall" else None
        cdf_models = "cubic,cubic;cubic,linear;linear,cubic"
        phase1_keep = "12"
        max_candidates = "40" if args.profile == "siftsmall" else "80"
        max_model_trains = "80" if args.profile == "siftsmall" else "180"
        coarse_base = "5000" if args.profile == "siftsmall" else "20000"
        coarse_query = "64" if args.profile == "siftsmall" else "128"
        eval_base = "10000" if args.profile == "siftsmall" else "60000"
        eval_query = "100" if args.profile == "siftsmall" else "512"
        latency_iters = "512" if args.profile == "siftsmall" else "1024"
        max_search_seconds = args.budget_seconds or (600 if args.profile == "siftsmall" else 3600)
        recall_rounds = "3"
        if large_profile and max_search_seconds >= 1200:
            phase1_keep = "16"
            max_candidates = "160"
            max_model_trains = "360"
            coarse_base = "30000"
            coarse_query = "192"
            eval_base = "100000"
            eval_query = "768"
            recall_rounds = "4"
            max_stagnant_contexts = "2"
            min_recall_improvement = "0.0005"

    if args.target_skeleton_percentages:
        target_percentages = args.target_skeleton_percentages
    if args.k_values:
        k_values = args.k_values
    if args.knn_values:
        knn_values = args.knn_values
    if args.cdf_branches:
        branches = args.cdf_branches
    if args.cdf_models:
        cdf_models = args.cdf_models
    two_opt_iters = args.two_opt_iters or "4,8,12"
    if args.final_calibration_count is not None:
        final_calibration_count = args.final_calibration_count
    elif args.profile == "siftsmall":
        final_calibration_count = 8
    elif large_profile and args.preset == "deep" and max_search_seconds >= 1800:
        final_calibration_count = 32
    else:
        final_calibration_count = 24

    selector = [
        str(paths["vortex_model_selector"]),
        "--dataset", str(profile["base"]),
        "--query", str(profile["query"]),
        "--index", str(profile["hnsw_index"]),
        "--selector-profile", "balanced" if args.preset != "deep" else "exhaustive",
        "--selector-parallelism", str(args.selector_parallelism),
        "--threads", str(args.candidate_threads),
        "--seed", str(args.seed),
        "--hash_bits", str(args.hash_bits),
        "--cdf-models", cdf_models,
        "--two-opt-iters", two_opt_iters,
        "--phase1-keep", phase1_keep,
        "--max-candidates", max_candidates,
        "--recommend", "8",
        "--eval-k", str(args.eval_k),
        "--window", str(args.window),
        "--coarse-base-limit", coarse_base,
        "--coarse-query-limit", coarse_query,
        "--eval-base-limit", eval_base,
        "--eval-query-limit", eval_query,
        "--coarse-latency-iters", "128",
        "--latency-iters", latency_iters,
        "--coarse-latency-warmup", "32",
        "--latency-warmup", "64",
        "--max-search-seconds", str(max_search_seconds),
        "--max-model-trains", max_model_trains,
        "--max-stagnant-contexts", max_stagnant_contexts,
        "--min-recall-improvement", min_recall_improvement,
        "--optimize-for-recall",
        "--max-recall-refine-rounds", recall_rounds,
        "--post-exploration-candidates", str(args.post_exploration_candidates)
        if args.post_exploration_candidates is not None else "0",
        "--final-calibration-count", str(final_calibration_count),
        "--final-calibration-query-limit", str(args.final_calibration_query_limit or args.max_queries),
        "--output", str(run_dir / "selector.json"),
        "--html-report", str(run_dir / "selector.html"),
    ]
    if target_percentages:
        selector.extend(["--target-skeleton-percentages", target_percentages])
    if k_values:
        selector.extend(["--K-values", k_values])
    if knn_values:
        selector.extend(["--knn-values", knn_values])
    if branches:
        selector.extend(["--cdf-branches", branches])
    if args.window_values:
        selector.extend(["--window-values", args.window_values])
    if args.node_count_values:
        selector.extend(["--node-count-values", args.node_count_values])
    if args.final_calibration_full_count is not None:
        selector.extend(["--final-calibration-full-count", str(args.final_calibration_full_count)])
    if args.disable_final_calibration_screen:
        selector.append("--disable-final-calibration-screen")
    if args.final_calibration_screen_candidate_count is not None:
        selector.extend([
            "--final-calibration-screen-candidate-count",
            str(args.final_calibration_screen_candidate_count),
        ])
    if args.final_calibration_screen_query_limit is not None:
        selector.extend([
            "--final-calibration-screen-query-limit",
            str(args.final_calibration_screen_query_limit),
        ])
    if args.disable_post_exploration:
        selector.append("--disable-post-exploration")
    if args.disable_attribution_guided_compaction:
        selector.append("--disable-attribution-guided-compaction")
    if args.disable_full_dataset_assignments:
        selector.append("--disable-full-dataset-assignments")
    if args.disable_graph_centroid_order:
        selector.append("--disable-graph-centroid-order")
    if args.assignment_sample_limit is not None:
        selector.extend(["--assignment-sample-limit", str(args.assignment_sample_limit)])
    if args.selector_memory_budget_bytes is not None:
        selector.extend(["--selector-memory-budget-bytes", str(args.selector_memory_budget_bytes)])
    return selector
