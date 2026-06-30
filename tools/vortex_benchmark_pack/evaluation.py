"""Generated-code build/load and final recall/latency evaluation."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from .common import file_hash64
from .command import run_command


def evaluate_codegen(
    args: argparse.Namespace,
    paths: dict[str, Path],
    profile: dict[str, Path],
    run_dir: Path,
    codegen_dir: Path,
) -> dict[str, Any]:
    build_dir = codegen_dir / "codegen" / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    common = [
        str(paths["vortex_eval"]),
        "--codegen-dir", str(codegen_dir),
        "--centroid-index-pivots", str(args.centroid_index_pivots),
        "--threads", str(args.jobs),
    ]
    eval_env = os.environ.copy()
    if args.codegen_build_cache_root is not None:
        args.codegen_build_cache_root.mkdir(parents=True, exist_ok=True)
        eval_env["VORTEX_CODEGEN_BUILD_CACHE"] = str(args.codegen_build_cache_root)
    cold = run_command(
        "codegen_cold_build_load",
        [*common, "--load-only"],
        run_dir / "logs",
        env=eval_env,
    )
    warm = run_command(
        "codegen_warm_load",
        [*common, "--load-only"],
        run_dir / "logs",
        env=eval_env,
    )
    eval_cmd = [
        *common,
        "--base", str(profile["base"]),
        "--query", str(profile["query"]),
        "--k", str(args.eval_k),
        "--window", str(args.window),
        "--max-queries", str(args.max_queries),
        "--bench",
        "--bench-count", str(args.bench_count),
        "--bench-warmup", str(args.bench_warmup),
    ]
    groundtruth = profile.get("groundtruth")
    if groundtruth is not None and groundtruth.exists():
        eval_cmd.extend(["--groundtruth", str(groundtruth)])
    if args.node_count_values:
        eval_cmd.extend(["--node-count-values", args.node_count_values])
    if args.eval_cache_root is not None:
        args.eval_cache_root.mkdir(parents=True, exist_ok=True)
        truth_cache_name = (
            f"{args.profile}_truth_k{args.eval_k}_"
            f"w{args.window}_q{args.max_queries}.bin"
        )
        model_hash = file_hash64(codegen_dir / "model.bin")
        hash_cache_name = (
            f"{args.profile}_hash_index_{model_hash}_"
            f"h{args.hash_bits}.bin"
        )
        eval_cmd.extend([
            "--truth-cache", str(args.eval_cache_root / truth_cache_name),
            "--hash-index-cache", str(args.eval_cache_root / hash_cache_name),
        ])
    if args.bench_breakdown:
        eval_cmd.append("--bench-breakdown")
    eval_result = run_command(
        "vortex_eval_recall_hash_latency",
        eval_cmd,
        run_dir / "logs",
        env=eval_env,
    )
    return {
        "codegen_cold_build_load": cold,
        "codegen_warm_load": warm,
        "vortex_eval": eval_result,
    }
