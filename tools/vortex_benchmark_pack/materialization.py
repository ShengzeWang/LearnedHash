"""Materialize a selected Vortex model and generated-code package."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from .common import ROOT, dir_size, safe_label
from .command import run_command


def candidate_for_role(selector_json: dict[str, Any], role: str) -> dict[str, Any] | None:
    if role == "best":
        candidate = selector_json.get("best")
    else:
        candidate = (selector_json.get("recommendation_roles") or {}).get(role)
    if isinstance(candidate, dict) and candidate.get("ok", False):
        return candidate
    return None

def resolve_materialization_role(args: argparse.Namespace,
                                 selector_json: dict[str, Any]) -> tuple[str, str]:
    requested = args.materialize_role
    if requested != "auto":
        return requested, requested
    return "auto", "best"

def select_materialization_candidate(args: argparse.Namespace,
                                     selector_json: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    requested, resolved = resolve_materialization_role(args, selector_json)
    candidate = candidate_for_role(selector_json, resolved)
    if candidate is None and requested == "auto":
        resolved = "best"
        candidate = candidate_for_role(selector_json, resolved)
    if candidate is None:
        raise RuntimeError(f"selector did not produce an ok candidate for role '{resolved}'")
    config = candidate.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"selector candidate for role '{resolved}' is missing config")
    return requested, resolved, candidate

def build_materialized_model(
    args: argparse.Namespace,
    paths: dict[str, Path],
    profile: dict[str, Path],
    run_dir: Path,
    selector_json: dict[str, Any],
    config: dict[str, Any],
    materialized_role: str,
) -> dict[str, Any]:
    target_skeleton = int(config.get("target_skeleton") or 0)
    if target_skeleton <= 0:
        raise RuntimeError("benchmark pack expects selector to choose a target_skeleton > 0")

    role_label = safe_label(materialized_role)
    nsw_path = run_dir / f"{args.profile}_{role_label}_ts{target_skeleton}.csr"
    extract = [
        str(paths["vortex_v1_cli"]), "extract_nsw",
        "--index", str(profile["hnsw_index"]),
        "--target_skeleton", str(target_skeleton),
        "--output", str(nsw_path),
    ]
    extract_result = run_command("extract_materialized_nsw", extract, run_dir / "logs")

    codegen_dir = run_dir / f"{role_label}_codegen"
    model_path = run_dir / f"{role_label}.model"
    codegen_name = f"{args.profile}_bench_{role_label}"
    train_threads = args.train_threads if args.train_threads > 0 else args.jobs
    train = [
        str(paths["vortex_v1_cli"]), "train",
        "--dataset", str(profile["base"]),
        "--nsw", str(nsw_path),
        "--K", str(config["K"]),
        "--hash_bits", str(args.hash_bits),
        "--cdf_models", str(config["cdf_models"]),
        "--cdf_branch", str(config["cdf_branch"]),
        "--centroid_knn", str(config["centroid_knn"]),
        "--two_opt_iters", str(config["two_opt_iters"]),
        "--seed", str(args.seed),
        "--threads", str(train_threads),
        "--output", str(model_path),
        "--codegen",
        "--codegen-dir", str(codegen_dir),
        "--codegen-name", codegen_name,
    ]
    train.append("--enable_2opt" if config.get("enable_2opt", True) else "--disable_2opt")
    if args.disable_full_dataset_assignments:
        train.append("--disable-full-dataset-assignments")
    selection = selector_json.get("selection") or {}
    if (args.disable_graph_centroid_order or
            selection.get("graph_centroid_order") is False or
            config.get("graph_centroid_order") is False):
        train.append("--disable_graph_centroid_order")
    assignment_sample_limit = selection.get("assignment_sample_limit")
    if isinstance(assignment_sample_limit, int) and assignment_sample_limit > 0:
        train.extend(["--assignment-sample-limit", str(assignment_sample_limit)])
    env = os.environ.copy()
    env["VORTEX_ROOT"] = str(ROOT)
    train_threads_fallback = False
    try:
        train_result = run_command("train_codegen_materialized", train, run_dir / "logs", env=env)
    except RuntimeError:
        if train_threads <= 1:
            raise
        if model_path.exists():
            model_path.unlink()
        if codegen_dir.exists():
            shutil.rmtree(codegen_dir)
        fallback_train = list(train)
        threads_index = fallback_train.index("--threads") + 1
        fallback_train[threads_index] = "1"
        train_result = run_command(
            "train_codegen_materialized_retry_threads1",
            fallback_train,
            run_dir / "logs",
            env=env,
        )
        train_threads = 1
        train_threads_fallback = True

    model_bin = codegen_dir / "model.bin"
    centroid_index = codegen_dir / "centroid_index.bin"
    return {
        "extract_nsw": extract_result,
        "train_codegen": train_result,
        "materialized_role": materialized_role,
        "nsw_path": str(nsw_path),
        "model_path": str(model_path),
        "codegen_dir": str(codegen_dir),
        "codegen_name": codegen_name,
        "train_threads": train_threads,
        "train_threads_fallback": train_threads_fallback,
        "model_file_bytes": model_path.stat().st_size if model_path.exists() else None,
        "codegen_model_bin_bytes": model_bin.stat().st_size if model_bin.exists() else None,
        "codegen_centroid_index_bytes": centroid_index.stat().st_size if centroid_index.exists() else None,
        "codegen_package_bytes": dir_size(codegen_dir),
    }
