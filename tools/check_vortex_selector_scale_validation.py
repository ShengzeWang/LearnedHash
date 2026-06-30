#!/usr/bin/env python3
"""Regression checks for the D8 selector scale-validation runner."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from path_safety import PathSafetyError
import vortex_selector_scale_validation as scale


def check_quality_status() -> None:
    ok_metrics = {
        "eval_recall_at_k_in_window": 0.71,
        "eval_overlay_match_score": 0.66,
        "eval_node_locality_score": 0.62,
        "materialized_recall_validation_status": "matched",
    }
    status, problems = scale.quality_status(ok_metrics)
    if status != "ok" or problems:
        raise AssertionError(f"valid metrics rejected: {status} {problems}")

    missing_status, missing_problems = scale.quality_status({})
    if missing_status != "unstable" or len(missing_problems) < 4:
        raise AssertionError("missing quality metrics were not rejected")

    mismatch = dict(ok_metrics)
    mismatch["materialized_recall_validation_status"] = "mismatch"
    mismatch_status, mismatch_problems = scale.quality_status(mismatch)
    if mismatch_status != "unstable" or not any("not matched" in item for item in mismatch_problems):
        raise AssertionError("recall validation mismatch was not rejected")


def check_budget_parser() -> None:
    budgets = scale.parse_budget_map("smoke=7,baseline=11,deep=13")
    if budgets != {"smoke": 7, "baseline": 11, "deep": 13}:
        raise AssertionError(f"unexpected budget map: {budgets}")
    try:
        scale.parse_budget_map("smoke=0")
    except Exception:
        pass
    else:
        raise AssertionError("zero budget was accepted")


def check_effective_budget_scaling() -> None:
    sift_preflight = {"manifest": {"files": {"base": {"count": 1_000_000}}}}
    sift_preflight_without_manifest = {"base": {"count": 1_000_000}}
    ten_m_preflight = {"manifest": {"files": {"base": {"count": 10_000_000}}}}
    if scale.effective_budget_seconds("smoke", 60, sift_preflight) != 180:
        raise AssertionError("SIFT1M smoke budget floor should cover cold-candidate guard")
    if scale.effective_budget_seconds("smoke", 60, sift_preflight_without_manifest) != 180:
        raise AssertionError("SIFT1M smoke budget floor should not require an optional manifest")
    if scale.effective_budget_seconds("smoke", 60, ten_m_preflight) != 600:
        raise AssertionError("10M smoke budget floor should cover cold-candidate guard")
    if scale.effective_budget_seconds("baseline", 60, ten_m_preflight) != 1200:
        raise AssertionError("10M baseline budget floor should allow sustained recall search")
    if scale.effective_budget_seconds("deep", 1200, ten_m_preflight) != 3600:
        raise AssertionError("10M deep budget floor should allow hour-scale recall search")
    if scale.effective_budget_seconds("deep", 4000, ten_m_preflight) != 4000:
        raise AssertionError("explicit larger user budgets should be preserved")


def check_command_builder() -> None:
    args = SimpleNamespace(
        build_dir=Path("build"),
        benchmark_output_root=Path("vortex_v1_output/benchmarks"),
        jobs=8,
        selector_parallelism=8,
        candidate_threads=1,
        max_queries=512,
        bench_count=2000,
        bench_warmup=100,
        node_count_values="32,64,128,256,512,1024",
        eval_cache_root=Path("vortex_v1_output/eval_cache"),
        codegen_build_cache_root=Path("vortex_v1_output/codegen_build_cache"),
        skip_build=True,
        force_build=False,
        extra_benchmark_arg=["--materialize-role", "peak_recall"],
    )
    cmd = scale.benchmark_command(args, "sift10m", "smoke", "run", 17)
    joined = " ".join(map(str, cmd))
    for token in (
        "tools/vortex_sift_benchmark_pack.py",
        "--profile sift10m",
        "--preset smoke",
        "--budget-seconds 17",
        "--node-count-values 32,64,128,256,512,1024",
        "--skip-build",
        "--materialize-role peak_recall",
    ):
        if token not in joined:
            raise AssertionError(f"missing command token {token!r}: {joined}")


def check_path_safety() -> None:
    if scale.benchmark_run_name("fixture", "sift10m", "deep") != "fixture_sift10m_deep":
        raise AssertionError("benchmark child run name changed unexpectedly")
    try:
        scale.benchmark_run_name("bad:run", "sift", "smoke")
    except PathSafetyError:
        pass
    else:
        raise AssertionError("unsafe child benchmark run name was accepted")


def check_output_writer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = {
            "run_name": "fixture",
            "status": "ok",
            "run_count": 1,
            "completed_count": 1,
            "stable_count": 1,
            "blocked_count": 0,
            "problem_count": 0,
            "runs": [
                {
                    "profile": "sift",
                    "preset": "smoke",
                    "status": "ok",
                    "quality_status": "ok",
                    "budget_seconds": 60,
                    "eval_recall_at_k_in_window": 0.7,
                    "eval_overlay_match_score": 0.6,
                    "selected_target_skeleton": 30000,
                    "selected_K": 512,
                    "model_file_bytes": 592432,
                    "selector_wall_seconds": 1.2,
                    "candidate_ok_count": 2,
                    "candidate_count": 3,
                    "run_dir": "vortex_v1_output/benchmarks/fixture_sift_smoke",
                }
            ],
        }
        scale.write_outputs(root, summary)
        for name in ("summary.json", "summary.csv", "report.html", "README.md"):
            path = root / name
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"missing output: {name}")
        loaded = json.loads((root / "summary.json").read_text())
        if loaded["status"] != "ok":
            raise AssertionError("summary JSON status not preserved")
        if "Vortex Selector Scale Validation" not in (root / "report.html").read_text():
            raise AssertionError("HTML report title missing")
        html = (root / "report.html").read_text()
        if "Skeleton" not in html or "Model bytes" not in html:
            raise AssertionError("HTML report should include selected model details")


def main() -> int:
    check_quality_status()
    check_budget_parser()
    check_effective_budget_scaling()
    check_command_builder()
    check_path_safety()
    check_output_writer()
    print("vortex selector scale-validation checks: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
