#!/usr/bin/env python3
"""Run bounded D8 selector scale-policy validation suites.

The runner is intentionally a thin orchestrator around the benchmark pack. It
preflights dataset manifests, runs a profile/preset matrix with explicit time
budgets, and writes one compact JSON/CSV/HTML summary focused on selector scale
policy, recall, and overlay-match stability.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from path_safety import PathSafetyError, validate_component, validate_output_path
from vortex_benchmark_pack.common import (
    PROFILE_PATHS,
    ROOT,
    cpu_count,
    fvecs_info,
    profile_manifest_info,
)


DEFAULT_PRESET_BUDGETS = {
    "smoke": 60,
    "baseline": 180,
    "deep": 300,
}
PROFILE_BUDGET_FLOORS = (
    (10_000_000, {"smoke": 600, "baseline": 1200, "deep": 3600}),
    (1_000_000, {"smoke": 180, "baseline": 180, "deep": 300}),
)
QUALITY_METRIC_KEYS = (
    "eval_recall_at_k_in_window",
    "eval_overlay_match_score",
    "eval_node_locality_score",
)


def parse_csv_list(value: str) -> list[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return out


def parse_budget_map(value: str | None) -> dict[str, int]:
    budgets = dict(DEFAULT_PRESET_BUDGETS)
    if not value:
        return budgets
    for item in parse_csv_list(value):
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "--preset-budgets entries must be preset=seconds"
            )
        preset, raw_seconds = item.split("=", 1)
        preset = preset.strip()
        if preset not in budgets:
            raise argparse.ArgumentTypeError(f"unknown preset in --preset-budgets: {preset}")
        try:
            seconds = int(raw_seconds)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid budget seconds for {preset}: {raw_seconds}"
            ) from exc
        if seconds <= 0:
            raise argparse.ArgumentTypeError(f"budget for {preset} must be > 0")
        budgets[preset] = seconds
    return budgets


def utc_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_summary_csv(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = row.get("metric")
            if key:
                out[key] = parse_scalar(row.get("value", ""))
    return out


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        return float(value)
    except ValueError:
        return value


def finite_unit(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0.0 <= value <= 1.0


def preflight_base_count(preflight: dict[str, Any] | None) -> int:
    manifest = (preflight or {}).get("manifest") or {}
    files = manifest.get("files") or {}
    base = files.get("base") or {}
    count = base.get("count")
    if isinstance(count, int) and count > 0:
        return count
    base = (preflight or {}).get("base") or {}
    count = base.get("count")
    return count if isinstance(count, int) and count > 0 else 0


def effective_budget_seconds(preset: str,
                             requested_seconds: int,
                             preflight: dict[str, Any] | None) -> int:
    budget = requested_seconds
    dataset_count = preflight_base_count(preflight)
    for threshold, floors in PROFILE_BUDGET_FLOORS:
        if dataset_count >= threshold:
            budget = max(budget, floors[preset])
            break
    return budget


def quality_status(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    problems: list[str] = []
    for key in QUALITY_METRIC_KEYS:
        value = metrics.get(key)
        if not finite_unit(value):
            problems.append(f"{key} missing or outside [0,1]: {value!r}")
    recall_validation = metrics.get("materialized_recall_validation_status")
    if recall_validation != "matched":
        problems.append(
            "materialized_recall_validation_status is not matched: "
            f"{recall_validation!r}"
        )
    return ("ok" if not problems else "unstable"), problems


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=parse_csv_list, default=parse_csv_list("sift,sift10m"),
                        help="Comma-separated benchmark-pack profiles to validate.")
    parser.add_argument("--presets", type=parse_csv_list, default=parse_csv_list("smoke,baseline,deep"),
                        help="Comma-separated presets to run for each profile.")
    parser.add_argument("--preset-budgets", default="",
                        help="Override selector budgets, e.g. smoke=60,baseline=240,deep=600.")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "vortex_v1_output" / "selector_scale_validation")
    parser.add_argument("--benchmark-output-root", type=Path,
                        default=ROOT / "vortex_v1_output" / "benchmarks")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--selector-parallelism", type=int, default=0)
    parser.add_argument("--candidate-threads", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=512)
    parser.add_argument("--bench-count", type=int, default=2000)
    parser.add_argument("--bench-warmup", type=int, default=100)
    parser.add_argument("--node-count-values", default="32,64,128,256,512,1024")
    parser.add_argument("--eval-cache-root", type=Path,
                        default=ROOT / "vortex_v1_output" / "eval_cache")
    parser.add_argument("--codegen-build-cache-root", type=Path,
                        default=ROOT / "vortex_v1_output" / "codegen_build_cache")
    parser.add_argument("--timeout-slack-seconds", type=int, default=900,
                        help="Extra wall-clock allowance beyond selector budget for build/load/eval.")
    parser.add_argument("--skip-build", action="store_true",
                        help="Pass --skip-build to each benchmark-pack run.")
    parser.add_argument("--force-build", action="store_true",
                        help="Pass --force-build to each benchmark-pack run.")
    parser.add_argument("--build-missing-hnsw", action="store_true",
                        help="Allow benchmark pack to build a missing HNSW index. Without this, "
                             "missing HNSW indices are reported as blocked.")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Exit 0 even if runs are blocked/failed/unstable; useful for planning reports.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only write planned commands; do not execute benchmark-pack runs.")
    parser.add_argument("--extra-benchmark-arg", action="append", default=[],
                        help="Additional raw argument passed through to every benchmark-pack run. "
                             "Repeat for each token.")
    return parser


def normalize_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def benchmark_run_name(parent_run: str, profile: str, preset: str) -> str:
    return validate_component(
        f"{parent_run}_{profile}_{preset}",
        label="benchmark child run name",
    )


def benchmark_command(args: argparse.Namespace,
                      profile: str,
                      preset: str,
                      run_name: str,
                      budget_seconds: int) -> list[str]:
    cmd = [
        sys.executable,
        "tools/vortex_sift_benchmark_pack.py",
        "--profile", profile,
        "--preset", preset,
        "--run-name", run_name,
        "--overwrite",
        "--build-dir", str(args.build_dir),
        "--output-root", str(args.benchmark_output_root),
        "--budget-seconds", str(budget_seconds),
        "--jobs", str(args.jobs),
        "--selector-parallelism", str(args.selector_parallelism),
        "--candidate-threads", str(args.candidate_threads),
        "--max-queries", str(args.max_queries),
        "--bench-count", str(args.bench_count),
        "--bench-warmup", str(args.bench_warmup),
        "--node-count-values", args.node_count_values,
        "--eval-cache-root", str(args.eval_cache_root),
        "--codegen-build-cache-root", str(args.codegen_build_cache_root),
    ]
    if args.skip_build:
        cmd.append("--skip-build")
    if args.force_build:
        cmd.append("--force-build")
    cmd.extend(args.extra_benchmark_arg)
    return cmd


def run_command(cmd: list[str], timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "wall_seconds": time.perf_counter() - started,
        "timed_out": False,
    }


def run_command_with_timeout(cmd: list[str], timeout_seconds: int) -> dict[str, Any]:
    try:
        return run_command(cmd, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": exc.stdout or "",
            "wall_seconds": timeout_seconds,
            "timed_out": True,
        }


def profile_preflight(profile_name: str) -> dict[str, Any]:
    if profile_name not in PROFILE_PATHS:
        raise RuntimeError(f"unknown profile: {profile_name}")
    profile = {key: Path(value) for key, value in PROFILE_PATHS[profile_name].items()}
    manifest = profile_manifest_info(profile_name, profile)
    base_info = fvecs_info(profile["base"]) if profile["base"].is_file() else None
    hnsw_path = profile.get("hnsw_index")
    return {
        "profile": profile_name,
        "manifest": manifest,
        "base": base_info,
        "hnsw_index_path": str(hnsw_path) if hnsw_path else None,
        "hnsw_index_exists": bool(hnsw_path and hnsw_path.exists()),
        "hnsw_index_bytes": hnsw_path.stat().st_size if hnsw_path and hnsw_path.exists() else None,
    }


def make_blocked_row(profile: str,
                     preset: str,
                     reason: str,
                     preflight: dict[str, Any] | None,
                     cmd: list[str] | None = None) -> dict[str, Any]:
    return {
        "profile": profile,
        "preset": preset,
        "status": "blocked",
        "quality_status": "unavailable",
        "reason": reason,
        "command": shlex.join(cmd or []),
        "preflight": preflight,
    }


def collect_completed_row(profile: str,
                          preset: str,
                          budget_seconds: int,
                          run_dir: Path,
                          cmd: list[str],
                          result: dict[str, Any]) -> dict[str, Any]:
    summary_path = run_dir / "summary.csv"
    manifest_path = run_dir / "manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return {
            "profile": profile,
            "preset": preset,
            "status": "failed",
            "quality_status": "unavailable",
            "reason": "benchmark-pack completed without summary.csv/manifest.json",
            "budget_seconds": budget_seconds,
            "command": shlex.join(cmd),
            "returncode": result.get("returncode"),
            "timed_out": result.get("timed_out"),
            "wall_seconds": result.get("wall_seconds"),
            "run_dir": str(run_dir),
        }
    metrics = read_summary_csv(summary_path)
    q_status, q_problems = quality_status(metrics)
    status = "ok" if q_status == "ok" else "unstable"
    return {
        "profile": profile,
        "preset": preset,
        "status": status,
        "quality_status": q_status,
        "quality_problems": q_problems,
        "budget_seconds": budget_seconds,
        "command": shlex.join(cmd),
        "returncode": result.get("returncode"),
        "timed_out": result.get("timed_out"),
        "wall_seconds": result.get("wall_seconds"),
        "run_dir": str(run_dir),
        "summary_csv": str(summary_path),
        "manifest_json": str(manifest_path),
        "report_html": str(run_dir / "report.html"),
        "selector_wall_seconds": metrics.get("selector_wall_seconds"),
        "candidate_count": metrics.get("candidate_count"),
        "candidate_ok_count": metrics.get("candidate_ok_count"),
        "best_recall_at_k_in_window": metrics.get("best_recall_at_k_in_window"),
        "eval_recall_at_k_in_window": metrics.get("eval_recall_at_k_in_window"),
        "eval_overlay_match_score": metrics.get("eval_overlay_match_score"),
        "eval_node_locality_score": metrics.get("eval_node_locality_score"),
        "materialized_recall_validation_status": metrics.get(
            "materialized_recall_validation_status"
        ),
        "selected_target_skeleton": metrics.get("materialized_config_target_skeleton"),
        "selected_K": metrics.get("materialized_config_K"),
        "model_file_bytes": metrics.get("model_file_bytes"),
        "dataset_manifest_path": metrics.get("dataset_manifest_path"),
        "dataset_manifest_sha256": metrics.get("dataset_manifest_sha256"),
    }


def write_outputs(run_dir: Path, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = summary["runs"]
    csv_keys = [
        "profile", "preset", "status", "quality_status", "reason", "run_dir",
        "budget_seconds", "selector_wall_seconds", "candidate_ok_count", "candidate_count",
        "best_recall_at_k_in_window", "eval_recall_at_k_in_window",
        "eval_overlay_match_score", "eval_node_locality_score",
        "materialized_recall_validation_status", "selected_target_skeleton",
        "selected_K", "model_file_bytes", "wall_seconds",
    ]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_html(run_dir / "report.html", summary)
    write_readme(run_dir / "README.md", summary)


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_html(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for row in summary["runs"]:
        cells = [
            row.get("profile"),
            row.get("preset"),
            row.get("status"),
            row.get("quality_status"),
            row.get("budget_seconds"),
            row.get("eval_recall_at_k_in_window"),
            row.get("eval_overlay_match_score"),
            row.get("selected_target_skeleton"),
            row.get("selected_K"),
            row.get("model_file_bytes"),
            row.get("selector_wall_seconds"),
            row.get("candidate_ok_count"),
            row.get("candidate_count"),
            row.get("run_dir") or row.get("reason"),
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(fmt(cell))}</td>" for cell in cells) + "</tr>")
    html_text = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Vortex Selector Scale Validation</title>
<style>
:root {{ --ink:#111827; --muted:#64748b; --line:#d7ded1; --paper:#f6f7f3; --panel:#fff; --ok:#166534; --bad:#991b1b; }}
body {{ margin:0; padding:24px; background:var(--paper); color:var(--ink); font:14px/1.45 \"Avenir Next\", \"Segoe UI\", sans-serif; font-variant-numeric:tabular-nums; }}
.wrap {{ max-width:1280px; margin:0 auto; display:grid; gap:16px; }}
.hero {{ background:#101827; color:#f8fafc; border-radius:8px; padding:20px 22px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; }}
.card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
.label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; font-weight:700; }}
.value {{ font-size:22px; font-weight:750; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ border-bottom:1px solid var(--line); padding:.45rem .55rem; text-align:left; vertical-align:top; }} th {{ background:#f8fafc; }} code {{ background:#eef1ea; padding:.1rem .25rem; border-radius:.2rem; }}
</style></head><body><div class=\"wrap\">
<section class=\"hero\"><h1>Vortex Selector Scale Validation</h1><div>Run: <code>{html.escape(summary['run_name'])}</code> · status: <strong>{html.escape(summary['status'])}</strong></div></section>
<section class=\"grid\">
<div class=\"card\"><div class=\"label\">Completed</div><div class=\"value\">{summary['completed_count']}/{summary['run_count']}</div></div>
<div class=\"card\"><div class=\"label\">Stable Quality</div><div class=\"value\">{summary['stable_count']}/{summary['run_count']}</div></div>
<div class=\"card\"><div class=\"label\">Blocked</div><div class=\"value\">{summary['blocked_count']}</div></div>
<div class=\"card\"><div class=\"label\">Failed/Unstable</div><div class=\"value\">{summary['problem_count']}</div></div>
</section>
<section class=\"panel\"><h2>Profile/Preset Results</h2><table><thead><tr><th>Profile</th><th>Preset</th><th>Status</th><th>Quality</th><th>Budget s</th><th>Recall</th><th>Overlay Match</th><th>Skeleton</th><th>K</th><th>Model bytes</th><th>Selector s</th><th>OK</th><th>Total</th><th>Artifact / Reason</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class=\"panel\"><h2>Policy</h2><p>The validation is strict: missing HNSW indices, missing manifests, absent recall/overlay metrics, or selector/eval recall mismatches mark the run blocked or unstable instead of silently passing.</p></section>
</div></body></html>"""
    path.write_text(html_text, encoding="utf-8")


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Vortex Selector Scale Validation",
        "",
        f"- run_name: `{summary['run_name']}`",
        f"- status: `{summary['status']}`",
        f"- completed: `{summary['completed_count']}/{summary['run_count']}`",
        f"- stable_quality: `{summary['stable_count']}/{summary['run_count']}`",
        "",
        "## Runs",
        "",
    ]
    for row in summary["runs"]:
        lines.append(
            f"- `{row['profile']}/{row['preset']}`: status=`{row['status']}`, "
            f"quality=`{row.get('quality_status')}`, budget_s=`{row.get('budget_seconds')}`, "
            f"recall=`{row.get('eval_recall_at_k_in_window')}`, "
            f"overlay=`{row.get('eval_overlay_match_score')}`, "
            f"skeleton=`{row.get('selected_target_skeleton')}`, K=`{row.get('selected_K')}`, "
            f"model_bytes=`{row.get('model_file_bytes')}`, "
            f"artifact=`{row.get('run_dir') or row.get('reason')}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.output_root = normalize_path(args.output_root)
    args.benchmark_output_root = normalize_path(args.benchmark_output_root)
    args.build_dir = normalize_path(args.build_dir)
    args.eval_cache_root = normalize_path(args.eval_cache_root)
    args.codegen_build_cache_root = normalize_path(args.codegen_build_cache_root)
    args.jobs = args.jobs if args.jobs > 0 else cpu_count()
    args.selector_parallelism = args.selector_parallelism if args.selector_parallelism > 0 else cpu_count()
    try:
        validate_output_path(args.output_root, root=ROOT, label="--output-root")
        validate_output_path(
            args.benchmark_output_root,
            root=ROOT,
            label="--benchmark-output-root",
        )
        validate_output_path(args.eval_cache_root, root=ROOT, label="--eval-cache-root")
        validate_output_path(
            args.codegen_build_cache_root,
            root=ROOT,
            label="--codegen-build-cache-root",
        )
    except PathSafetyError as exc:
        parser.error(str(exc))
    if args.candidate_threads <= 0:
        parser.error("--candidate-threads must be > 0")
    if args.timeout_slack_seconds < 0:
        parser.error("--timeout-slack-seconds must be >= 0")
    if args.skip_build and args.force_build:
        parser.error("use only one of --skip-build or --force-build")
    budgets = parse_budget_map(args.preset_budgets)
    for preset in args.presets:
        if preset not in budgets:
            parser.error(f"unsupported preset: {preset}")
    for profile in args.profiles:
        if profile not in PROFILE_PATHS:
            parser.error(f"unsupported profile: {profile}")

    run_name = args.run_name or f"d8_scale_validation_{utc_slug()}"
    try:
        run_name = validate_component(run_name, label="--run-name", max_chars=96)
        run_dir = validate_output_path(
            args.output_root / run_name,
            root=ROOT,
            label="selector scale-validation run directory",
        )
    except PathSafetyError as exc:
        parser.error(str(exc))
    runs: list[dict[str, Any]] = []
    preflights: dict[str, Any] = {}

    for profile in args.profiles:
        try:
            preflight = profile_preflight(profile)
        except Exception as exc:
            preflight = {"profile": profile, "error": str(exc)}
        preflights[profile] = preflight
        missing_hnsw = not preflight.get("hnsw_index_exists")
        for preset in args.presets:
            child_run = benchmark_run_name(run_name, profile, preset)
            budget = effective_budget_seconds(preset, budgets[preset], preflight)
            cmd = benchmark_command(args, profile, preset, child_run, budget)
            if "error" in preflight:
                runs.append(make_blocked_row(profile, preset, preflight["error"], preflight, cmd))
                continue
            if missing_hnsw and not args.build_missing_hnsw:
                runs.append(make_blocked_row(
                    profile,
                    preset,
                    "missing HNSW index; rerun with --build-missing-hnsw after confirming resource budget",
                    preflight,
                    cmd,
                ))
                continue
            if args.dry_run:
                runs.append({
                    "profile": profile,
                    "preset": preset,
                    "status": "planned",
                    "quality_status": "unavailable",
                    "budget_seconds": budget,
                    "command": shlex.join(cmd),
                    "preflight": preflight,
                })
                continue
            timeout = budget + args.timeout_slack_seconds
            result = run_command_with_timeout(cmd, timeout)
            if result.get("returncode") != 0 or result.get("timed_out"):
                status = "timed_out" if result.get("timed_out") else "failed"
                log_path = run_dir / "logs" / f"{profile}_{preset}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(result.get("stdout") or "", encoding="utf-8", errors="replace")
                runs.append({
                    "profile": profile,
                    "preset": preset,
                    "status": status,
                    "quality_status": "unavailable",
                    "reason": f"benchmark-pack {status}; see {log_path}",
                    "budget_seconds": budget,
                    "command": shlex.join(cmd),
                    "returncode": result.get("returncode"),
                    "timed_out": result.get("timed_out"),
                    "wall_seconds": result.get("wall_seconds"),
                    "log": str(log_path),
                    "preflight": preflight,
                })
                continue
            child_dir = args.benchmark_output_root / child_run
            runs.append(collect_completed_row(profile, preset, budget, child_dir, cmd, result))

    completed_count = sum(1 for row in runs if row.get("status") == "ok")
    stable_count = sum(1 for row in runs if row.get("quality_status") == "ok")
    blocked_count = sum(1 for row in runs if row.get("status") == "blocked")
    problem_count = sum(
        1 for row in runs
        if row.get("status") not in {"ok", "planned"}
    )
    status = "ok" if completed_count == len(runs) and stable_count == len(runs) else "incomplete"
    summary = {
        "schema_version": 1,
        "run_name": run_name,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "profiles": args.profiles,
        "presets": args.presets,
        "requested_preset_budgets": {preset: budgets[preset] for preset in args.presets},
        "profile_budget_floors": [
            {"min_base_count": threshold, "budgets": floors}
            for threshold, floors in PROFILE_BUDGET_FLOORS
        ],
        "preflights": preflights,
        "runs": runs,
        "run_count": len(runs),
        "completed_count": completed_count,
        "stable_count": stable_count,
        "blocked_count": blocked_count,
        "problem_count": problem_count,
        "dry_run": args.dry_run,
    }
    write_outputs(run_dir, summary)
    print(f"selector_scale_validation_dir: {run_dir}")
    print(f"summary_json: {run_dir / 'summary.json'}")
    print(f"summary_csv: {run_dir / 'summary.csv'}")
    print(f"report_html: {run_dir / 'report.html'}")
    print(f"status: {status}")
    return 0 if status == "ok" or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
