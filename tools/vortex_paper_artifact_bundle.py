#!/usr/bin/env python3
"""Create a validated Vortex benchmark artifact bundle.

The bundle is intentionally metadata-first: it records source dataset manifests,
validation reports, exact commands, environment, git state, report paths,
checksums, and final metric summaries without copying raw datasets, generated
models, or large cache directories. Use --copy-reports to copy small report and
manifest files into the bundle for portability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from path_safety import (
    PathSafetyError,
    safe_component,
    validate_component,
    validate_output_path,
)


SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path("vortex_v1_output/paper_artifacts")
FINAL_METRIC_COLUMNS = (
    "profile",
    "preset",
    "status",
    "quality_status",
    "run_dir",
    "dataset_manifest_path",
    "dataset_manifest_sha256",
    "dataset_base_count",
    "dataset_query_count",
    "dim",
    "selected_target_skeleton",
    "selected_K",
    "selector_wall_seconds",
    "wall_seconds",
    "candidate_count",
    "candidate_ok_count",
    "eval_overlay_match_score",
    "eval_node_locality_score",
    "eval_recall_at_k_in_window",
    "best_recall_at_k_in_window",
    "materialized_recall_validation_status",
    "model_file_bytes",
    "codegen_package_bytes",
    "hash_latency_avg_ms",
    "hash_latency_p95_ms",
    "hash_latency_p99_ms",
    "dominant_cost",
    "dominant_cost_wall_seconds",
)


class BundleError(RuntimeError):
    """Raised when an input artifact is missing or internally inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root(default: Path | None = None) -> Path:
    if default is not None:
        return default.resolve()
    return Path(__file__).resolve().parents[1]


def run_capture(cmd: list[str], *, cwd: Path, timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_info(root: Path) -> dict[str, Any]:
    status = run_capture(
        ["git", "status", "--short", "--untracked-files=all"], cwd=root
    )
    status_lines = status.splitlines() if status else []
    return {
        "commit": run_capture(["git", "rev-parse", "HEAD"], cwd=root),
        "branch": run_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root),
        "dirty": bool(status_lines),
        "status_short": status_lines,
    }


def environment_info(root: Path) -> dict[str, Any]:
    cmake_version = run_capture(["cmake", "--version"], cwd=root)
    ctest_version = run_capture(["ctest", "--version"], cwd=root)
    return {
        "created_utc": utc_now(),
        "cwd": str(root),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cmake_version": cmake_version.splitlines()[0] if cmake_version else None,
        "ctest_version": ctest_version.splitlines()[0] if ctest_version else None,
    }


def resolve_input_path(path: str | Path, *, root: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def relative_to_or_none(path: Path, root: Path) -> str | None:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(
    path: str | Path,
    *,
    root: Path,
    label: str,
    kind: str,
    scope: str = "source",
    required: bool = True,
    copy_eligible: bool = True,
) -> dict[str, Any]:
    resolved = resolve_input_path(path, root=root)
    if required and not resolved.is_file():
        raise BundleError(f"required {kind} file is missing: {resolved}")
    record: dict[str, Any] = {
        "label": label,
        "kind": kind,
        "scope": scope,
        "path": str(resolved),
        "relative_path": relative_to_or_none(resolved, root),
        "exists": resolved.is_file(),
        "copy_eligible": copy_eligible,
    }
    if resolved.is_file():
        record["bytes"] = resolved.stat().st_size
        record["sha256"] = sha256_file(resolved)
    return record


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise BundleError(f"invalid JSON in {path}: {exc}") from exc


def read_metric_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["metric", "value"]:
            return {}
        return {str(row["metric"]): str(row.get("value", "")) for row in reader}


def get_nested(mapping: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def safe_label(label: str) -> str:
    return safe_component(label, fallback="artifact")


def copied_report_name(label: str, source_name: str) -> str:
    source = safe_component(source_name, fallback="file", max_chars=80)
    stem_budget = max(16, 180 - len(source) - 1)
    stem = safe_component(label, fallback="artifact", max_chars=stem_budget)
    return safe_component(f"{stem}_{source}", fallback="artifact", max_chars=180)


def collect_scale_validation(
    scale_dir: Path,
    *,
    root: Path,
    file_records: list[dict[str, Any]],
    commands: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[Path]]:
    scale_dir = resolve_input_path(scale_dir, root=root)
    summary_json = scale_dir / "summary.json"
    summary_csv = scale_dir / "summary.csv"
    report_html = scale_dir / "report.html"
    readme = scale_dir / "README.md"
    prefix = f"scale:{scale_dir.name}"
    for name, path, kind, required in (
        ("summary_json", summary_json, "scale_summary_json", True),
        ("summary_csv", summary_csv, "scale_summary_csv", True),
        ("report_html", report_html, "scale_report_html", True),
        ("readme", readme, "scale_readme", False),
    ):
        file_records.append(
            file_record(path, root=root, label=f"{prefix}:{name}", kind=kind, required=required)
        )
    summary = load_json(summary_json)
    if not isinstance(summary, dict):
        raise BundleError(f"scale summary is not a JSON object: {summary_json}")
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs:
        raise BundleError(f"scale summary contains no runs: {summary_json}")

    run_dirs: list[Path] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise BundleError(f"scale run entry {index} is not an object: {summary_json}")
        label = f"scale:{scale_dir.name}:{run.get('profile', index)}:{run.get('preset', '')}"
        command = run.get("command")
        if command:
            commands.append(
                {
                    "label": label,
                    "source": str(summary_json),
                    "command_line": command,
                    "returncode": run.get("returncode"),
                    "timed_out": run.get("timed_out"),
                    "wall_seconds": run.get("wall_seconds"),
                }
            )
        run_dir_value = run.get("run_dir")
        if run_dir_value:
            run_dirs.append(resolve_input_path(str(run_dir_value), root=root))
        elif run.get("manifest_json"):
            run_dirs.append(resolve_input_path(str(run["manifest_json"]), root=root).parent)
    record = {
        "path": str(scale_dir),
        "relative_path": relative_to_or_none(scale_dir, root),
        "status": summary.get("status"),
        "run_name": summary.get("run_name"),
        "created_utc": summary.get("created_utc"),
        "run_count": summary.get("run_count"),
        "completed_count": summary.get("completed_count"),
        "stable_count": summary.get("stable_count"),
        "problem_count": summary.get("problem_count"),
        "profiles": summary.get("profiles"),
        "presets": summary.get("presets"),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "report_html": str(report_html),
    }
    return record, run_dirs


def command_entries_from_manifest(
    *,
    run_label: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    root: Path,
    file_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    commands = manifest.get("commands", {})
    if not isinstance(commands, dict):
        return entries
    for name, command_info in sorted(commands.items()):
        if not isinstance(command_info, dict):
            continue
        entry = {
            "label": f"{run_label}:{name}",
            "source": str(manifest_path),
            "command": command_info.get("command"),
            "command_line": command_info.get("command_line"),
            "returncode": command_info.get("returncode"),
            "timed_out": command_info.get("timed_out"),
            "wall_seconds": command_info.get("wall_seconds"),
            "peak_rss_kib": command_info.get("peak_rss_kib"),
        }
        log_path = command_info.get("log")
        if log_path:
            try:
                rec = file_record(
                    log_path,
                    root=root,
                    label=f"{run_label}:{name}:log",
                    kind="command_log",
                    required=False,
                    copy_eligible=False,
                )
                file_records.append(rec)
                entry["log"] = rec["path"]
                entry["log_sha256"] = rec.get("sha256")
            except BundleError:
                entry["log"] = str(log_path)
        entries.append(entry)
    return entries


def metric_row(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, str],
    scale_run: dict[str, Any] | None,
) -> dict[str, Any]:
    scale_run = scale_run or {}
    dataset_manifest = get_nested(manifest, ("dataset", "manifest"), {}) or {}
    dominant = manifest.get("dominant_cost", {}) if isinstance(manifest.get("dominant_cost"), dict) else {}
    row = {
        "profile": first_present(manifest.get("profile"), scale_run.get("profile"), metrics.get("profile")),
        "preset": first_present(manifest.get("preset"), scale_run.get("preset"), metrics.get("preset")),
        "status": first_present(scale_run.get("status"), "ok"),
        "quality_status": first_present(scale_run.get("quality_status"), ""),
        "run_dir": str(run_dir),
        "dataset_manifest_path": first_present(dataset_manifest.get("path"), scale_run.get("dataset_manifest_path")),
        "dataset_manifest_sha256": first_present(dataset_manifest.get("sha256"), scale_run.get("dataset_manifest_sha256")),
        "dataset_base_count": first_present(
            get_nested(manifest, ("dataset", "base", "count")), metrics.get("base_count")
        ),
        "dataset_query_count": first_present(
            get_nested(manifest, ("dataset", "query", "count")), metrics.get("query_count")
        ),
        "dim": first_present(get_nested(manifest, ("dataset", "base", "dim")), metrics.get("dim")),
        "selected_target_skeleton": first_present(
            scale_run.get("selected_target_skeleton"), metrics.get("best_config_target_skeleton")
        ),
        "selected_K": first_present(scale_run.get("selected_K"), metrics.get("best_config_K")),
        "selector_wall_seconds": first_present(
            scale_run.get("selector_wall_seconds"), metrics.get("selector_wall_seconds")
        ),
        "wall_seconds": first_present(scale_run.get("wall_seconds"), metrics.get("total_wall_seconds")),
        "candidate_count": first_present(scale_run.get("candidate_count"), metrics.get("candidate_count")),
        "candidate_ok_count": first_present(
            scale_run.get("candidate_ok_count"), metrics.get("candidate_ok_count")
        ),
        "eval_overlay_match_score": first_present(
            scale_run.get("eval_overlay_match_score"), metrics.get("eval_overlay_match_score")
        ),
        "eval_node_locality_score": first_present(
            scale_run.get("eval_node_locality_score"), metrics.get("eval_node_locality_score")
        ),
        "eval_recall_at_k_in_window": first_present(
            scale_run.get("eval_recall_at_k_in_window"), metrics.get("eval_recall_at_k_in_window")
        ),
        "best_recall_at_k_in_window": first_present(
            scale_run.get("best_recall_at_k_in_window"), metrics.get("best_recall_at_k_in_window")
        ),
        "materialized_recall_validation_status": first_present(
            scale_run.get("materialized_recall_validation_status"),
            metrics.get("materialized_recall_validation_status"),
        ),
        "model_file_bytes": first_present(scale_run.get("model_file_bytes"), metrics.get("model_file_bytes")),
        "codegen_package_bytes": first_present(metrics.get("codegen_package_bytes")),
        "hash_latency_avg_ms": first_present(metrics.get("hash_latency_avg_ms")),
        "hash_latency_p95_ms": first_present(metrics.get("hash_latency_p95_ms")),
        "hash_latency_p99_ms": first_present(metrics.get("hash_latency_p99_ms")),
        "dominant_cost": first_present(dominant.get("name"), metrics.get("dominant_cost")),
        "dominant_cost_wall_seconds": first_present(
            dominant.get("wall_seconds"), metrics.get("dominant_cost_wall_seconds")
        ),
    }
    return {column: row.get(column, "") for column in FINAL_METRIC_COLUMNS}


def collect_run(
    run_dir: Path,
    *,
    root: Path,
    file_records: list[dict[str, Any]],
    commands: list[dict[str, Any]],
    scale_run_by_dir: dict[str, dict[str, Any]],
    require_dataset_manifest: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = resolve_input_path(run_dir, root=root)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.csv"
    report_path = run_dir / "report.html"
    for name, path, kind, required in (
        ("manifest_json", manifest_path, "benchmark_manifest_json", True),
        ("summary_csv", summary_path, "benchmark_summary_csv", True),
        ("report_html", report_path, "benchmark_report_html", True),
        ("readme", run_dir / "README.md", "benchmark_readme", False),
        ("selector_json", run_dir / "selector.json", "selector_json", False),
        ("selector_html", run_dir / "selector.html", "selector_html", False),
    ):
        file_records.append(
            file_record(
                path,
                root=root,
                label=f"run:{run_dir.name}:{name}",
                kind=kind,
                required=required,
            )
        )
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise BundleError(f"run manifest is not a JSON object: {manifest_path}")
    if "summary_schema_version" not in manifest:
        raise BundleError(f"run manifest lacks summary_schema_version: {manifest_path}")

    metrics = read_metric_csv(summary_path)
    profile = str(first_present(manifest.get("profile"), metrics.get("profile"), run_dir.name))
    preset = str(first_present(manifest.get("preset"), metrics.get("preset"), ""))
    run_label = f"run:{profile}:{preset}:{run_dir.name}"

    dataset_manifest = get_nested(manifest, ("dataset", "manifest"), {}) or {}
    dataset_manifest_available = bool(dataset_manifest.get("available"))
    dataset_manifest_path = dataset_manifest.get("path")
    if require_dataset_manifest and not dataset_manifest_available:
        raise BundleError(f"dataset manifest is not available for reported benchmark run: {run_dir}")
    dataset_manifest_record: dict[str, Any] | None = None
    if dataset_manifest_path:
        dataset_manifest_record = file_record(
            dataset_manifest_path,
            root=root,
            label=f"{run_label}:dataset_manifest",
            kind="dataset_source_manifest",
            required=dataset_manifest_available or require_dataset_manifest,
        )
        file_records.append(dataset_manifest_record)
        expected_sha = dataset_manifest.get("sha256")
        actual_sha = dataset_manifest_record.get("sha256")
        if expected_sha and actual_sha and expected_sha != actual_sha:
            raise BundleError(
                f"dataset manifest checksum mismatch for {run_dir}: "
                f"manifest records {expected_sha}, actual {actual_sha}"
            )
    commands.extend(
        command_entries_from_manifest(
            run_label=run_label,
            manifest_path=manifest_path,
            manifest=manifest,
            root=root,
            file_records=file_records,
        )
    )

    scale_run = scale_run_by_dir.get(str(run_dir))
    final_metrics = metric_row(
        run_dir=run_dir, manifest=manifest, metrics=metrics, scale_run=scale_run
    )
    run_record = {
        "profile": profile,
        "preset": preset,
        "run_dir": str(run_dir),
        "relative_path": relative_to_or_none(run_dir, root),
        "manifest_json": str(manifest_path),
        "summary_csv": str(summary_path),
        "report_html": str(report_path),
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_file": dataset_manifest_record,
        "git": manifest.get("git"),
        "host": manifest.get("host"),
        "dominant_cost": manifest.get("dominant_cost"),
        "final_metrics": final_metrics,
    }
    return run_record, final_metrics


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FINAL_METRIC_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in FINAL_METRIC_COLUMNS})


def write_commands(path: Path, commands: list[dict[str, Any]]) -> None:
    lines = [
        "# Vortex benchmark artifact command ledger",
        "# Commands are recorded exactly from the source run manifests.",
        "# Review local absolute paths before replaying on another machine.",
        "",
    ]
    for entry in commands:
        lines.append(f"## {entry.get('label', 'command')}")
        if entry.get("wall_seconds") is not None:
            lines.append(f"# wall_seconds={entry['wall_seconds']}")
        if entry.get("returncode") is not None:
            lines.append(f"# returncode={entry['returncode']} timed_out={entry.get('timed_out')}")
        if entry.get("log"):
            lines.append(f"# log={entry['log']}")
        command_line = entry.get("command_line")
        if command_line:
            lines.append(str(command_line))
        elif entry.get("command"):
            lines.append(" ".join(map(str, entry["command"])))
        else:
            lines.append("# command unavailable")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    if not rows:
        return "_No runs recorded._\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def write_bundle_readme(
    path: Path,
    *,
    bundle_name: str,
    manifest: dict[str, Any],
    metric_rows: list[dict[str, Any]],
) -> None:
    git = manifest["git"]
    scale_dirs = manifest["inputs"]["scale_validations"]
    text = [
        f"# Vortex Benchmark Artifact Bundle: {bundle_name}",
        "",
        "This bundle records the metadata needed to connect reported benchmark "
        "numbers back to source manifests, validation reports, exact commands, "
        "environment, git state, report paths, and checksums.",
        "",
        "It intentionally does not copy raw vector datasets, generated models, "
        "shared-library build caches, or other large payloads.",
        "",
        "## Git And Environment",
        "",
        f"- Commit: `{git.get('commit')}`",
        f"- Branch: `{git.get('branch')}`",
        f"- Dirty checkout: `{str(git.get('dirty')).lower()}`",
        f"- Created UTC: `{manifest['created_utc']}`",
        f"- Python: `{manifest['environment'].get('python')}`",
        f"- Platform: `{manifest['environment'].get('platform')}`",
        "",
        "## Source Validation Inputs",
        "",
    ]
    if scale_dirs:
        for scale in scale_dirs:
            text.append(
                f"- `{scale['path']}`: status `{scale.get('status')}`, "
                f"runs `{scale.get('run_count')}`, report `{scale.get('report_html')}`"
            )
    else:
        text.append("- No scale-validation summary was provided.")
    text.extend(
        [
            "",
            "## Final Metrics Summary",
            "",
            markdown_table(
                metric_rows,
                (
                    "profile",
                    "preset",
                    "selected_target_skeleton",
                    "selected_K",
                    "eval_overlay_match_score",
                    "eval_node_locality_score",
                    "eval_recall_at_k_in_window",
                    "selector_wall_seconds",
                    "candidate_ok_count",
                    "model_file_bytes",
                ),
            ).rstrip(),
            "",
            "## Files",
            "",
            "- `artifact_manifest.json`: complete machine-readable bundle index.",
            "- `metrics_summary.csv`: compact final metric table for reports.",
            "- `commands.txt`: exact command ledger from source run manifests.",
            "- `file_checksums.sha256`: SHA-256 checksums for recorded source files "
            "and copied small reports, if `--copy-reports` was used.",
            "- `files/`: optional copied manifests/reports; never raw datasets or "
            "generated model/cache payloads.",
            "",
            "## Verification",
            "",
            "```bash",
            f"tools/vortex_paper_artifact_bundle.py --verify-bundle {path.parent}",
            "```",
            "",
            "For a submission-ready artifact, rerun the bundle after committing the "
            "release candidate and pass `--require-clean-git` so dirty local state "
            "cannot be accidentally reported as final.",
        ]
    )
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def copy_small_reports(
    *,
    bundle_dir: Path,
    root: Path,
    file_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    files_dir = bundle_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for record in list(file_records):
        if record.get("scope") != "source" or not record.get("copy_eligible"):
            continue
        if not record.get("exists"):
            continue
        src = Path(str(record["path"]))
        if not src.is_file():
            continue
        filename = copied_report_name(str(record.get("label", src.stem)), src.name)
        base_filename = filename
        index = 1
        while filename in used:
            suffix = f"_{index}"
            filename = f"{base_filename[:180 - len(suffix)]}{suffix}"
            index += 1
        used.add(filename)
        dst = files_dir / filename
        shutil.copy2(src, dst)
        copied.append(
            file_record(
                dst,
                root=root,
                label=f"copy:{record.get('label')}",
                kind=str(record.get("kind", "artifact_copy")),
                scope="bundle_copy",
                required=True,
                copy_eligible=False,
            )
        )
    file_records.extend(copied)
    return copied


def write_checksums(path: Path, records: list[dict[str, Any]]) -> None:
    lines = []
    for record in records:
        sha = record.get("sha256")
        if not sha:
            continue
        lines.append(f"{sha}  {record['path']}")
    path.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")


def scale_run_map(scale_dirs: list[Path], root: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for scale_dir in scale_dirs:
        summary_path = resolve_input_path(scale_dir, root=root) / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        for run in summary.get("runs", []):
            if isinstance(run, dict) and run.get("run_dir"):
                mapping[str(resolve_input_path(str(run["run_dir"]), root=root))] = run
    return mapping


def build_bundle(
    *,
    root: Path,
    output_root: Path,
    bundle_name: str,
    run_dirs: list[Path],
    scale_validation_dirs: list[Path],
    copy_reports: bool,
    overwrite: bool,
    require_clean_git: bool,
    require_dataset_manifest: bool,
) -> Path:
    root = root.resolve()
    git = git_info(root)
    if require_clean_git and git.get("dirty"):
        raise BundleError("working tree is dirty; rerun without --require-clean-git or commit first")

    output_root = resolve_input_path(output_root, root=root)
    try:
        validate_output_path(output_root, root=root, label="--output-root")
        bundle_name = validate_component(bundle_name, label="--bundle-name")
        bundle_dir = validate_output_path(
            output_root / bundle_name,
            root=root,
            label="benchmark artifact bundle directory",
        )
    except PathSafetyError as exc:
        raise BundleError(str(exc)) from exc
    if bundle_dir.exists():
        if not overwrite:
            raise BundleError(f"bundle directory already exists: {bundle_dir}")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    file_records: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    scale_records: list[dict[str, Any]] = []
    discovered_run_dirs: list[Path] = []
    for scale_dir in scale_validation_dirs:
        scale_record, scale_runs = collect_scale_validation(
            scale_dir, root=root, file_records=file_records, commands=commands
        )
        scale_records.append(scale_record)
        discovered_run_dirs.extend(scale_runs)

    all_run_dirs: list[Path] = []
    seen: set[str] = set()
    for path in [*run_dirs, *discovered_run_dirs]:
        resolved = resolve_input_path(path, root=root)
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            all_run_dirs.append(resolved)
    if not all_run_dirs:
        raise BundleError("no benchmark run directories were provided or discovered")

    scale_by_run = scale_run_map(scale_validation_dirs, root)
    run_records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for run_dir in all_run_dirs:
        run_record, row = collect_run(
            run_dir,
            root=root,
            file_records=file_records,
            commands=commands,
            scale_run_by_dir=scale_by_run,
            require_dataset_manifest=require_dataset_manifest,
        )
        run_records.append(run_record)
        metric_rows.append(row)

    if copy_reports:
        copy_small_reports(bundle_dir=bundle_dir, root=root, file_records=file_records)

    artifact_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bundle_name": bundle_name,
        "bundle_dir": str(bundle_dir),
        "created_utc": utc_now(),
        "repo_root": str(root),
        "git": git,
        "environment": environment_info(root),
        "source_policy": {
            "raw_datasets_copied": False,
            "generated_models_copied": False,
            "report_manifests_copied": copy_reports,
            "dataset_manifest_required": require_dataset_manifest,
        },
        "inputs": {
            "scale_validations": scale_records,
            "benchmark_runs": run_records,
        },
        "commands": commands,
        "final_metrics_summary": metric_rows,
        "file_records": file_records,
    }

    write_metrics_csv(bundle_dir / "metrics_summary.csv", metric_rows)
    write_commands(bundle_dir / "commands.txt", commands)
    write_checksums(bundle_dir / "file_checksums.sha256", file_records)
    write_bundle_readme(
        bundle_dir / "README.md",
        bundle_name=bundle_name,
        manifest=artifact_manifest,
        metric_rows=metric_rows,
    )
    (bundle_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle_dir


def verify_bundle(bundle_dir: Path, *, verify_source_paths: bool = False) -> dict[str, Any]:
    bundle_dir = bundle_dir.expanduser().resolve()
    manifest_path = bundle_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"bundle manifest is missing: {manifest_path}")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise BundleError(f"bundle manifest is not a JSON object: {manifest_path}")
    records = manifest.get("file_records", [])
    if not isinstance(records, list) or not records:
        raise BundleError("bundle manifest has no file_records")
    copied = [record for record in records if record.get("scope") == "bundle_copy"]
    selected = records if verify_source_paths or not copied else copied
    problems: list[str] = []
    checked = 0
    for record in selected:
        if not isinstance(record, dict):
            problems.append("non-object file record")
            continue
        expected = record.get("sha256")
        if not expected:
            continue
        path = Path(str(record.get("path", ""))).expanduser()
        if not path.is_file():
            problems.append(f"missing file: {path}")
            continue
        actual = sha256_file(path)
        checked += 1
        if actual != expected:
            problems.append(f"checksum mismatch: {path}")
    return {
        "bundle_dir": str(bundle_dir),
        "schema_version": manifest.get("schema_version"),
        "checked_files": checked,
        "source_paths_checked": verify_source_paths or not copied,
        "status": "ok" if not problems else "failed",
        "problems": problems,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script location).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where benchmark artifact bundles are written.",
    )
    parser.add_argument(
        "--bundle-name",
        default=f"vortex_benchmark_artifact_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        help="Output bundle directory name.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Benchmark-pack run directory. Can be repeated.",
    )
    parser.add_argument(
        "--scale-validation-dir",
        type=Path,
        action="append",
        default=[],
        help="Selector scale-validation directory; benchmark runs are discovered from summary.json.",
    )
    parser.add_argument(
        "--copy-reports",
        action="store_true",
        help="Copy small manifests/reports/log-independent files into the bundle.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing bundle directory.")
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Fail if git status is dirty. Use for final submission artifacts.",
    )
    parser.add_argument(
        "--allow-missing-dataset-manifest",
        action="store_true",
        help="Allow benchmark runs without source dataset manifests. Not for reported benchmark bundles.",
    )
    parser.add_argument(
        "--verify-bundle",
        type=Path,
        help="Verify an existing bundle instead of creating one.",
    )
    parser.add_argument(
        "--verify-source-paths",
        action="store_true",
        help="With --verify-bundle, verify original source paths instead of copied report files.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON status.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root(args.root)
    started = time.perf_counter()
    try:
        if args.verify_bundle is not None:
            report = verify_bundle(args.verify_bundle, verify_source_paths=args.verify_source_paths)
            report["seconds"] = round(time.perf_counter() - started, 3)
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(f"Vortex benchmark artifact bundle verification: {report['status']}")
                print(f"Bundle: {report['bundle_dir']}")
                print(f"Checked files: {report['checked_files']}")
                for problem in report["problems"]:
                    print(f"  - {problem}")
            return 0 if report["status"] == "ok" else 1

        bundle_dir = build_bundle(
            root=root,
            output_root=args.output_root,
            bundle_name=args.bundle_name,
            run_dirs=args.run_dir,
            scale_validation_dirs=args.scale_validation_dir,
            copy_reports=args.copy_reports,
            overwrite=args.overwrite,
            require_clean_git=args.require_clean_git,
            require_dataset_manifest=not args.allow_missing_dataset_manifest,
        )
        report = {
            "status": "ok",
            "bundle_dir": str(bundle_dir),
            "seconds": round(time.perf_counter() - started, 3),
        }
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("Vortex benchmark artifact bundle: ok")
            print(f"Bundle: {bundle_dir}")
        return 0
    except BundleError as exc:
        report = {
            "status": "failed",
            "error": str(exc),
            "seconds": round(time.perf_counter() - started, 3),
        }
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        else:
            print(f"Vortex benchmark artifact bundle failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
