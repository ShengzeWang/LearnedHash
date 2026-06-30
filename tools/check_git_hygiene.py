#!/usr/bin/env python3
"""Audit Git hygiene for local development and source packaging.

The check is intentionally lightweight: it does not inspect file contents or raw
vector payloads. It validates names, refs, and local safety/performance config so
paths that break source archives or cross-platform checkouts, plus stale conflict
refs, are caught before fetch, archive, or push operations fail.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

from path_safety import (
    BAD_SYNC_TOKENS,
    MAX_SAFE_RELATIVE_CHARS,
    MAX_SYNC_ABSOLUTE_CHARS,
    component_issues,
    relative_path_issues,
)

SAFE_REF_RE = re.compile(r"^[A-Za-z0-9._/+-]+$")


class RecommendedConfig(NamedTuple):
    key: str
    value: str
    reason: str


RECOMMENDED_LOCAL_CONFIGS = (
    RecommendedConfig("core.protectHFS", "true", "reject HFS-colliding paths"),
    RecommendedConfig("core.protectNTFS", "true", "reject reserved Windows paths"),
    RecommendedConfig("core.preloadIndex", "true", "speed up status/diff on large trees"),
    RecommendedConfig("core.untrackedCache", "true", "cache untracked-file scans"),
    RecommendedConfig("core.fsmonitor", "true", "use Git filesystem monitor when available"),
    RecommendedConfig("fetch.prune", "true", "drop deleted remote refs during fetch"),
    RecommendedConfig("pull.ff", "only", "avoid accidental merge commits on pull"),
    RecommendedConfig("push.default", "simple", "push only the current branch upstream"),
    RecommendedConfig("gc.auto", "256", "avoid excessive auto-gc prompts"),
    RecommendedConfig(
        "status.showUntrackedFiles",
        "normal",
        "keep status explicit without expensive full ignored scans",
    ),
)

PROJECTED_OUTPUT_PATHS = (
    "vortex_v1_output/benchmarks/sift_deep_20260526T120000Z/manifest.json",
    "vortex_v1_output/benchmarks/sift10m_deep_20260526T120000Z/best_codegen/centroid_index.bin",
    "vortex_v1_output/benchmarks/deep10m_l2_deep_20260526T120000Z/logs/vortex_eval_recall_hash_latency.log",
    "vortex_v1_output/benchmarks/cache/sift10m_hash_index_0123456789abcdef_h64.bin",
    "vortex_v1_output/selector_scale_validation/d8_scale_validation_20260526_120000/report.html",
    "vortex_v1_output/selector_scale_validation/d8_scale_validation_20260526_120000/logs/deep10m_l2_deep.log",
    "vortex_v1_output/paper_artifacts/vortex_paper_artifact_20260526T120000Z/artifact_manifest.json",
    "vector_datasets/deep10m_l2/deep10m_l2_groundtruth.ivecs",
    "vector_datasets/_sources/spacev1b/msspacev-gt-10M",
)


class GitResult(NamedTuple):
    returncode: int
    stdout: str
    stderr: str


def run_git(root: Path, args: list[str]) -> GitResult:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return GitResult(result.returncode, result.stdout, result.stderr)


def git_z(root: Path, args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip())
    return [
        item.decode("utf-8", "surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]


def normalize_bool(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "on", "1"}:
        return "true"
    if lowered in {"false", "no", "off", "0"}:
        return "false"
    return lowered


def local_config(root: Path, key: str) -> str | None:
    result = run_git(root, ["config", "--local", "--get", key])
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def set_local_config(root: Path, key: str, value: str) -> None:
    result = run_git(root, ["config", "--local", key, value])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git config failed: {key}")


def apply_local_config(root: Path) -> list[dict[str, str]]:
    changed: list[dict[str, str]] = []
    for item in RECOMMENDED_LOCAL_CONFIGS:
        before = local_config(root, item.key)
        if before is not None and normalize_bool(before) == item.value.lower():
            continue
        set_local_config(root, item.key, item.value)
        changed.append(
            {
                "key": item.key,
                "old": "" if before is None else before,
                "new": item.value,
                "reason": item.reason,
            }
        )
    return changed


def check_component(component: str) -> list[str]:
    return component_issues(component)


def check_relative_path(rel: str, absolute_root: Path) -> list[str]:
    return relative_path_issues(rel, absolute_root)


def worktree_paths(root: Path, include_ignored: bool) -> dict[str, list[str]]:
    paths = {
        "tracked": git_z(root, ["ls-files", "-z"]),
        "untracked": git_z(root, ["ls-files", "--others", "--exclude-standard", "-z"]),
    }
    if include_ignored:
        paths["ignored"] = git_z(
            root, ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"]
        )
    return paths


def check_worktree_paths(
    root: Path,
    absolute_root: Path,
    include_ignored: bool,
) -> tuple[list[str], dict[str, object]]:
    problems: list[str] = []
    groups = worktree_paths(root, include_ignored)
    max_rel = (0, "")
    max_abs = (0, "")
    counts: dict[str, int] = {}
    for group, paths in groups.items():
        counts[group] = len(paths)
        for rel in paths:
            rel_len = len(rel)
            abs_len = len(str(absolute_root / rel))
            if rel_len > max_rel[0]:
                max_rel = (rel_len, rel)
            if abs_len > max_abs[0]:
                max_abs = (abs_len, rel)
            for issue in check_relative_path(rel, absolute_root):
                problems.append(f"{group} path {rel}: {issue}")
    summary: dict[str, object] = {
        "counts": counts,
        "max_relative_path": {"chars": max_rel[0], "path": max_rel[1]},
        "max_absolute_path": {"chars": max_abs[0], "path": max_abs[1]},
    }
    return problems, summary


def check_projected_output_paths(
    absolute_root: Path,
) -> tuple[list[str], dict[str, object]]:
    problems: list[str] = []
    max_rel = (0, "")
    max_abs = (0, "")
    records: list[dict[str, object]] = []
    for rel in PROJECTED_OUTPUT_PATHS:
        rel_len = len(rel)
        abs_len = len(str(absolute_root / rel))
        if rel_len > max_rel[0]:
            max_rel = (rel_len, rel)
        if abs_len > max_abs[0]:
            max_abs = (abs_len, rel)
        issues = check_relative_path(rel, absolute_root)
        records.append({"path": rel, "issues": issues})
        for issue in issues:
            problems.append(f"projected output path {rel}: {issue}")
    return problems, {
        "count": len(PROJECTED_OUTPUT_PATHS),
        "max_relative_path": {"chars": max_rel[0], "path": max_rel[1]},
        "max_absolute_path": {"chars": max_abs[0], "path": max_abs[1]},
        "budget": {
            "max_relative_chars": MAX_SAFE_RELATIVE_CHARS,
            "max_absolute_chars": MAX_SYNC_ABSOLUTE_CHARS,
        },
        "paths": records,
    }


def check_refs(root: Path) -> list[str]:
    problems: list[str] = []
    result = run_git(
        root, ["for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes"]
    )
    if result.returncode != 0:
        return [f"git for-each-ref failed: {result.stderr.strip()}"]
    for ref in result.stdout.splitlines():
        ref = ref.strip()
        if not ref:
            continue
        check = run_git(root, ["check-ref-format", ref])
        if check.returncode != 0:
            problems.append(f"active ref has invalid format: {ref}")
        if not SAFE_REF_RE.match(ref):
            problems.append(f"active ref is not ASCII/path-safe: {ref}")
        if any(token in ref.lower() for token in BAD_SYNC_TOKENS):
            problems.append(f"active ref looks like a sync/conflict ref: {ref}")
    return problems


def iter_git_metadata_paths(git_dir: Path) -> Iterable[Path]:
    skip_dirs = {"objects", "lfs", "modules"}
    for dirpath, dirnames, filenames in os.walk(git_dir):
        dirnames[:] = [name for name in dirnames if name not in skip_dirs]
        current = Path(dirpath)
        for name in filenames:
            yield current / name
        for name in dirnames:
            yield current / name


def check_git_metadata(root: Path) -> list[str]:
    problems: list[str] = []
    result = run_git(root, ["rev-parse", "--git-dir"])
    if result.returncode != 0:
        return [f"git rev-parse --git-dir failed: {result.stderr.strip()}"]
    git_dir = Path(result.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = root / git_dir
    for path in iter_git_metadata_paths(git_dir):
        rel = str(path.relative_to(git_dir))
        for component in Path(rel).parts:
            lowered = component.lower()
            if any(token in lowered for token in BAD_SYNC_TOKENS):
                problems.append(f"git metadata has sync/conflict filename: .git/{rel}")
                break
    return sorted(set(problems))


def check_configs(
    root: Path, strict: bool
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    problems: list[str] = []
    warnings: list[str] = []
    report: list[dict[str, str]] = []
    for item in RECOMMENDED_LOCAL_CONFIGS:
        actual = local_config(root, item.key)
        actual_norm = "" if actual is None else normalize_bool(actual)
        ok = actual_norm == item.value.lower()
        entry = {
            "key": item.key,
            "expected": item.value,
            "actual": "" if actual is None else actual,
            "status": "ok" if ok else "missing_or_different",
            "reason": item.reason,
        }
        report.append(entry)
        if not ok:
            message = (
                f"local git config {item.key} should be {item.value} "
                f"({item.reason}); run tools/check_git_hygiene.py --apply-local-config"
            )
            if strict:
                problems.append(message)
            else:
                warnings.append(message)
    return problems, warnings, report


def build_report(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"not a Git worktree root: {root}")
    applied = apply_local_config(root) if args.apply_local_config else []
    absolute_root = args.path_budget_root.resolve() if args.path_budget_root else root
    problems: list[str] = []
    warnings: list[str] = []
    path_problems, path_summary = check_worktree_paths(
        root, absolute_root, not args.skip_ignored
    )
    problems.extend(path_problems)
    projected_summary: dict[str, object] | None = None
    if not args.skip_projected_outputs:
        projected_problems, projected_summary = check_projected_output_paths(
            absolute_root
        )
        problems.extend(projected_problems)
    problems.extend(check_refs(root))
    if not args.skip_git_metadata:
        problems.extend(check_git_metadata(root))
    config_problems, config_warnings, config_report = check_configs(
        root, args.strict_local_config
    )
    problems.extend(config_problems)
    warnings.extend(config_warnings)
    return {
        "repository": str(root),
        "path_budget_root": str(absolute_root),
        "ok": not problems,
        "applied_local_config": applied,
        "path_summary": path_summary,
        "projected_output_paths": projected_summary,
        "local_config": config_report,
        "warnings": warnings,
        "problems": sorted(set(problems)),
    }


def print_text(report: dict[str, object]) -> None:
    print("Git hygiene audit")
    print(f"Repository: {report['repository']}")
    print(f"Path budget root: {report['path_budget_root']}")
    print(f"Ready: {str(report['ok']).lower()}")
    summary = report["path_summary"]
    print(f"Path counts: {summary['counts']}")
    print(f"Max relative path: {summary['max_relative_path']}")
    print(f"Max absolute path: {summary['max_absolute_path']}")
    projected = report.get("projected_output_paths")
    if projected:
        print(f"Projected output paths: {projected['count']}")
        print(f"Projected max relative path: {projected['max_relative_path']}")
        print(f"Projected max absolute path: {projected['max_absolute_path']}")
    applied = report["applied_local_config"]
    if applied:
        print("\nApplied local Git config:")
        for item in applied:
            print(f"  - {item['key']}: {item['old']!r} -> {item['new']!r}")
    warnings = report["warnings"]
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    problems = report["problems"]
    if problems:
        print("\nProblems:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nNo Git hygiene problems found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script location).",
    )
    parser.add_argument(
        "--path-budget-root",
        type=Path,
        help=(
            "Root used for absolute path-length budgeting. Defaults to --root; "
            "pass an alternate checkout root to simulate packaging path budgets."
        ),
    )
    parser.add_argument(
        "--apply-local-config",
        action="store_true",
        help="Apply recommended local Git safety/performance config before checking.",
    )
    parser.add_argument(
        "--strict-local-config",
        action="store_true",
        help="Treat missing recommended local Git config as a failure.",
    )
    parser.add_argument(
        "--skip-ignored",
        action="store_true",
        help="Skip ignored-file path checks. Tracked and untracked files are still checked.",
    )
    parser.add_argument(
        "--skip-git-metadata",
        action="store_true",
        help="Skip .git metadata conflict-name checks.",
    )
    parser.add_argument(
        "--skip-projected-outputs",
        action="store_true",
        help="Skip projected generated-output path checks.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
    except Exception as exc:
        print(f"Git hygiene audit failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
