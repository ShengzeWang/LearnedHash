#!/usr/bin/env python3
"""Audit and optionally remove ignored local artifacts.

The default mode is a dry run. Deletion requires ``--apply`` and still refuses
to remove tracked files. Dataset roots are reported as preserved and are never
deleted by this helper.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_LIMIT = 200
DATASET_ROOTS = ("dataset", "vector_datasets")
OUTPUT_ROOTS = ("lead_output", "rm_model_output", "vortex_v1_output")
BENCHMARK_PRESERVE_TOKENS = (
    "m53_",
    "m99_",
    "paper",
    "readiness",
    "risk_resolved",
)
VORTEX_ROOT_GENERATED_FILE_SUFFIXES = frozenset({".csr", ".index", ".model"})
VORTEX_ROOT_GENERATED_DIR_NAMES = frozenset(
    {
        "codegen_build_cache",
        "eval_cache",
        "paper_artifacts",
        "selector_scale_validation",
    }
)
VORTEX_ROOT_GENERATED_PRESERVE_NAMES = frozenset(
    {
        "paper_artifacts",
    }
)


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    reason: str
    bytes: int


@dataclass(frozen=True)
class PreservedArtifact:
    path: Path
    kind: str
    reason: str


def run_git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def has_tracked_content(root: Path, path: Path) -> bool:
    rel = relative(path, root)
    result = run_git(root, ["ls-files", "--", rel])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git ls-files failed for {rel}")
    return bool(result.stdout.strip())


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.lstat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.lstat().st_size
        except OSError:
            continue
    return total


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def is_root_build_dir(path: Path) -> bool:
    name = path.name
    return path.is_dir() and (
        name == "build" or name.startswith("build_") or name.startswith("build-")
    )


def benchmark_pack_is_preserved(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in BENCHMARK_PRESERVE_TOKENS)


def iter_benchmark_codegen_build_dirs(pack: Path) -> Iterable[Path]:
    if not pack.exists():
        return
    for path in sorted(pack.rglob("build")):
        if path.is_dir() and path.parent.name == "codegen":
            yield path


def is_vortex_root_generated_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VORTEX_ROOT_GENERATED_FILE_SUFFIXES


def is_vortex_root_generated_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    return (
        name in VORTEX_ROOT_GENERATED_DIR_NAMES
        or name.endswith("_codegen")
        or "_codegen_" in name
    )


def iter_root_selector_reports(root: Path) -> Iterable[Path]:
    output = root / "vortex_v1_output"
    if not output.exists():
        return
    for path in sorted(output.iterdir()):
        if (
            path.is_file()
            and path.suffix in {".json", ".html"}
            and "selector" in path.name
        ):
            yield path


def collect_artifacts(
    root: Path,
    include_benchmark_packs: bool,
    include_benchmark_cache: bool,
    include_vortex_generated: bool,
    preserve_build_dirs: bool,
) -> tuple[list[Artifact], list[PreservedArtifact]]:
    candidates: list[Artifact] = []
    preserved: list[PreservedArtifact] = []

    for name in DATASET_ROOTS:
        path = root / name
        if path.exists():
            preserved.append(
                PreservedArtifact(
                    path=path,
                    kind="dataset-root",
                    reason="dataset roots are never deleted by this helper",
                )
            )

    for name in OUTPUT_ROOTS:
        path = root / name
        if path.exists() and has_tracked_content(root, path):
            preserved.append(
                PreservedArtifact(
                    path=path,
                    kind="tracked-output-root",
                    reason="contains tracked content; refusing automatic cleanup",
                )
            )

    for child in sorted(root.iterdir()):
        if not is_root_build_dir(child):
            continue
        if preserve_build_dirs:
            preserved.append(
                PreservedArtifact(
                    path=child,
                    kind="build-dir",
                    reason="deferred by --preserve-build-dirs",
                )
            )
            continue
        if has_tracked_content(root, child):
            preserved.append(
                PreservedArtifact(
                    path=child,
                    kind="build-dir",
                    reason="contains tracked content; refusing automatic cleanup",
                )
            )
            continue
        candidates.append(
            Artifact(
                path=child,
                kind="build-dir",
                reason="ignored root build directory",
                bytes=path_size(child),
            )
        )

    tools_root = root / "tools"
    if tools_root.exists():
        for cache in sorted(tools_root.rglob("__pycache__")):
            if has_tracked_content(root, cache):
                preserved.append(
                    PreservedArtifact(
                        path=cache,
                        kind="python-cache",
                        reason="contains tracked content; refusing automatic cleanup",
                    )
                )
                continue
            candidates.append(
                Artifact(
                    path=cache,
                    kind="python-cache",
                    reason="ignored Python bytecode cache",
                    bytes=path_size(cache),
                )
            )

    for report in iter_root_selector_reports(root):
        if has_tracked_content(root, report):
            preserved.append(
                PreservedArtifact(
                    path=report,
                    kind="selector-report",
                    reason="tracked selector report; refusing automatic cleanup",
                )
            )
            continue
        candidates.append(
            Artifact(
                path=report,
                kind="selector-report",
                reason="old root-level Vortex selector JSON/HTML report",
                bytes=path_size(report),
            )
        )

    output_root = root / "vortex_v1_output"
    if output_root.exists():
        for artifact in sorted(output_root.iterdir()):
            if artifact.name == "benchmarks":
                continue
            if not (
                is_vortex_root_generated_file(artifact)
                or is_vortex_root_generated_dir(artifact)
            ):
                continue
            kind = (
                "vortex-generated-file"
                if artifact.is_file()
                else "vortex-generated-dir"
            )
            if artifact.name in VORTEX_ROOT_GENERATED_PRESERVE_NAMES:
                preserved.append(
                    PreservedArtifact(
                        path=artifact,
                        kind=kind,
                        reason="preserved benchmark artifact evidence",
                    )
                )
            elif has_tracked_content(root, artifact):
                preserved.append(
                    PreservedArtifact(
                        path=artifact,
                        kind=kind,
                        reason="contains tracked content; refusing automatic cleanup",
                    )
                )
            elif include_vortex_generated:
                candidates.append(
                    Artifact(
                        path=artifact,
                        kind=kind,
                        reason="ignored root Vortex generated artifact selected by --include-vortex-generated",
                        bytes=path_size(artifact),
                    )
                )
            else:
                preserved.append(
                    PreservedArtifact(
                        path=artifact,
                        kind=kind,
                        reason="deferred; pass --include-vortex-generated to include root Vortex generated artifacts",
                    )
                )

    benchmark_root = root / "vortex_v1_output" / "benchmarks"
    if benchmark_root.exists():
        for pack in sorted(path for path in benchmark_root.iterdir() if path.is_dir()):
            if pack.name == "cache":
                if has_tracked_content(root, pack):
                    preserved.append(
                        PreservedArtifact(
                            path=pack,
                            kind="benchmark-cache",
                            reason="contains tracked content; refusing automatic cleanup",
                        )
                    )
                elif include_benchmark_cache:
                    candidates.append(
                        Artifact(
                            path=pack,
                            kind="benchmark-cache",
                            reason="ignored benchmark cache selected by --include-benchmark-cache",
                            bytes=path_size(pack),
                        )
                    )
                else:
                    preserved.append(
                        PreservedArtifact(
                            path=pack,
                            kind="benchmark-cache",
                            reason="deferred; pass --include-benchmark-cache to include cache",
                        )
                    )
                continue
            pack_is_preserved = benchmark_pack_is_preserved(pack)
            pack_has_tracked_content = has_tracked_content(root, pack)
            if pack_is_preserved or pack_has_tracked_content or not include_benchmark_packs:
                for build_dir in iter_benchmark_codegen_build_dirs(pack):
                    if has_tracked_content(root, build_dir):
                        preserved.append(
                            PreservedArtifact(
                                path=build_dir,
                                kind="benchmark-codegen-build",
                                reason="contains tracked content; refusing automatic cleanup",
                            )
                        )
                    else:
                        candidates.append(
                            Artifact(
                                path=build_dir,
                                kind="benchmark-codegen-build",
                                reason="rebuildable generated-code CMake build directory",
                                bytes=path_size(build_dir),
                            )
                        )
            if pack_is_preserved:
                preserved.append(
                    PreservedArtifact(
                        path=pack,
                        kind="benchmark-pack",
                        reason="preserved benchmark evidence",
                    )
                )
                continue
            if pack_has_tracked_content:
                preserved.append(
                    PreservedArtifact(
                        path=pack,
                        kind="benchmark-pack",
                        reason="contains tracked content; refusing automatic cleanup",
                    )
                )
                continue
            if include_benchmark_packs:
                candidates.append(
                    Artifact(
                        path=pack,
                        kind="benchmark-pack",
                        reason="old ignored benchmark pack selected by --include-benchmark-packs",
                        bytes=path_size(pack),
                    )
                )
            else:
                preserved.append(
                    PreservedArtifact(
                        path=pack,
                        kind="benchmark-pack",
                        reason="deferred; pass --include-benchmark-packs to include old packs",
                    )
                )

    safe_candidates = []
    for artifact in candidates:
        if not is_within_root(artifact.path, root):
            preserved.append(
                PreservedArtifact(
                    path=artifact.path,
                    kind=artifact.kind,
                    reason="outside repository root; refusing automatic cleanup",
                )
            )
        elif has_tracked_content(root, artifact.path):
            preserved.append(
                PreservedArtifact(
                    path=artifact.path,
                    kind=artifact.kind,
                    reason="contains tracked content; refusing automatic cleanup",
                )
            )
        else:
            safe_candidates.append(artifact)

    return safe_candidates, preserved


def summarize(artifacts: Iterable[Artifact]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        row = summary.setdefault(artifact.kind, {"count": 0, "bytes": 0})
        row["count"] += 1
        row["bytes"] += artifact.bytes
    return dict(sorted(summary.items()))


def remove_artifact(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def apply_cleanup(root: Path, artifacts: list[Artifact]) -> list[str]:
    deleted: list[str] = []
    for artifact in artifacts:
        if has_tracked_content(root, artifact.path):
            raise RuntimeError(
                f"refusing to delete tracked path: {relative(artifact.path, root)}"
            )
        remove_artifact(artifact.path)
        deleted.append(relative(artifact.path, root))
    return deleted


def report_to_json(
    root: Path,
    candidates: list[Artifact],
    preserved: list[PreservedArtifact],
    deleted: list[str],
    applied: bool,
) -> dict[str, object]:
    return {
        "repository": str(root),
        "mode": "apply" if applied else "dry-run",
        "candidate_count": len(candidates),
        "candidate_bytes": sum(item.bytes for item in candidates),
        "candidate_summary": summarize(candidates),
        "candidates": [
            {
                "path": relative(item.path, root),
                "kind": item.kind,
                "reason": item.reason,
                "bytes": item.bytes,
            }
            for item in candidates
        ],
        "preserved": [
            {
                "path": relative(item.path, root),
                "kind": item.kind,
                "reason": item.reason,
            }
            for item in preserved
        ],
        "deleted": deleted,
    }


def print_text(
    root: Path,
    candidates: list[Artifact],
    preserved: list[PreservedArtifact],
    deleted: list[str],
    applied: bool,
    limit: int,
) -> None:
    total_bytes = sum(item.bytes for item in candidates)
    print("Local artifact cleanup audit")
    print(f"Repository: {root}")
    print(f"Mode: {'apply' if applied else 'dry-run'}")
    print(f"Deletion candidates: {len(candidates)} ({format_bytes(total_bytes)})")

    summary = summarize(candidates)
    if summary:
        print("\nCandidate summary:")
        for kind, row in summary.items():
            print(f"  - {kind}: {row['count']} paths, {format_bytes(row['bytes'])}")
    else:
        print("\nCandidate summary: none")

    if candidates:
        print("\nCandidates:")
        for artifact in candidates[:limit]:
            print(
                f"  - [{artifact.kind}] {relative(artifact.path, root)} "
                f"({format_bytes(artifact.bytes)}) -- {artifact.reason}"
            )
        if len(candidates) > limit:
            print(f"  - ... {len(candidates) - limit} more omitted by --limit")

    if preserved:
        print("\nPreserved / deferred:")
        for item in preserved[:limit]:
            print(f"  - [{item.kind}] {relative(item.path, root)} -- {item.reason}")
        if len(preserved) > limit:
            print(f"  - ... {len(preserved) - limit} more omitted by --limit")

    if applied:
        print(f"\nDeleted: {len(deleted)} paths")
        for path in deleted[:limit]:
            print(f"  - {path}")
        if len(deleted) > limit:
            print(f"  - ... {len(deleted) - limit} more omitted by --limit")
    else:
        print("\nDry run only. Re-run with --apply to delete listed candidates.")
        print(
            "Old benchmark packs are deferred unless --include-benchmark-packs is set."
        )
        print("Benchmark caches are deferred unless --include-benchmark-cache is set.")
        print(
            "Root Vortex generated artifacts are deferred unless --include-vortex-generated is set."
        )
        print("Dataset roots are reported but never deleted by this helper.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script location).",
    )
    parser.add_argument(
        "--include-benchmark-packs",
        action="store_true",
        help=(
            "Include old ignored directories under vortex_v1_output/benchmarks. "
            "Readiness/risk-resolution packs remain preserved."
        ),
    )
    parser.add_argument(
        "--include-benchmark-cache",
        action="store_true",
        help=(
            "Include the reusable cache directory under "
            "vortex_v1_output/benchmarks/cache."
        ),
    )
    parser.add_argument(
        "--include-vortex-generated",
        action="store_true",
        help=(
            "Include ignored root-level Vortex generated artifacts under "
            "vortex_v1_output/, such as *.index, *.csr, *.model, codegen "
            "directories, eval caches, and selector-scale-validation output. "
            "Benchmark packs remain controlled by --include-benchmark-packs."
        ),
    )
    parser.add_argument(
        "--preserve-build-dirs",
        action="store_true",
        help="Preserve root build directories even when applying cleanup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect cleanup candidates without deleting anything (default).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the listed candidates. Omit for the default dry run.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of text."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum paths to print per section in text mode (default: {DEFAULT_LIMIT}).",
    )
    args = parser.parse_args()
    if args.dry_run and args.apply:
        parser.error("--dry-run cannot be combined with --apply")
    return args


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not (root / ".git").exists():
        raise SystemExit(f"not a git repository root: {root}")

    candidates, preserved = collect_artifacts(
        root,
        args.include_benchmark_packs,
        args.include_benchmark_cache,
        args.include_vortex_generated,
        args.preserve_build_dirs,
    )
    deleted: list[str] = []
    if args.apply:
        deleted = apply_cleanup(root, candidates)

    if args.json:
        print(
            json.dumps(
                report_to_json(root, candidates, preserved, deleted, args.apply),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print_text(root, candidates, preserved, deleted, args.apply, max(args.limit, 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
