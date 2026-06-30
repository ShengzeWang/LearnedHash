#!/usr/bin/env python3
"""Check source package contents.

A source archive should contain the library source, documentation, tests,
and required third-party attribution while excluding generated datasets, models,
build outputs, and local logs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, NamedTuple


PUBLIC_DATASET_METADATA = frozenset(
    (
        "dataset/README.md",
        "dataset/SOURCES.md",
    )
)


REQUIRED_GITIGNORE_PATTERNS = (
    "/build/",
    "/build_warnings/",
    "/CMakeCache.txt",
    "/CMakeFiles/",
    "/CTestTestfile.cmake",
    "/Makefile",
    "/.cmake/",
    "/cmake_install.cmake",
    "/compile_commands.json",
    "/dataset/*",
    "!/dataset/",
    "!/dataset/README.md",
    "!/dataset/SOURCES.md",
    "/vector_datasets/",
    "/lead_output/",
    "/rm_model_output/",
    "/vortex_v1_output/",
    ".research/",
    "CMakeUserPresets.json",
)

REQUIRED_EXPORT_IGNORE_PATHS = (
    "AGENTS.md",
    "CMakeUserPresets.json",
    "CMakeCache.txt",
    "CMakeFiles",
    "CMakeFiles/example-placeholder",
    "CTestTestfile.cmake",
    "Makefile",
    ".cmake",
    ".cmake/example-placeholder",
    "cmake_install.cmake",
    "compile_commands.json",
    "docs/internal",
    "docs/internal/README.md",
    "tasks",
    "tasks/README.md",
    "dataset/example-placeholder",
    "vector_datasets",
    "vector_datasets/example-placeholder",
    "lead_output",
    "lead_output/example-placeholder",
    "rm_model_output",
    "rm_model_output/example-placeholder",
    "vortex_v1_output",
    "vortex_v1_output/example-placeholder",
    "build",
    "build/example-placeholder",
    "build_warnings",
    "build_warnings/example-placeholder",
    ".research",
    ".research/example-placeholder",
)

REQUIRED_SOURCE_PACKAGE_FILES = (
    "README.md",
    "CMakeLists.txt",
    "CMakePresets.json",
    "LICENSE",
    "LearnedHash.svg",
    "cmake/faiss-config.cmake.in",
    "cmake/LearnedHashConfig.cmake.in",
    "docs/README.md",
    "docs/public/README.md",
    "docs/public/faiss_provider.md",
    "docs/public/quickstart.md",
    "docs/public/release_checklist.md",
    "docs/public/tool_reference.md",
    "docs/public/vortex_project.md",
    "docs/public/vortex_benchmark_pack.md",
    "docs/public/vortex_generated_code_api.md",
    "docs/public/vortex_model_selector.md",
    "dataset/README.md",
    "dataset/SOURCES.md",
    "tools/audit_faiss_vendor.py",
    "tools/check_git_hygiene.py",
    "tools/check_install_export.py",
    "tools/check_cleanup_local_artifacts.py",
    "tools/check_source_package.py",
    "tools/check_release_ready.py",
    "tools/path_safety.py",
    "tools/check_vortex_benchmark_pack_schema.py",
    "tools/check_vortex_dataset_prepare.py",
    "tools/check_vortex_paper_artifact_bundle.py",
    "tools/check_vortex_selector_scale_validation.py",
    "tools/vortex_dataset_prepare.py",
    "tools/vortex_dataset_prepare_lib/__init__.py",
    "tools/vortex_dataset_prepare_lib/binary_io.py",
    "tools/vortex_dataset_prepare_lib/cli.py",
    "tools/vortex_dataset_prepare_lib/contracts.py",
    "tools/vortex_dataset_prepare_lib/conversion.py",
    "tools/vortex_dataset_prepare_lib/download.py",
    "tools/vortex_dataset_prepare_lib/file_ops.py",
    "tools/vortex_dataset_prepare_lib/manifest.py",
    "tools/vortex_dataset_prepare_lib/materialize.py",
    "tools/vortex_dataset_prepare_lib/paths.py",
    "tools/vortex_dataset_prepare_lib/specs.py",
    "tools/vortex_dataset_prepare_lib/validation.py",
    "tools/vortex_paper_artifact_bundle.py",
    "tools/vortex_selector_scale_validation.py",
)

REQUIRED_EXECUTABLE_FILES = (
    "tools/audit_faiss_vendor.py",
    "tools/check_git_hygiene.py",
    "tools/check_install_export.py",
    "tools/check_cleanup_local_artifacts.py",
    "tools/check_source_package.py",
    "tools/check_release_ready.py",
    "tools/check_vortex_benchmark_pack_schema.py",
    "tools/check_vortex_dataset_prepare.py",
    "tools/check_vortex_paper_artifact_bundle.py",
    "tools/check_vortex_selector_scale_validation.py",
    "tools/cleanup_local_artifacts.py",
    "tools/vortex_dataset_prepare.py",
    "tools/vortex_paper_artifact_bundle.py",
    "tools/vortex_selector_scale_validation.py",
    "tools/vortex_sift_benchmark_pack.py",
)

TRACKED_GENERATED_PREFIXES = (
    "build/",
    "build_warnings/",
    "CMakeCache.txt",
    "CMakeFiles/",
    "CTestTestfile.cmake",
    "Makefile",
    ".cmake/",
    "cmake_install.cmake",
    "compile_commands.json",
    "dataset/",
    "vector_datasets/",
    "lead_output/",
    "rm_model_output/",
    "vortex_v1_output/",
    ".research/",
)

ARCHIVE_FORBIDDEN_PREFIXES = (
    "build/",
    "build_warnings/",
    "CMakeCache.txt",
    "CMakeFiles/",
    "CTestTestfile.cmake",
    "Makefile",
    ".cmake/",
    "cmake_install.cmake",
    "compile_commands.json",
    "dataset/",
    "vector_datasets/",
    "lead_output/",
    "rm_model_output/",
    "vortex_v1_output/",
    "docs/internal/",
    "tasks/",
    ".research/",
    "CMakeUserPresets.json",
)

SOURCE_DOC_GLOBS = (
    "README.md",
    "dataset/*.md",
    "docs/README.md",
    "docs/public/*.md",
    "hash_functions_core/*/README.md",
    "learned_structures/*/README.md",
)


def token(*parts: str) -> str:
    return "".join(parts)


FORBIDDEN_DOC_DETAIL_TOKENS = (
    token("co", "dex"),
    token("chat", "g", "pt"),
    token("clau", "de"),
    token("co", "pilot"),
    token("cur", "sor"),
    token("l", "lm"),
    token("g", "pt"),
    token("ai ", "agent"),
    token("coding ", "agent"),
    token("agent ", "instruction"),
    token("ae-", "review"),
    token("review", "er"),
    token("review", "er-safe"),
    token("review", "er-facing"),
    token("internal ", "mock"),
    token("pre", "tend"),
    token("re", "hearsal"),
    token("source-", "backed"),
    token("fail-", "closed"),
    token("paper-", "facing"),
    token("paper-", "artifact"),
    token("paper ", "artifact"),
    token("paper-", "result"),
    token("paper ", "result"),
    token("reported-", "benchmark"),
    "tasks/history.md",
    "tasks/lessons.md",
    "tasks/todo.md",
    "docs/internal/papers",
    "deep_cleanup_roadmap",
    "vortexhash_construction_training_nsdi_draft",
    token("/us", "ers/"),
    token("private ", "key"),
    token("api ", "key"),
    token("pass", "word"),
    token("un", "published"),
)


class ArchiveCheck(NamedTuple):
    problems: list[str]
    entries: set[str]
    missing_required: list[str]


def run_git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def tracked_files(root: Path) -> list[str]:
    result = run_git(root, ["ls-files"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return result.stdout.splitlines()


def tracked_file_modes(root: Path) -> dict[str, str]:
    result = run_git(root, ["ls-files", "-s"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files -s failed")
    modes: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4:
            modes[parts[3]] = parts[0]
    return modes


def untracked_required_source_files(root: Path, paths: set[str]) -> list[str]:
    return [
        path
        for path in REQUIRED_SOURCE_PACKAGE_FILES
        if path not in paths and (root / path).is_file()
    ]


def source_doc_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in SOURCE_DOC_GLOBS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def check_required_gitignore(root: Path) -> list[str]:
    lines = set(read_lines(root / ".gitignore"))
    missing = [
        pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in lines
    ]
    return [f".gitignore missing required pattern: {pattern}" for pattern in missing]


def check_required_source_files(root: Path) -> list[str]:
    problems: list[str] = []
    for rel in REQUIRED_SOURCE_PACKAGE_FILES:
        path = root / rel
        if not path.is_file():
            problems.append(f"required source file is missing: {rel}")
    return problems


def check_required_executable_files(
    root: Path, tracked_modes: dict[str, str]
) -> list[str]:
    problems: list[str] = []
    for rel in REQUIRED_EXECUTABLE_FILES:
        path = root / rel
        if not path.is_file():
            problems.append(f"required executable script is missing: {rel}")
            continue
        if not os.access(path, os.X_OK):
            problems.append(f"required executable script is not executable: {rel}")
        mode = tracked_modes.get(rel)
        if mode != "100755":
            mode_label = mode or "untracked"
            problems.append(
                "required executable script is not marked executable in Git "
                f"index: {rel} (mode {mode_label})"
            )
    return problems


def check_export_ignore(root: Path) -> list[str]:
    problems: list[str] = []
    for path in REQUIRED_EXPORT_IGNORE_PATHS:
        result = run_git(root, ["check-attr", "export-ignore", "--", path])
        if result.returncode != 0:
            problems.append(
                f"git check-attr failed for {path}: {result.stderr.strip()}"
            )
            continue
        if "export-ignore: set" not in result.stdout:
            problems.append(f"{path} is not covered by export-ignore")
    return problems


def check_tracked_generated(paths: Iterable[str]) -> list[str]:
    return [
        f"tracked generated/local path should not be in source package: {path}"
        for path in paths
        if path.startswith(TRACKED_GENERATED_PREFIXES)
        and path not in PUBLIC_DATASET_METADATA
    ]


def archive_entry_forbidden(name: str) -> bool:
    if name.rstrip("/") == "dataset":
        return False
    if name in PUBLIC_DATASET_METADATA:
        return False
    if name.startswith("dataset/"):
        return True
    normalized = name.rstrip("/") + ("/" if name.endswith("/") else "")
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in ARCHIVE_FORBIDDEN_PREFIXES
    )


def check_archive_payload(root: Path) -> ArchiveCheck:
    problems: list[str] = []
    entries: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="learnedhash-source-") as tmp_dir:
        archive_path = Path(tmp_dir) / "source.tar"
        result = run_git(
            root,
            [
                "archive",
                "--worktree-attributes",
                "--format=tar",
                "--output",
                str(archive_path),
                "HEAD",
            ],
        )
        if result.returncode != 0:
            return ArchiveCheck(
                problems=[f"git archive failed: {result.stderr.strip()}"],
                entries=entries,
                missing_required=list(REQUIRED_SOURCE_PACKAGE_FILES),
            )
        try:
            with tarfile.open(archive_path, "r") as tar:
                entries = set(tar.getnames())
                forbidden = sorted(
                    name for name in entries if archive_entry_forbidden(name)
                )
        except tarfile.TarError as exc:
            return ArchiveCheck(
                problems=[f"generated archive could not be read: {exc}"],
                entries=entries,
                missing_required=list(REQUIRED_SOURCE_PACKAGE_FILES),
            )
    problems.extend(
        f"source archive contains excluded path: {name}" for name in forbidden
    )
    missing_required = [path for path in REQUIRED_SOURCE_PACKAGE_FILES if path not in entries]
    return ArchiveCheck(
        problems=problems,
        entries=entries,
        missing_required=missing_required,
    )


def check_source_docs(root: Path) -> list[str]:
    problems: list[str] = []
    for path in source_doc_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        rel = path.relative_to(root)
        for token in FORBIDDEN_DOC_DETAIL_TOKENS:
            if token.lower() in lowered:
                problems.append(f"{rel} contains excluded detail token: {token}")
    return problems


def build_report(root: Path, strict_archive: bool = False) -> dict[str, object]:
    paths = tracked_files(root)
    modes = tracked_file_modes(root)
    tracked = set(paths)
    problems: list[str] = []
    warnings: list[str] = []
    problems.extend(check_required_gitignore(root))
    problems.extend(check_required_source_files(root))
    problems.extend(check_required_executable_files(root, modes))
    problems.extend(check_export_ignore(root))
    problems.extend(check_tracked_generated(paths))
    untracked_required = untracked_required_source_files(root, tracked)
    warnings.extend(
        f"required source file is not tracked and will not be in git archive HEAD: {path}"
        for path in untracked_required
    )
    archive = check_archive_payload(root)
    problems.extend(archive.problems)
    untracked_required_set = set(untracked_required)
    warnings.extend(
        f"HEAD source archive is missing required source file: {path}"
        for path in archive.missing_required
        if path not in untracked_required_set and (root / path).is_file()
    )
    problems.extend(check_source_docs(root))
    if strict_archive:
        problems.extend(warnings)
        warnings = []
    return {
        "repository": str(root),
        "source_package_ready": not problems,
        "tracked_files": len(paths),
        "required_source_files": len(REQUIRED_SOURCE_PACKAGE_FILES),
        "required_executable_files": len(REQUIRED_EXECUTABLE_FILES),
        "archive_ref": "HEAD",
        "archive_entry_count": len(archive.entries),
        "strict_archive": strict_archive,
        "untracked_required_source_files": untracked_required,
        "archive_missing_required_source_files": archive.missing_required,
        "warnings": warnings,
        "problems": problems,
    }


def print_text(report: dict[str, object]) -> None:
    print("Source package audit")
    print(f"Repository: {report['repository']}")
    print(f"Tracked files: {report['tracked_files']}")
    print(f"Required source files: {report['required_source_files']}")
    print(f"Required executable scripts: {report['required_executable_files']}")
    print(f"Archive ref: {report['archive_ref']}")
    print(f"Archive entries: {report['archive_entry_count']}")
    print(f"Strict archive: {str(report['strict_archive']).lower()}")
    print(f"Ready: {str(report['source_package_ready']).lower()}")
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
        print("\nNo source-package problems found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script location).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of text."
    )
    parser.add_argument(
        "--strict-archive",
        action="store_true",
        help=(
            "Fail if the committed HEAD archive is missing required source files. "
            "Use after committing a release candidate."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    report = build_report(root, strict_archive=args.strict_archive)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)
    return 0 if report["source_package_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
