#!/usr/bin/env python3
"""Audit the local FAISS dependency surface.

The audit is intentionally conservative. It reports first-party FAISS includes,
vendored subtree sizes, local license files, FAISS CMake source/header lists, and
transitive FAISS include closure. Conditional FAISS CMake appends are counted
when their files exist, so the closure is a safe upper bound. The audit does not
delete vendor files; use the output to decide whether a later explicit trim patch
is safe.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Iterable, Optional


DIRECT_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"](?P<header>faiss/[^>"]+)[>"]')
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"](?P<header>[^>"]+)[>"]')
CMAKE_LIST_START_RE = re.compile(r'^\s*(?P<kind>set|list\s*\(\s*APPEND)\s*\(?\s*(?P<name>[A-Za-z0-9_]+)\b(?P<rest>.*)$')
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".ipp",
}
CMAKE_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
CMAKE_HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".ipp"}
FIRST_PARTY_ROOTS = (
    "hash_functions_core",
    "learned_structures",
    "tests",
)
OPTIONAL_VENDOR_SUBTREES = (
    "gpu",
    "python",
    "cppcontrib",
    "docs",
)
LICENSE_GLOBS = (
    "LICENSE*",
    "COPYING*",
    "NOTICE*",
    "README*",
)


def iter_source_files(root: Path) -> Iterable[Path]:
    for relative in FIRST_PARTY_ROOTS:
        base = root / relative
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                yield path


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def git_tracked_count(root: Path, relative: str) -> Optional[int]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", relative],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return sum(
        1
        for line in result.stdout.splitlines()
        if line and (root / line).exists()
    )


def relative_to_root(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def collect_direct_includes(root: Path) -> dict[str, list[str]]:
    includes: dict[str, list[str]] = {}
    for path in iter_source_files(root):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            match = DIRECT_INCLUDE_RE.match(line)
            if not match:
                continue
            header = match.group("header")
            location = f"{path.relative_to(root)}:{lineno}"
            includes.setdefault(header, []).append(location)
    return dict(sorted(includes.items()))


def find_license_files(vendor_root: Path, root: Path) -> list[str]:
    matches: list[Path] = []
    for pattern in LICENSE_GLOBS:
        matches.extend(vendor_root.rglob(pattern))
    return sorted({str(path.relative_to(root)) for path in matches if path.is_file()})


def strip_cmake_comment(line: str) -> str:
    # The vendored FAISS source/header lists do not use quoted '#' in paths.
    return line.split("#", 1)[0].strip()


def cmake_tokens(text: str) -> list[str]:
    cleaned = text.replace("(", " ").replace(")", " ")
    return [token.strip() for token in cleaned.split() if token.strip()]


def collect_cmake_list(root: Path, variable: str) -> tuple[list[str], list[str]]:
    cmake_path = root / "include" / "faiss" / "CMakeLists.txt"
    if not cmake_path.exists():
        return [], ["include/faiss/CMakeLists.txt"]

    entries: set[str] = set()
    missing: set[str] = set()
    active = False
    buffer = ""
    try:
        lines = cmake_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return [], ["include/faiss/CMakeLists.txt"]

    for line in lines:
        line = strip_cmake_comment(line)
        if not line:
            continue
        if not active:
            start = CMAKE_LIST_START_RE.match(line)
            if not start or start.group("name") != variable:
                continue
            active = True
            buffer = start.group("rest")
        else:
            buffer += " " + line

        if ")" not in line:
            continue

        for token in cmake_tokens(buffer):
            if token in {"set", "list", "APPEND", variable}:
                continue
            suffix = Path(token).suffix
            if suffix not in CMAKE_SOURCE_SUFFIXES and suffix not in CMAKE_HEADER_SUFFIXES:
                continue
            candidate = root / "include" / "faiss" / token
            if candidate.exists():
                entries.add(relative_to_root(root, candidate))
            else:
                missing.add(f"include/faiss/{token}")
        active = False
        buffer = ""

    return sorted(entries), sorted(missing)


def resolve_faiss_include(root: Path, vendor_root: Path, including_file: Path, header: str) -> Optional[Path]:
    if header.startswith("faiss/"):
        candidate = root / "include" / header
        return candidate if candidate.exists() else None

    # FAISS uses both public-prefix includes and local quoted includes in the
    # vendor tree. Resolve local quotes relative to the including file first.
    local_candidate = including_file.parent / header
    if local_candidate.exists() and vendor_root in local_candidate.resolve().parents:
        return local_candidate

    vendor_candidate = vendor_root / header
    if vendor_candidate.exists():
        return vendor_candidate

    return None


def collect_include_closure(root: Path, seed_rel_paths: Iterable[str]) -> dict[str, object]:
    vendor_root = root / "include" / "faiss"
    seen: set[Path] = set()
    missing: dict[str, list[str]] = {}
    queue: deque[Path] = deque()

    for rel in seed_rel_paths:
        path = root / rel
        if path.exists() and path.is_file():
            queue.append(path.resolve())
        else:
            missing.setdefault(rel, []).append("seed")

    while queue:
        path = queue.popleft()
        if path in seen:
            continue
        seen.add(path)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            missing.setdefault(relative_to_root(root, path), []).append("read_error")
            continue

        for lineno, line in enumerate(lines, start=1):
            match = INCLUDE_RE.match(line)
            if not match:
                continue
            header = match.group("header")
            resolved = resolve_faiss_include(root, vendor_root, path, header)
            if resolved is None:
                if header.startswith("faiss/"):
                    missing.setdefault(header, []).append(f"{relative_to_root(root, path)}:{lineno}")
                continue
            resolved = resolved.resolve()
            if resolved not in seen:
                queue.append(resolved)

    closure = sorted(relative_to_root(root, path) for path in seen)
    return {
        "files": closure,
        "file_count": len(closure),
        "missing_includes": dict(sorted(missing.items())),
        "missing_count": sum(len(locations) for locations in missing.values()),
    }


def summarize_optional_subtree_closure(root: Path, closure_files: Iterable[str]) -> dict[str, dict[str, object]]:
    closure_set = set(closure_files)
    summary: dict[str, dict[str, object]] = {}
    for subtree in OPTIONAL_VENDOR_SUBTREES:
        prefix = f"include/faiss/{subtree}/"
        in_closure = sorted(path for path in closure_set if path.startswith(prefix))
        summary[f"include/faiss/{subtree}"] = {
            "closure_files": len(in_closure),
            "referenced_by_closure": bool(in_closure),
            "sample": in_closure[:20],
        }
    return summary


def build_report(root: Path) -> dict[str, object]:
    vendor_root = root / "include" / "faiss"
    direct_includes = collect_direct_includes(root)
    optional_subtrees = {}
    for subtree in OPTIONAL_VENDOR_SUBTREES:
        path = vendor_root / subtree
        optional_subtrees[f"include/faiss/{subtree}"] = {
            "files": count_files(path),
            "tracked_files": git_tracked_count(root, f"include/faiss/{subtree}"),
        }

    cmake_sources, missing_cmake_sources = collect_cmake_list(root, "FAISS_SRC")
    cmake_headers, missing_cmake_headers = collect_cmake_list(root, "FAISS_HEADERS")
    first_party_include_seeds = sorted(
        str(Path("include") / header) for header in direct_includes.keys()
    )
    closure_seeds = sorted(set(cmake_sources) | set(cmake_headers) | set(first_party_include_seeds))
    include_closure = collect_include_closure(root, closure_seeds)
    optional_closure = summarize_optional_subtree_closure(root, include_closure["files"])
    closure_ready = (
        vendor_root.exists()
        and not missing_cmake_sources
        and not missing_cmake_headers
        and include_closure["missing_count"] == 0
    )
    optional_trim_candidates = sorted(
        subtree for subtree, stats in optional_closure.items()
        if not stats["referenced_by_closure"]
        and (
            optional_subtrees[subtree]["files"] > 0
            or (optional_subtrees[subtree]["tracked_files"] or 0) > 0
        )
    )
    optional_remaining_files = sum(
        int(stats["files"]) for stats in optional_subtrees.values()
    )
    optional_remaining_tracked_files = sum(
        int(stats["tracked_files"] or 0) for stats in optional_subtrees.values()
    )
    trim_ready = (
        closure_ready
        and optional_remaining_files == 0
        and optional_remaining_tracked_files == 0
    )

    report: dict[str, object] = {
        "repository": str(root),
        "vendor_root": "include/faiss",
        "vendor_exists": vendor_root.exists(),
        "vendor_files": count_files(vendor_root),
        "vendor_tracked_files": git_tracked_count(root, "include/faiss"),
        "license_files": find_license_files(vendor_root, root) if vendor_root.exists() else [],
        "direct_first_party_includes": direct_includes,
        "first_party_include_seeds": first_party_include_seeds,
        "optional_vendor_subtrees": optional_subtrees,
        "cmake_source_list": {
            "count": len(cmake_sources),
            "paths": cmake_sources,
            "missing": missing_cmake_sources,
        },
        "cmake_header_list": {
            "count": len(cmake_headers),
            "paths": cmake_headers,
            "missing": missing_cmake_headers,
        },
        "transitive_include_closure": include_closure,
        "optional_subtree_closure": optional_closure,
        "optional_subtree_trim_candidates": optional_trim_candidates,
        "optional_subtree_remaining_files": optional_remaining_files,
        "optional_subtree_remaining_tracked_files": optional_remaining_tracked_files,
        "closure_ready": closure_ready,
        "trim_ready": trim_ready,
        "trim_blocker": (
            "none"
            if trim_ready
            else (
                "CMake source/header and transitive include closure must be clean "
                "and optional GPU/Python/cppcontrib/docs subtrees must remain "
                "absent from the bundled CPU FAISS snapshot."
            )
        ),
    }
    return report


def print_text_report(report: dict[str, object]) -> None:
    print("FAISS vendor audit")
    print(f"Repository: {report['repository']}")
    print(f"Vendor root: {report['vendor_root']}")
    print(f"Vendor files: {report['vendor_files']} (tracked: {report['vendor_tracked_files']})")

    print("\nLicense/attribution files:")
    license_files = report["license_files"]
    if license_files:
        for path in license_files:
            print(f"  - {path}")
    else:
        print("  - MISSING")

    print("\nDirect first-party FAISS includes:")
    direct_includes = report["direct_first_party_includes"]
    if direct_includes:
        for header, locations in direct_includes.items():
            print(f"  - {header}")
            for location in locations:
                print(f"      {location}")
    else:
        print("  - none")

    print("\nCMake FAISS target lists:")
    source_list = report["cmake_source_list"]
    header_list = report["cmake_header_list"]
    print(f"  - sources: {source_list['count']} files, missing: {len(source_list['missing'])}")
    print(f"  - headers: {header_list['count']} files, missing: {len(header_list['missing'])}")

    closure = report["transitive_include_closure"]
    print("\nTransitive FAISS include closure:")
    print(f"  - seeds: {len(set(report['first_party_include_seeds']) | set(source_list['paths']) | set(header_list['paths']))}")
    print(f"  - closure files: {closure['file_count']}")
    print(f"  - missing includes: {closure['missing_count']}")

    print("\nOptional vendor subtree sizes:")
    optional_closure = report["optional_subtree_closure"]
    for subtree, stats in report["optional_vendor_subtrees"].items():
        closure_stats = optional_closure[subtree]
        print(
            f"  - {subtree}: {stats['files']} files (tracked: {stats['tracked_files']}), "
            f"closure refs: {closure_stats['closure_files']}"
        )

    print("\nTrim readiness:")
    print(f"  - closure_ready: {str(report['closure_ready']).lower()}")
    print(f"  - ready: {str(report['trim_ready']).lower()}")
    print(f"  - optional files remaining: {report['optional_subtree_remaining_files']}")
    print(f"  - optional tracked files remaining: {report['optional_subtree_remaining_tracked_files']}")
    candidates = report["optional_subtree_trim_candidates"]
    if candidates:
        print("  - unreferenced optional subtree candidates:")
        for subtree in candidates:
            print(f"      {subtree}")
    else:
        print("  - unreferenced optional subtree candidates: none")
    print(f"  - blocker: {report['trim_blocker']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script location).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    report = build_report(root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    return 0 if report["license_files"] and report["closure_ready"] and report["trim_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
