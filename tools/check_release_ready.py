#!/usr/bin/env python3
"""Run repository readiness checks.

The check uses fresh build directories by default. It validates source-package
contents, Python maintenance scripts, local cleanup safety, bundled FAISS audit,
Git hygiene, benchmark pack schema, dataset preparation schema, selector scale
validation schema, artifact-bundle schema, full Release build/test/install-export,
library-only build/test/install-export, and lean runtime build/test/install-export.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


def run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    steps: list[dict[str, object]],
    name: str,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    started = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    seconds = time.perf_counter() - started
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    steps.append(
        {
            "name": name,
            "kind": "command",
            "command": cmd,
            "returncode": result.returncode,
            "seconds": round(seconds, 3),
            "status": "ok" if result.returncode == 0 else "failed",
        }
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )
    return result


def record_step(
    steps: list[dict[str, object]],
    name: str,
    fn: Callable[[], T],
) -> T:
    started = time.perf_counter()
    try:
        value = fn()
    except Exception as exc:
        steps.append(
            {
                "name": name,
                "kind": "internal",
                "returncode": None,
                "seconds": round(time.perf_counter() - started, 3),
                "status": "failed",
                "error": str(exc),
            }
        )
        raise
    steps.append(
        {
            "name": name,
            "kind": "internal",
            "returncode": None,
            "seconds": round(time.perf_counter() - started, 3),
            "status": "ok",
        }
    )
    return value


def python_files(root: Path) -> list[Path]:
    return sorted((root / "tools").rglob("*.py"))


def check_python_syntax(root: Path) -> None:
    for path in python_files(root):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    print(f"Python syntax check: {len(python_files(root))} files ok")


def run_static_checks(
    root: Path,
    env: dict[str, str],
    steps: list[dict[str, object]],
    strict_source_archive: bool,
) -> None:
    record_step(steps, "python_maintenance_syntax", lambda: check_python_syntax(root))
    run(
        ["git", "diff", "--check"],
        cwd=root,
        env=env,
        steps=steps,
        name="git_diff_check",
    )
    run(
        [sys.executable, "tools/check_git_hygiene.py"],
        cwd=root,
        env=env,
        steps=steps,
        name="git_hygiene",
    )
    source_package_cmd = [sys.executable, "tools/check_source_package.py"]
    if strict_source_archive:
        source_package_cmd.append("--strict-archive")
    run(
        source_package_cmd,
        cwd=root,
        env=env,
        steps=steps,
        name="source_package_audit",
    )
    run(
        [sys.executable, "tools/check_cleanup_local_artifacts.py"],
        cwd=root,
        env=env,
        steps=steps,
        name="cleanup_artifact_hygiene",
    )
    run(
        [sys.executable, "tools/audit_faiss_vendor.py", "--json"],
        cwd=root,
        env=env,
        steps=steps,
        name="faiss_vendor_audit",
    )
    run(
        [sys.executable, "tools/check_vortex_benchmark_pack_schema.py"],
        cwd=root,
        env=env,
        steps=steps,
        name="benchmark_pack_schema",
    )
    run(
        [sys.executable, "tools/check_vortex_dataset_prepare.py"],
        cwd=root,
        env=env,
        steps=steps,
        name="dataset_prepare_schema",
    )
    run(
        [sys.executable, "tools/check_vortex_selector_scale_validation.py"],
        cwd=root,
        env=env,
        steps=steps,
        name="selector_scale_validation_schema",
    )
    run(
        [sys.executable, "tools/check_vortex_paper_artifact_bundle.py"],
        cwd=root,
        env=env,
        steps=steps,
        name="artifact_bundle_schema",
    )


def configure_build_test(
    *,
    root: Path,
    build_dir: Path,
    cmake: str,
    jobs: int,
    build_tools: bool,
    vortex_faiss_components: bool,
    mode: str,
    env: dict[str, str],
    steps: list[dict[str, object]],
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    build_tools_value = "ON" if build_tools else "OFF"
    vortex_faiss_value = "ON" if vortex_faiss_components else "OFF"
    run(
        [
            cmake,
            "-S",
            str(root),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLEARNEDHASH_FAISS_PROVIDER=bundled",
            f"-DLEARNEDHASH_BUILD_TOOLS={build_tools_value}",
            f"-DLEARNEDHASH_BUILD_VORTEX_FAISS_COMPONENTS={vortex_faiss_value}",
        ],
        cwd=root,
        env=env,
        steps=steps,
        name=f"{mode}_configure",
    )
    run(
        [
            cmake,
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--parallel",
            str(jobs),
        ],
        cwd=root,
        env=env,
        steps=steps,
        name=f"{mode}_build",
    )
    run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-C",
            "Release",
            "--output-on-failure",
            "-j",
            str(jobs),
        ],
        cwd=root,
        env=env,
        steps=steps,
        name=f"{mode}_ctest",
    )
    run(
        [
            cmake,
            "--build",
            str(build_dir),
            "--target",
            "learnedhash_check_install_export",
            "--config",
            "Release",
            "--parallel",
            str(jobs),
        ],
        cwd=root,
        env=env,
        steps=steps,
        name=f"{mode}_install_export",
    )


def summarize_steps(steps: list[dict[str, object]]) -> None:
    print("\nReadiness step summary:")
    for step in steps:
        status = step["status"]
        seconds = step["seconds"]
        print(f"  - {step['name']}: {status} ({seconds}s)")


def write_report(report_path: Path, report: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Readiness JSON report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script location).",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        help="Directory for fresh readiness builds. Defaults to a temp dir.",
    )
    parser.add_argument("--cmake", default="cmake", help="CMake executable.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel build/test jobs (default: detected CPU count).",
    )
    parser.add_argument(
        "--keep-builds",
        action="store_true",
        help="Keep temporary build directories for inspection.",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Remove an explicit --build-root before running.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Write a machine-readable readiness report to this path.",
    )
    parser.add_argument(
        "--strict-source-archive",
        action="store_true",
        help=(
            "Pass --strict-archive to the source package audit. Use after the "
            "release candidate files are committed."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not (root / "CMakeLists.txt").exists():
        print(f"repository root does not contain CMakeLists.txt: {root}", file=sys.stderr)
        return 2
    if args.jobs < 1:
        print("--jobs must be >= 1", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    steps: list[dict[str, object]] = []
    started = time.perf_counter()
    report: dict[str, object] = {
        "schema_version": 1,
        "repository": str(root),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cmake": args.cmake,
        "jobs": args.jobs,
        "strict_source_archive": args.strict_source_archive,
        "status": "running",
        "steps": steps,
    }

    temp_build_root: str | None = None
    if args.build_root is None:
        build_root = Path(tempfile.mkdtemp(prefix="learnedhash-readiness-"))
        temp_build_root = str(build_root)
    else:
        build_root = args.build_root.resolve()
        if build_root.exists():
            if not args.force_clean:
                print(
                    f"build root already exists; pass --force-clean to remove it: {build_root}",
                    file=sys.stderr,
                )
                return 2
            shutil.rmtree(build_root)
        build_root.mkdir(parents=True)
    report["build_root"] = str(build_root)

    try:
        print(f"Readiness build root: {build_root}")
        run_static_checks(root, env, steps, args.strict_source_archive)
        configure_build_test(
            root=root,
            build_dir=build_root / "release-bundled",
            cmake=args.cmake,
            jobs=args.jobs,
            build_tools=True,
            vortex_faiss_components=True,
            mode="release_bundled",
            env=env,
            steps=steps,
        )
        configure_build_test(
            root=root,
            build_dir=build_root / "release-libonly",
            cmake=args.cmake,
            jobs=args.jobs,
            build_tools=False,
            vortex_faiss_components=True,
            mode="release_libonly",
            env=env,
            steps=steps,
        )
        configure_build_test(
            root=root,
            build_dir=build_root / "release-lean-runtime",
            cmake=args.cmake,
            jobs=args.jobs,
            build_tools=False,
            vortex_faiss_components=False,
            mode="release_lean_runtime",
            env=env,
            steps=steps,
        )
        report["status"] = "ok"
        report["total_seconds"] = round(time.perf_counter() - started, 3)
        summarize_steps(steps)
        if args.report_json is not None:
            write_report(args.report_json.resolve(), report)
        print("LearnedHash release-readiness gate: ok")
        if args.keep_builds or args.build_root is not None:
            print(f"Build root: {build_root}")
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["total_seconds"] = round(time.perf_counter() - started, 3)
        if steps:
            summarize_steps(steps)
        if args.report_json is not None:
            write_report(args.report_json.resolve(), report)
        print(f"LearnedHash release-readiness gate failed: {exc}", file=sys.stderr)
        if temp_build_root is not None:
            print(f"Failed build root retained for debugging: {build_root}", file=sys.stderr)
            temp_build_root = None
        return 1
    finally:
        if temp_build_root is not None and not args.keep_builds:
            shutil.rmtree(temp_build_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
