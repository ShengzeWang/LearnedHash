#!/usr/bin/env python3
"""Regression-check the local artifact cleanup helper.

The cleanup helper intentionally operates on ignored local state, so the checks
build small temporary git repositories instead of touching the working tree.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cleanup_local_artifacts as cleanup


def run_git(root: Path, args: list[str]) -> None:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr.strip() or f"git {' '.join(args)} failed")


def write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_repo() -> tempfile.TemporaryDirectory[str]:
    tmp = tempfile.TemporaryDirectory(prefix="learnedhash-cleanup-test-")
    run_git(Path(tmp.name), ["init", "-q"])
    return tmp


def rel_paths(
    root: Path, items: list[cleanup.Artifact] | list[cleanup.PreservedArtifact]
) -> set[str]:
    return {cleanup.relative(item.path, root) for item in items}


def test_benchmark_pack_and_cache_policy() -> None:
    with make_repo() as tmp:
        root = Path(tmp)
        write_file(root / "build" / "artifact.o")
        write_file(root / "dataset" / "base.fvecs")
        write_file(root / "vector_datasets" / "sift" / "base.fvecs")
        write_file(root / "vortex_v1_output" / "benchmarks" / "cache" / "truth.bin")
        write_file(root / "vortex_v1_output" / "benchmarks" / "old_run" / "summary.csv")
        write_file(
            root
            / "vortex_v1_output"
            / "benchmarks"
            / "m99_sift_baseline_20260521"
            / "summary.csv"
        )
        write_file(
            root
            / "vortex_v1_output"
            / "benchmarks"
            / "m53_sift_readiness_20260520"
            / "summary.csv"
        )

        candidates, preserved = cleanup.collect_artifacts(
            root,
            include_benchmark_packs=True,
            include_benchmark_cache=True,
            include_vortex_generated=False,
            preserve_build_dirs=True,
        )

        candidate_paths = rel_paths(root, candidates)
        preserved_paths = rel_paths(root, preserved)
        expected_candidates = {
            "vortex_v1_output/benchmarks/cache",
            "vortex_v1_output/benchmarks/old_run",
        }
        if candidate_paths != expected_candidates:
            raise AssertionError(
                f"unexpected cleanup candidates: {sorted(candidate_paths)}"
            )
        for expected in (
            "build",
            "dataset",
            "vector_datasets",
            "vortex_v1_output/benchmarks/m99_sift_baseline_20260521",
            "vortex_v1_output/benchmarks/m53_sift_readiness_20260520",
        ):
            if expected not in preserved_paths:
                raise AssertionError(f"expected preserved path missing: {expected}")


def test_cache_cleanup_can_preserve_build_tree() -> None:
    with make_repo() as tmp:
        root = Path(tmp)
        write_file(root / "build" / "artifact.o")
        write_file(root / "vortex_v1_output" / "benchmarks" / "cache" / "truth.bin")

        candidates, preserved = cleanup.collect_artifacts(
            root,
            include_benchmark_packs=False,
            include_benchmark_cache=True,
            include_vortex_generated=False,
            preserve_build_dirs=True,
        )

        if rel_paths(root, candidates) != {"vortex_v1_output/benchmarks/cache"}:
            raise AssertionError(
                "cache-only cleanup should target only the benchmark cache"
            )
        if "build" not in rel_paths(root, preserved):
            raise AssertionError("cache-only cleanup should preserve build/")

        deleted = cleanup.apply_cleanup(root, candidates)
        if deleted != ["vortex_v1_output/benchmarks/cache"]:
            raise AssertionError(f"unexpected deleted paths: {deleted}")
        if not (root / "build").exists():
            raise AssertionError("build/ was deleted during cache-only cleanup")


def test_tracked_content_is_never_deleted() -> None:
    with make_repo() as tmp:
        root = Path(tmp)
        tracked_pack = (
            root / "vortex_v1_output" / "benchmarks" / "old_run" / "summary.csv"
        )
        write_file(tracked_pack)
        run_git(root, ["add", cleanup.relative(tracked_pack, root)])

        candidates, preserved = cleanup.collect_artifacts(
            root,
            include_benchmark_packs=True,
            include_benchmark_cache=True,
            include_vortex_generated=False,
            preserve_build_dirs=False,
        )

        if "vortex_v1_output/benchmarks/old_run" in rel_paths(root, candidates):
            raise AssertionError("tracked benchmark pack was offered for deletion")
        if "vortex_v1_output/benchmarks/old_run" not in rel_paths(root, preserved):
            raise AssertionError("tracked benchmark pack was not reported as preserved")


def test_vortex_root_generated_artifacts_are_explicit_opt_in() -> None:
    with make_repo() as tmp:
        root = Path(tmp)
        write_file(root / "vortex_v1_output" / "sift_benchmark_hnsw.index")
        write_file(root / "vortex_v1_output" / "sift_nsw.csr")
        write_file(root / "vortex_v1_output" / "sift_model.model")
        write_file(root / "vortex_v1_output" / "sift_codegen" / "manifest.json")
        write_file(root / "vortex_v1_output" / "eval_cache" / "hash.bin")
        write_file(
            root / "vortex_v1_output" / "paper_artifacts" / "artifact_manifest.json"
        )
        write_file(root / "vortex_v1_output" / "benchmarks" / "old_run" / "summary.csv")

        candidates, preserved = cleanup.collect_artifacts(
            root,
            include_benchmark_packs=False,
            include_benchmark_cache=False,
            include_vortex_generated=False,
            preserve_build_dirs=False,
        )

        if rel_paths(root, candidates):
            raise AssertionError(
                "root Vortex generated artifacts should be deferred by default"
            )
        preserved_paths = rel_paths(root, preserved)
        for expected in (
            "vortex_v1_output/sift_benchmark_hnsw.index",
            "vortex_v1_output/sift_nsw.csr",
            "vortex_v1_output/sift_model.model",
            "vortex_v1_output/sift_codegen",
            "vortex_v1_output/eval_cache",
            "vortex_v1_output/paper_artifacts",
            "vortex_v1_output/benchmarks/old_run",
        ):
            if expected not in preserved_paths:
                raise AssertionError(f"expected deferred path missing: {expected}")

        candidates, preserved = cleanup.collect_artifacts(
            root,
            include_benchmark_packs=False,
            include_benchmark_cache=False,
            include_vortex_generated=True,
            preserve_build_dirs=False,
        )

        candidate_paths = rel_paths(root, candidates)
        expected_candidates = {
            "vortex_v1_output/sift_benchmark_hnsw.index",
            "vortex_v1_output/sift_nsw.csr",
            "vortex_v1_output/sift_model.model",
            "vortex_v1_output/sift_codegen",
            "vortex_v1_output/eval_cache",
        }
        if candidate_paths != expected_candidates:
            raise AssertionError(
                f"unexpected Vortex generated candidates: {sorted(candidate_paths)}"
            )
        if "vortex_v1_output/paper_artifacts" not in rel_paths(root, preserved):
            raise AssertionError(
                "benchmark artifacts should stay preserved in opt-in cleanup mode"
            )
        if "vortex_v1_output/benchmarks/old_run" not in rel_paths(root, preserved):
            raise AssertionError(
                "benchmark packs should stay controlled by --include-benchmark-packs"
            )


def test_named_benchmark_packs_are_preserved() -> None:
    with make_repo() as tmp:
        root = Path(tmp)
        write_file(
            root
            / "vortex_v1_output"
            / "benchmarks"
            / "m99_current_best"
            / "summary.csv"
        )
        write_file(root / "vortex_v1_output" / "benchmarks" / "old_run" / "summary.csv")

        candidates, preserved = cleanup.collect_artifacts(
            root,
            include_benchmark_packs=True,
            include_benchmark_cache=False,
            include_vortex_generated=False,
            preserve_build_dirs=False,
        )

        if rel_paths(root, candidates) != {"vortex_v1_output/benchmarks/old_run"}:
            raise AssertionError("only unprotected benchmark pack should be candidate")
        if "vortex_v1_output/benchmarks/m99_current_best" not in rel_paths(
            root, preserved
        ):
            raise AssertionError("protected benchmark pack should be preserved")


def test_codegen_build_dirs_inside_preserved_packs_are_cleanable() -> None:
    with make_repo() as tmp:
        root = Path(tmp)
        build_dir = (
            root
            / "vortex_v1_output"
            / "benchmarks"
            / "m99_current"
            / "best_codegen"
            / "codegen"
            / "build"
        )
        write_file(build_dir / "CMakeFiles" / "model.dir" / "model.cpp.o")
        write_file(
            root / "vortex_v1_output" / "benchmarks" / "m99_current" / "summary.csv"
        )

        candidates, preserved = cleanup.collect_artifacts(
            root,
            include_benchmark_packs=False,
            include_benchmark_cache=False,
            include_vortex_generated=False,
            preserve_build_dirs=False,
        )

        candidate_paths = rel_paths(root, candidates)
        expected_build = (
            "vortex_v1_output/benchmarks/m99_current/best_codegen/codegen/build"
        )
        if expected_build not in candidate_paths:
            raise AssertionError("nested generated-code build dir was not cleanable")
        if "vortex_v1_output/benchmarks/m99_current" not in rel_paths(root, preserved):
            raise AssertionError("preserved benchmark pack was not reported")

        deleted = cleanup.apply_cleanup(root, candidates)
        if expected_build not in deleted:
            raise AssertionError(f"nested generated-code build dir not deleted: {deleted}")
        if not (root / "vortex_v1_output" / "benchmarks" / "m99_current" / "summary.csv").exists():
            raise AssertionError("benchmark pack metadata was removed")


def main() -> int:
    test_benchmark_pack_and_cache_policy()
    test_cache_cleanup_can_preserve_build_tree()
    test_tracked_content_is_never_deleted()
    test_vortex_root_generated_artifacts_are_explicit_opt_in()
    test_named_benchmark_packs_are_preserved()
    test_codegen_build_dirs_inside_preserved_packs_are_cleanable()
    print("cleanup local artifacts regression checks: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
