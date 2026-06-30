#!/usr/bin/env python3
"""Regression checks for the Vortex benchmark artifact bundle helper."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "vortex_paper_artifact_bundle.py"


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def write_fixture(root: Path) -> tuple[Path, Path]:
    dataset_manifest = root / "dataset_manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "dataset": "Fixture1M",
                "metric": "L2",
                "slice_rule": "official_fixture_split",
                "groundtruth_exact": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_sha = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()

    run_dir = root / "benchmarks" / "fixture_sift_baseline"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    (logs / "selector.log").write_text("selector ok\n", encoding="utf-8")
    (run_dir / "report.html").write_text("<h1>fixture report</h1>\n", encoding="utf-8")
    (run_dir / "README.md").write_text("# fixture run\n", encoding="utf-8")
    (run_dir / "selector.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "selector.html").write_text("<h1>selector</h1>\n", encoding="utf-8")
    write_csv(
        run_dir / "summary.csv",
        [
            ("base_count", "1000000"),
            ("query_count", "10000"),
            ("dim", "128"),
            ("best_config_K", "512"),
            ("best_config_target_skeleton", "30000"),
            ("selector_wall_seconds", "12.5"),
            ("candidate_count", "8"),
            ("candidate_ok_count", "7"),
            ("eval_overlay_match_score", "0.61"),
            ("eval_node_locality_score", "0.59"),
            ("eval_recall_at_k_in_window", "0.42"),
            ("best_recall_at_k_in_window", "0.43"),
            ("materialized_recall_validation_status", "matched"),
            ("model_file_bytes", "123456"),
            ("codegen_package_bytes", "234567"),
            ("hash_latency_avg_ms", "0.001"),
            ("hash_latency_p95_ms", "0.002"),
            ("hash_latency_p99_ms", "0.003"),
            ("dominant_cost", "selector_search"),
            ("dominant_cost_wall_seconds", "12.5"),
        ],
    )
    manifest = {
        "summary_schema_version": 2,
        "profile": "sift",
        "preset": "baseline",
        "git": {"commit": "fixture", "dirty": False},
        "host": {"cpu_count": 8, "platform": "fixture"},
        "dataset": {
            "base": {"count": 1_000_000, "dim": 128},
            "query": {"count": 10_000, "dim": 128},
            "manifest": {
                "available": True,
                "required": True,
                "path": str(dataset_manifest),
                "sha256": dataset_sha,
                "dataset": "Fixture1M",
                "metric": "L2",
                "groundtruth_exact": True,
            },
        },
        "commands": {
            "selector": {
                "command": ["vortex_model_selector", "--fixture"],
                "command_line": "vortex_model_selector --fixture",
                "returncode": 0,
                "timed_out": False,
                "wall_seconds": 12.5,
                "peak_rss_kib": 1024,
                "log": str(logs / "selector.log"),
            }
        },
        "dominant_cost": {"name": "selector_search", "wall_seconds": 12.5},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    scale_dir = root / "scale_validation" / "fixture_scale"
    scale_dir.mkdir(parents=True)
    scale_summary = {
        "schema_version": 1,
        "run_name": "fixture_scale",
        "status": "ok",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "run_count": 1,
        "completed_count": 1,
        "stable_count": 1,
        "problem_count": 0,
        "profiles": ["sift"],
        "presets": ["baseline"],
        "runs": [
            {
                "profile": "sift",
                "preset": "baseline",
                "status": "ok",
                "quality_status": "ok",
                "run_dir": str(run_dir),
                "manifest_json": str(run_dir / "manifest.json"),
                "summary_csv": str(run_dir / "summary.csv"),
                "report_html": str(run_dir / "report.html"),
                "dataset_manifest_path": str(dataset_manifest),
                "dataset_manifest_sha256": dataset_sha,
                "command": "python3 tools/vortex_sift_benchmark_pack.py --fixture",
                "returncode": 0,
                "timed_out": False,
                "wall_seconds": 13.0,
                "selector_wall_seconds": 12.5,
                "candidate_count": 8,
                "candidate_ok_count": 7,
                "selected_target_skeleton": 30000,
                "selected_K": 512,
                "eval_overlay_match_score": 0.61,
                "eval_node_locality_score": 0.59,
                "eval_recall_at_k_in_window": 0.42,
                "best_recall_at_k_in_window": 0.43,
                "materialized_recall_validation_status": "matched",
                "model_file_bytes": 123456,
            }
        ],
    }
    (scale_dir / "summary.json").write_text(
        json.dumps(scale_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (scale_dir / "summary.csv").write_text("profile,preset,status\nsift,baseline,ok\n", encoding="utf-8")
    (scale_dir / "report.html").write_text("<h1>scale report</h1>\n", encoding="utf-8")
    (scale_dir / "README.md").write_text("# scale fixture\n", encoding="utf-8")
    return run_dir, scale_dir


def run_cmd(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def check_bundle_creation_and_verification() -> None:
    with tempfile.TemporaryDirectory(prefix="vortex-benchmark-artifact-") as tmp:
        tmp_root = Path(tmp)
        run_dir, scale_dir = write_fixture(tmp_root)
        output_root = tmp_root / "bundles"
        result = run_cmd(
            [
                sys.executable,
                str(SCRIPT),
                "--scale-validation-dir",
                str(scale_dir),
                "--run-dir",
                str(run_dir),
                "--output-root",
                str(output_root),
                "--bundle-name",
                "fixture_bundle",
                "--copy-reports",
                "--overwrite",
            ],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            raise AssertionError(result.stdout)
        bundle_dir = output_root / "fixture_bundle"
        manifest_path = bundle_dir / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise AssertionError("artifact manifest was not written")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1:
            raise AssertionError("unexpected artifact schema version")
        if not manifest["inputs"]["scale_validations"]:
            raise AssertionError("scale-validation input missing from bundle")
        if not manifest["inputs"]["benchmark_runs"]:
            raise AssertionError("benchmark run missing from bundle")
        if not any(r.get("kind") == "dataset_source_manifest" for r in manifest["file_records"]):
            raise AssertionError("dataset source manifest was not recorded")
        if not any(c.get("command_line") == "vortex_model_selector --fixture" for c in manifest["commands"]):
            raise AssertionError("run command ledger missing selector command")
        rows = list(csv.DictReader((bundle_dir / "metrics_summary.csv").open()))
        if rows[0]["eval_overlay_match_score"] != "0.61":
            raise AssertionError("final metric summary did not preserve overlay score")
        if not (bundle_dir / "commands.txt").read_text(encoding="utf-8").strip():
            raise AssertionError("command ledger is empty")
        if not (bundle_dir / "file_checksums.sha256").read_text(encoding="utf-8").strip():
            raise AssertionError("checksum ledger is empty")
        verify = run_cmd(
            [sys.executable, str(SCRIPT), "--verify-bundle", str(bundle_dir)],
            cwd=REPO_ROOT,
        )
        if verify.returncode != 0:
            raise AssertionError(verify.stdout)


def check_missing_dataset_manifest_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="vortex-benchmark-artifact-missing-") as tmp:
        tmp_root = Path(tmp)
        run_dir, _ = write_fixture(tmp_root)
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset"]["manifest"] = {"available": False}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        result = run_cmd(
            [
                sys.executable,
                str(SCRIPT),
                "--run-dir",
                str(run_dir),
                "--output-root",
                str(tmp_root / "bundles"),
                "--bundle-name",
                "should_fail",
            ],
            cwd=REPO_ROOT,
        )
        if result.returncode == 0:
            raise AssertionError("missing dataset manifest was accepted")
        if "dataset manifest" not in result.stdout:
            raise AssertionError(f"unexpected error output: {result.stdout}")


def check_unsafe_bundle_name_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="vortex-benchmark-artifact-name-") as tmp:
        tmp_root = Path(tmp)
        run_dir, _ = write_fixture(tmp_root)
        result = run_cmd(
            [
                sys.executable,
                str(SCRIPT),
                "--run-dir",
                str(run_dir),
                "--output-root",
                str(tmp_root / "bundles"),
                "--bundle-name",
                "bad:name",
            ],
            cwd=REPO_ROOT,
        )
        if result.returncode == 0:
            raise AssertionError("unsafe bundle name was accepted")
        if "path-safe" not in result.stdout:
            raise AssertionError(f"unexpected unsafe-name error output: {result.stdout}")


def main() -> int:
    check_bundle_creation_and_verification()
    check_missing_dataset_manifest_fails()
    check_unsafe_bundle_name_fails()
    print("vortex benchmark artifact bundle checks: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
