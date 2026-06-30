"""Shared constants and filesystem helpers for benchmark-pack runs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

from path_safety import safe_component


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "vortex_v1_output" / "benchmarks"
DEFAULT_EVAL_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "cache"
DEFAULT_CODEGEN_BUILD_CACHE_ROOT = DEFAULT_EVAL_CACHE_ROOT / "codegen_build"
MATERIALIZATION_ROLES = ("best", "peak_recall", "knee", "fast", "small")
MANIFEST_REQUIRED_PROFILES = {"gist1m", "sift10m", "deep10m_l2", "spacev10m"}


PROFILE_PATHS = {
    "siftsmall": {
        "base": ROOT / "vector_datasets" / "siftsmall" / "siftsmall_base.fvecs",
        "query": ROOT / "vector_datasets" / "siftsmall" / "siftsmall_query.fvecs",
        "groundtruth": ROOT / "vector_datasets" / "siftsmall" / "siftsmall_groundtruth.ivecs",
        "manifest": ROOT / "vector_datasets" / "siftsmall" / "manifest.json",
        "hnsw_index": ROOT / "vortex_v1_output" / "siftsmall_benchmark_hnsw.index",
    },
    "sift": {
        "base": ROOT / "vector_datasets" / "sift" / "sift_base.fvecs",
        "query": ROOT / "vector_datasets" / "sift" / "sift_query.fvecs",
        "groundtruth": ROOT / "vector_datasets" / "sift" / "sift_groundtruth.ivecs",
        "manifest": ROOT / "vector_datasets" / "sift" / "manifest.json",
        "hnsw_index": ROOT / "vortex_v1_output" / "sift_benchmark_hnsw.index",
    },
    "gist1m": {
        "base": ROOT / "vector_datasets" / "gist1m" / "gist_base.fvecs",
        "query": ROOT / "vector_datasets" / "gist1m" / "gist_query.fvecs",
        "groundtruth": ROOT / "vector_datasets" / "gist1m" / "gist_groundtruth.ivecs",
        "manifest": ROOT / "vector_datasets" / "gist1m" / "manifest.json",
        "hnsw_index": ROOT / "vortex_v1_output" / "gist1m_benchmark_hnsw.index",
    },
    "sift10m": {
        "base": ROOT / "vector_datasets" / "sift10m" / "sift10m_base.fvecs",
        "query": ROOT / "vector_datasets" / "sift10m" / "sift10m_query.fvecs",
        "groundtruth": ROOT / "vector_datasets" / "sift10m" / "sift10m_groundtruth.ivecs",
        "manifest": ROOT / "vector_datasets" / "sift10m" / "manifest.json",
        "hnsw_index": ROOT / "vortex_v1_output" / "sift10m_benchmark_hnsw.index",
    },
    "deep10m_l2": {
        "base": ROOT / "vector_datasets" / "deep10m_l2" / "deep10m_l2_base.fvecs",
        "query": ROOT / "vector_datasets" / "deep10m_l2" / "deep10m_l2_query.fvecs",
        "groundtruth": ROOT / "vector_datasets" / "deep10m_l2" / "deep10m_l2_groundtruth.ivecs",
        "manifest": ROOT / "vector_datasets" / "deep10m_l2" / "manifest.json",
        "hnsw_index": ROOT / "vortex_v1_output" / "deep10m_l2_benchmark_hnsw.index",
    },
    "spacev10m": {
        "base": ROOT / "vector_datasets" / "spacev10m" / "spacev10m_base.fvecs",
        "query": ROOT / "vector_datasets" / "spacev10m" / "spacev10m_query.fvecs",
        "groundtruth": ROOT / "vector_datasets" / "spacev10m" / "spacev10m_groundtruth.ivecs",
        "manifest": ROOT / "vector_datasets" / "spacev10m" / "manifest.json",
        "hnsw_index": ROOT / "vortex_v1_output" / "spacev10m_benchmark_hnsw.index",
    },
}


def utc_now_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)

def safe_label(label: str) -> str:
    return safe_component(label, fallback="command")

def fmt_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    unit = 0
    while amount >= 1024.0 and unit + 1 < len(units):
        amount /= 1024.0
        unit += 1
    if unit == 0:
        return f"{int(amount)} {units[unit]}"
    return f"{amount:.2f} {units[unit]}"

def fvecs_info(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(4)
    if len(raw) != 4:
        raise RuntimeError(f"Invalid fvecs file: {path}")
    dim = struct.unpack("<i", raw)[0]
    if dim <= 0:
        raise RuntimeError(f"Invalid fvecs dimension {dim}: {path}")
    row_bytes = 4 + 4 * dim
    if size % row_bytes != 0:
        raise RuntimeError(f"fvecs file size is not row-aligned: {path}")
    return {
        "path": str(path),
        "count": size // row_bytes,
        "dim": dim,
        "bytes": size,
    }

def ivecs_info(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(4)
    if len(raw) != 4:
        raise RuntimeError(f"Invalid ivecs file: {path}")
    dim = struct.unpack("<i", raw)[0]
    if dim <= 0:
        raise RuntimeError(f"Invalid ivecs dimension {dim}: {path}")
    row_bytes = 4 + 4 * dim
    if size % row_bytes != 0:
        raise RuntimeError(f"ivecs file size is not row-aligned: {path}")
    return {
        "path": str(path),
        "count": size // row_bytes,
        "dim": dim,
        "bytes": size,
    }

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def _root_relative_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path

def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)

def _require_manifest_field(condition: bool, profile_name: str, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Invalid dataset manifest for profile {profile_name}: {message}")

def _file_entry_summary(entry: dict[str, Any], path: Path, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "count": info["count"],
        "dim": info["dim"],
        "bytes": info["bytes"],
        "sha256": entry.get("sha256"),
        "kind": entry.get("kind"),
        "expected_bytes": entry.get("expected_bytes", entry.get("bytes")),
        "dtype_cast": entry.get("dtype_cast"),
        "source_dtype": entry.get("source_dtype"),
        "source_format": entry.get("source_format"),
    }

def _validate_converted_manifest_files(
    profile_name: str,
    profile: dict[str, Path],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    conversion = manifest.get("vortex_conversion") or {}
    validation = conversion.get("validation") or {}
    _require_manifest_field(bool(validation.get("ok")), profile_name, "conversion validation is not ok")
    files: dict[str, Any] = {}
    roles = {
        "base": ("fvecs", fvecs_info),
        "query": ("fvecs", fvecs_info),
        "groundtruth": ("ivecs", ivecs_info),
    }
    for role, (expected_kind, info_fn) in roles.items():
        entry = conversion.get(role) or {}
        _require_manifest_field(entry.get("kind") == expected_kind, profile_name,
                                f"{role} conversion kind must be {expected_kind}")
        manifest_path = _root_relative_path(entry.get("path"))
        profile_path = profile[role]
        _require_manifest_field(_same_path(manifest_path, profile_path), profile_name,
                                f"{role} path does not match profile path")
        _require_manifest_field(profile_path.is_file(), profile_name,
                                f"{role} file is missing: {profile_path}")
        info = info_fn(profile_path)
        _require_manifest_field(entry.get("bytes") == info["bytes"], profile_name,
                                f"{role} byte size does not match manifest")
        _require_manifest_field(entry.get("count") == info["count"], profile_name,
                                f"{role} count does not match manifest")
        _require_manifest_field(entry.get("dim") == info["dim"], profile_name,
                                f"{role} dimension does not match manifest")
        _require_manifest_field(bool(entry.get("sha256")), profile_name,
                                f"{role} SHA-256 is missing from manifest")
        files[role] = _file_entry_summary(entry, profile_path, info)
    return {
        "conversion_available": True,
        "conversion_validated": True,
        "converted_at_utc": conversion.get("converted_at_utc"),
        "conversion_command": conversion.get("command"),
        "normalization": conversion.get("normalization"),
        "distance_convention": conversion.get("distance_convention"),
        "files": files,
    }

def _validate_texmex_manifest_files(
    profile_name: str,
    profile: dict[str, Path],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_files = manifest.get("files") or []
    files: dict[str, Any] = {}
    roles = {
        "base": ("fvecs", fvecs_info),
        "query": ("fvecs", fvecs_info),
        "groundtruth": ("ivecs", ivecs_info),
    }
    for role, (expected_kind, info_fn) in roles.items():
        profile_path = profile[role]
        matches = [
            entry
            for entry in manifest_files
            if entry.get("kind") == expected_kind
            and _same_path(_root_relative_path(entry.get("path")), profile_path)
        ]
        _require_manifest_field(len(matches) == 1, profile_name,
                                f"expected one {role} {expected_kind} entry")
        _require_manifest_field(profile_path.is_file(), profile_name,
                                f"{role} file is missing: {profile_path}")
        entry = matches[0]
        info = info_fn(profile_path)
        expected_bytes = entry.get("expected_bytes")
        _require_manifest_field(expected_bytes == info["bytes"], profile_name,
                                f"{role} byte size does not match manifest")
        _require_manifest_field(entry.get("count") == info["count"], profile_name,
                                f"{role} count does not match manifest")
        _require_manifest_field(entry.get("dim") == info["dim"], profile_name,
                                f"{role} dimension does not match manifest")
        _require_manifest_field(bool(entry.get("sha256")), profile_name,
                                f"{role} SHA-256 is missing from manifest")
        files[role] = _file_entry_summary(entry, profile_path, info)
    return {
        "conversion_available": False,
        "conversion_validated": None,
        "files": files,
    }

def profile_manifest_info(profile_name: str, profile: dict[str, Path]) -> dict[str, Any]:
    manifest_path = profile.get("manifest")
    required = profile_name in MANIFEST_REQUIRED_PROFILES
    if manifest_path is None or not manifest_path.is_file():
        if required:
            raise RuntimeError(
                f"Profile {profile_name} requires a validated local manifest. "
                f"Run tools/vortex_dataset_prepare.py materialize/convert for 10M profiles "
                f"or texmex-materialize for TexMex profiles before benchmarking."
            )
        return {
            "available": False,
            "required": False,
            "path": str(manifest_path) if manifest_path is not None else None,
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_manifest_field(manifest.get("profile") == profile_name, profile_name,
                            f"profile field is {manifest.get('profile')!r}")
    _require_manifest_field(manifest.get("metric") == "L2", profile_name,
                            "metric must be L2")
    _require_manifest_field(manifest.get("groundtruth_exact") is True, profile_name,
                            "groundtruth_exact must be true")
    _require_manifest_field(bool((manifest.get("validation") or {}).get("ok")),
                            profile_name, "raw/materialized validation is not ok")

    file_info = (
        _validate_converted_manifest_files(profile_name, profile, manifest)
        if manifest.get("vortex_conversion")
        else _validate_texmex_manifest_files(profile_name, profile, manifest)
    )
    files = file_info.pop("files")
    return {
        "available": True,
        "required": required,
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "profile": manifest.get("profile"),
        "dataset": manifest.get("dataset"),
        "role": manifest.get("role"),
        "metric": manifest.get("metric"),
        "source_family": manifest.get("source_family", "texmex"),
        "source_url": manifest.get("source_url"),
        "base_url": manifest.get("base_url"),
        "query_url": manifest.get("query_url"),
        "groundtruth_url": manifest.get("groundtruth_url"),
        "source_md5": manifest.get("source_md5"),
        "base_split": manifest.get("base_split"),
        "query_split": manifest.get("query_split"),
        "groundtruth_scope": manifest.get("groundtruth_scope"),
        "groundtruth_exact": manifest.get("groundtruth_exact"),
        "slice_based": manifest.get("slice_based"),
        "slice_rule": manifest.get("slice_rule"),
        "raw_validation_ok": bool((manifest.get("validation") or {}).get("ok")),
        "materialized_at_utc": manifest.get("materialized_at_utc"),
        "files": files,
        **file_info,
    }

def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total

def file_hash64(path: Path) -> str:
    """Stable FNV-1a fingerprint for cache filenames."""
    value = 1469598103934665603
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(64 * 1024)
            if not chunk:
                break
            for byte in chunk:
                value ^= byte
                value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"

def git_value(args: list[str], default: str = "") -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return default

def cmake_cache(build_dir: Path) -> dict[str, str]:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.exists():
        return {}
    out: dict[str, str] = {}
    for line in cache_path.read_text(errors="replace").splitlines():
        if line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if key in {
            "CMAKE_BUILD_TYPE",
            "CMAKE_CXX_COMPILER",
            "CMAKE_CXX_FLAGS",
            "LEARNEDHASH_FAISS_PROVIDER",
            "LEARNEDHASH_FAISS_PROVIDER_RESOLVED",
            "LEARNEDHASH_BUILD_TOOLS",
        }:
            out[key] = value
    return out
