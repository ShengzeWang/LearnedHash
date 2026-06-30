"""Manifest generation for materialized Vortex datasets."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .binary_io import maybe_sha256, sha256_file
from .paths import (
    dataset_output_dir,
    materialized_base_path,
    materialized_groundtruth_path,
    materialized_query_path,
    texmex_expected_size,
    texmex_file_path,
    texmex_output_dir,
)
from .specs import DatasetSpec, TexMexSpec


def manifest(
    spec: DatasetSpec,
    base: Path | None,
    query: Path | None,
    groundtruth: Path | None,
    *,
    include_file_sha256: bool = True,
    validation: dict[str, object] | None = None,
    vortex_conversion: dict[str, object] | None = None,
    materialized_at_utc: str | None = None,
) -> dict[str, object]:
    item = {
        "dataset": spec.display_name,
        "profile": spec.name,
        "role": spec.role,
        "source_family": spec.source_family,
        "base_url": spec.base_url,
        "query_url": spec.query_url,
        "groundtruth_url": spec.groundtruth_url,
        "slice_rule": spec.slice_rule,
        "slice_based": spec.slice_based,
        "base_split": spec.base_split,
        "query_split": spec.query_split,
        "byte_range": spec.byte_range,
        "base_dtype": spec.base_dtype,
        "query_dtype": spec.query_dtype,
        "dim": spec.dim,
        "metric": spec.metric,
        "base_count": spec.base_count,
        "source_count": spec.source_count,
        "query_count": spec.query_count,
        "groundtruth_topk": spec.groundtruth_topk,
        "groundtruth_scope": spec.groundtruth_scope,
        "groundtruth_exact": spec.groundtruth_exact,
        "expected_prefix_bytes": spec.prefix_bytes,
        "base_path": str(base) if base else None,
        "query_path": str(query) if query else None,
        "groundtruth_path": str(groundtruth) if groundtruth else None,
        "base_sha256": maybe_sha256(base, enabled=include_file_sha256),
        "query_sha256": maybe_sha256(query, enabled=include_file_sha256),
        "groundtruth_sha256": maybe_sha256(groundtruth, enabled=include_file_sha256),
        "conversion": f"{spec.base_dtype}_to_float32_fvecs_no_normalization",
        "validator": "header_size_query_gt_match_id_bounds_distance_order_exact_spot_check",
        "validation": validation,
    }
    if vortex_conversion is not None:
        item["vortex_conversion"] = vortex_conversion
    if materialized_at_utc is not None:
        item["materialized_at_utc"] = materialized_at_utc
    return item


def write_materialized_manifest(
    spec: DatasetSpec,
    output_root: Path,
    validation: dict[str, object],
    *,
    include_file_sha256: bool,
    vortex_conversion: dict[str, object] | None = None,
) -> Path:
    output_dir = dataset_output_dir(spec, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(
            manifest(
                spec,
                materialized_base_path(spec, output_root),
                materialized_query_path(spec, output_root),
                materialized_groundtruth_path(spec, output_root),
                include_file_sha256=include_file_sha256,
                validation=validation,
                vortex_conversion=vortex_conversion,
                materialized_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def texmex_manifest(
    spec: TexMexSpec,
    output_root: Path,
    archive_info: dict[str, object] | None,
    validation: dict[str, object] | None,
    *,
    include_file_sha256: bool,
) -> dict[str, object]:
    files = []
    for file_spec in spec.files:
        path = texmex_file_path(spec, output_root, file_spec)
        item = {
            "path": str(path),
            "archive_member": file_spec.archive_member,
            "kind": file_spec.kind,
            "count": file_spec.count,
            "dim": file_spec.dim,
            "expected_bytes": texmex_expected_size(file_spec),
        }
        if include_file_sha256 and path.is_file():
            item["sha256"] = sha256_file(path)
        files.append(item)
    return {
        "dataset": spec.display_name,
        "profile": spec.name,
        "role": spec.role,
        "metric": spec.metric,
        "slice_based": spec.slice_based,
        "base_split": spec.base_split,
        "query_split": spec.query_split,
        "groundtruth_scope": spec.groundtruth_scope,
        "groundtruth_exact": spec.groundtruth_exact,
        "source_url": spec.url,
        "source_md5": spec.md5,
        "archive": archive_info,
        "files": files,
        "validation": validation,
        "materialized_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def write_texmex_manifest(
    spec: TexMexSpec,
    output_root: Path,
    archive_info: dict[str, object] | None,
    validation: dict[str, object] | None,
    *,
    include_file_sha256: bool,
) -> Path:
    output_dir = texmex_output_dir(spec, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(
            texmex_manifest(
                spec,
                output_root,
                archive_info,
                validation,
                include_file_sha256=include_file_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
