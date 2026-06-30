"""High-level dataset materialization and conversion workflows."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from .binary_io import sha256_file
from .conversion import convert_groundtruth_to_ivecs, convert_xbin_to_fvecs
from .download import (
    extract_texmex_archive,
    materialize_prefix_base,
    materialize_query_and_groundtruth,
    ensure_texmex_archive,
)
from .manifest import write_materialized_manifest, write_texmex_manifest
from .paths import (
    converted_base_path,
    converted_groundtruth_path,
    converted_query_path,
    dataset_output_dir,
    materialized_base_path,
    materialized_groundtruth_path,
    materialized_query_path,
)
from .specs import DatasetSpec, TexMexSpec
from .validation import (
    validate_converted_dataset,
    validate_materialized_dataset,
    validate_texmex_dataset,
)


def convert_materialized_dataset(
    spec: DatasetSpec,
    *,
    output_root: Path,
    force_convert: bool,
    scan_converted: bool,
    include_file_sha256: bool,
    chunk_rows: int,
    command_line: str | None = None,
) -> dict[str, object]:
    raw_validation = validate_materialized_dataset(
        spec,
        output_root,
        scan_groundtruth=True,
    )
    base_info = convert_xbin_to_fvecs(
        materialized_base_path(spec, output_root),
        converted_base_path(spec, output_root),
        count=spec.base_count,
        dim=spec.dim,
        dtype=spec.base_dtype,
        label="base",
        force=force_convert,
        chunk_rows=chunk_rows,
    )
    query_info = convert_xbin_to_fvecs(
        materialized_query_path(spec, output_root),
        converted_query_path(spec, output_root),
        count=spec.query_count,
        dim=spec.dim,
        dtype=spec.query_dtype,
        label="query",
        force=force_convert,
        chunk_rows=chunk_rows,
    )
    gt_info = convert_groundtruth_to_ivecs(
        materialized_groundtruth_path(spec, output_root),
        converted_groundtruth_path(spec, output_root),
        query_count=spec.query_count,
        topk=spec.groundtruth_topk,
        base_count=spec.base_count,
        force=force_convert,
        chunk_rows=chunk_rows,
    )
    converted_validation = validate_converted_dataset(
        spec,
        output_root,
        scan_headers=scan_converted,
    )
    converted_at = dt.datetime.now(dt.timezone.utc).isoformat()
    conversion = {
        "converted_at_utc": converted_at,
        "command": command_line,
        "source_contract": "validated_canonical_prefix_base_query_gt_triple",
        "metric": spec.metric,
        "distance_convention": "squared_L2_ranking_equivalent",
        "normalization": "none",
        "base": base_info,
        "query": query_info,
        "groundtruth": gt_info,
        "validation": converted_validation,
    }
    if include_file_sha256:
        conversion["base"]["sha256"] = sha256_file(
            converted_base_path(spec, output_root)
        )
        conversion["query"]["sha256"] = sha256_file(
            converted_query_path(spec, output_root)
        )
        conversion["groundtruth"]["sha256"] = sha256_file(
            converted_groundtruth_path(spec, output_root)
        )
    manifest_path = write_materialized_manifest(
        spec,
        output_root,
        raw_validation,
        include_file_sha256=include_file_sha256,
        vortex_conversion=conversion,
    )
    return {
        "dataset": spec.name,
        "raw_validation": raw_validation,
        "conversion": conversion,
        "converted_validation": converted_validation,
        "manifest": str(manifest_path),
        "ok": True,
    }


def materialize_dataset(
    spec: DatasetSpec,
    *,
    output_root: Path,
    force_download: bool,
    scan_groundtruth: bool,
    include_file_sha256: bool,
    spot_check_queries: int = 0,
    spot_check_seed: int = 20260101,
    spot_check_base_chunk: int = 65_536,
    spot_check_tolerance: float = 1e-4,
) -> dict[str, object]:
    dataset_output_dir(spec, output_root).mkdir(parents=True, exist_ok=True)
    base_info = materialize_prefix_base(
        spec, output_root, force_download=force_download
    )
    downloaded = materialize_query_and_groundtruth(
        spec, output_root, force_download=force_download
    )
    validation = validate_materialized_dataset(
        spec,
        output_root,
        scan_groundtruth=scan_groundtruth,
        spot_check_queries=spot_check_queries,
        spot_check_seed=spot_check_seed,
        spot_check_base_chunk=spot_check_base_chunk,
        spot_check_tolerance=spot_check_tolerance,
    )
    manifest_path = write_materialized_manifest(
        spec,
        output_root,
        validation,
        include_file_sha256=include_file_sha256,
    )
    return {
        "dataset": spec.name,
        "base": base_info,
        "query": downloaded["query"],
        "groundtruth": downloaded["groundtruth"],
        "source_query": downloaded["source_query"],
        "source_groundtruth": downloaded["source_groundtruth"],
        "validation": validation,
        "manifest": str(manifest_path),
        "ok": True,
    }


def materialize_texmex(
    spec: TexMexSpec,
    *,
    source_root: Path,
    output_root: Path,
    force_download: bool,
    force_extract: bool,
    scan_headers: bool,
    include_file_sha256: bool,
) -> dict[str, object]:
    archive_info = ensure_texmex_archive(
        spec, source_root, force_download=force_download
    )
    extracted = extract_texmex_archive(
        spec,
        Path(str(archive_info["path"])),
        output_root,
        force_extract=force_extract,
    )
    validation = validate_texmex_dataset(spec, output_root, scan_headers=scan_headers)
    manifest_path = write_texmex_manifest(
        spec,
        output_root,
        archive_info,
        validation,
        include_file_sha256=include_file_sha256,
    )
    return {
        "dataset": spec.name,
        "archive": archive_info,
        "extracted": extracted,
        "validation": validation,
        "manifest": str(manifest_path),
        "ok": True,
    }
