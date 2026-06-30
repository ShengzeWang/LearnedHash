"""Path and size helpers for materialized Vortex datasets."""

from __future__ import annotations

from pathlib import Path

from .specs import DatasetSpec, TexMexFileSpec, TexMexSpec, ValidationError, xbin_size


def dataset_output_dir(spec: DatasetSpec, output_root: Path) -> Path:
    return output_root / spec.name


def dataset_source_dir(spec: DatasetSpec, output_root: Path) -> Path:
    return output_root / "_sources" / spec.source_family


def dataset_base_suffix(spec: DatasetSpec) -> str:
    return {
        "uint8": "u8bin",
        "int8": "i8bin",
        "float32": "fbin",
    }[spec.base_dtype]


def materialized_base_path(spec: DatasetSpec, output_root: Path) -> Path:
    return (
        dataset_output_dir(spec, output_root) / f"base.10M.{dataset_base_suffix(spec)}"
    )


def materialized_query_path(spec: DatasetSpec, output_root: Path) -> Path:
    return dataset_output_dir(spec, output_root) / Path(spec.query_url).name


def materialized_groundtruth_path(spec: DatasetSpec, output_root: Path) -> Path:
    return dataset_output_dir(spec, output_root) / Path(spec.groundtruth_url).name


def source_base_path(spec: DatasetSpec, output_root: Path) -> Path:
    return (
        dataset_source_dir(spec, output_root) / f"base.10M.{dataset_base_suffix(spec)}"
    )


def source_query_path(spec: DatasetSpec, output_root: Path) -> Path:
    return dataset_source_dir(spec, output_root) / Path(spec.query_url).name


def source_groundtruth_path(spec: DatasetSpec, output_root: Path) -> Path:
    return dataset_source_dir(spec, output_root) / Path(spec.groundtruth_url).name


def expected_query_bytes(spec: DatasetSpec) -> int:
    return xbin_size(spec.query_count, spec.dim, spec.query_dtype)


def expected_groundtruth_bytes(spec: DatasetSpec) -> int:
    return 8 + spec.query_count * spec.groundtruth_topk * 8


def fvecs_size(count: int, dim: int) -> int:
    return count * (4 + dim * 4)


def ivecs_size(count: int, dim: int) -> int:
    return count * (4 + dim * 4)


def converted_base_path(spec: DatasetSpec, output_root: Path) -> Path:
    return dataset_output_dir(spec, output_root) / f"{spec.name}_base.fvecs"


def converted_query_path(spec: DatasetSpec, output_root: Path) -> Path:
    return dataset_output_dir(spec, output_root) / f"{spec.name}_query.fvecs"


def converted_groundtruth_path(spec: DatasetSpec, output_root: Path) -> Path:
    return dataset_output_dir(spec, output_root) / f"{spec.name}_groundtruth.ivecs"


def texmex_archive_path(spec: TexMexSpec, source_root: Path) -> Path:
    return source_root / Path(spec.url).name


def texmex_output_dir(spec: TexMexSpec, output_root: Path) -> Path:
    return output_root / spec.name


def texmex_file_path(
    spec: TexMexSpec, output_root: Path, file_spec: TexMexFileSpec
) -> Path:
    return texmex_output_dir(spec, output_root) / file_spec.output_name


def texmex_expected_size(file_spec: TexMexFileSpec) -> int:
    if file_spec.kind == "fvecs":
        value_bytes = 4
    elif file_spec.kind == "ivecs":
        value_bytes = 4
    else:
        raise ValidationError(f"unsupported TexMex vector kind: {file_spec.kind}")
    return file_spec.count * (4 + file_spec.dim * value_bytes)
