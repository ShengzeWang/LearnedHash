"""Portable dataset profile and contract checks."""

from __future__ import annotations

from pathlib import Path

from .paths import converted_base_path, converted_groundtruth_path, converted_query_path
from .specs import DATASETS, TEXMEX_DATASETS


BENCHMARK_COMPAT_MANIFEST_OPTIONAL = frozenset({"siftsmall", "sift"})


def _texmex_file(spec, suffix: str):
    return next(
        file_spec for file_spec in spec.files if file_spec.output_name.endswith(suffix)
    )


def dataset_profile_rows() -> list[dict[str, object]]:
    """Return the unified dataset profile surface used by Vortex pipelines.

    These rows describe the local ``vector_datasets/<profile>/`` layout consumed
    by both LearnedHash model selection and the downstream benchmark pipeline.
    The public data-preparation tool remains standalone; benchmark-specific
    callers can consume this machine-readable profile list without importing any
    Vortex benchmark code.
    """

    rows: list[dict[str, object]] = []
    for spec in TEXMEX_DATASETS.values():
        base = _texmex_file(spec, "_base.fvecs")
        query = _texmex_file(spec, "_query.fvecs")
        groundtruth = _texmex_file(spec, "_groundtruth.ivecs")
        rows.append(
            {
                "profile": spec.name,
                "display_name": spec.display_name,
                "role": spec.role,
                "family": "texmex",
                "source_family": "texmex",
                "metric": spec.metric,
                "directory": spec.name,
                "base_file": base.output_name,
                "query_file": query.output_name,
                "groundtruth_file": groundtruth.output_name,
                "manifest_file": "manifest.json",
                "manifest_required": (
                    spec.name not in BENCHMARK_COMPAT_MANIFEST_OPTIONAL
                ),
                "base_count": base.count,
                "query_count": query.count,
                "dim": base.dim,
                "groundtruth_topk": groundtruth.dim,
                "slice_based": spec.slice_based,
                "base_split": spec.base_split,
                "query_split": spec.query_split,
                "groundtruth_scope": spec.groundtruth_scope,
                "groundtruth_exact": spec.groundtruth_exact,
            }
        )
    for spec in DATASETS.values():
        rows.append(
            {
                "profile": spec.name,
                "display_name": spec.display_name,
                "role": spec.role,
                "family": spec.source_family,
                "source_family": spec.source_family,
                "metric": spec.metric,
                "directory": spec.name,
                "base_file": converted_base_path(spec, output_root=Path()).name,
                "query_file": converted_query_path(spec, output_root=Path()).name,
                "groundtruth_file": converted_groundtruth_path(
                    spec, output_root=Path()
                ).name,
                "manifest_file": "manifest.json",
                "manifest_required": True,
                "base_count": spec.base_count,
                "query_count": spec.query_count,
                "dim": spec.dim,
                "groundtruth_topk": spec.groundtruth_topk,
                "slice_based": spec.slice_based,
                "base_split": spec.base_split,
                "query_split": spec.query_split,
                "groundtruth_scope": spec.groundtruth_scope,
                "groundtruth_exact": spec.groundtruth_exact,
            }
        )
    return rows


def workload_contract_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in DATASETS.values():
        rows.append(
            {
                "profile": spec.name,
                "dataset": spec.display_name,
                "family": spec.source_family,
                "metric": spec.metric,
                "slice_based": spec.slice_based,
                "slice_rule": spec.slice_rule,
                "base_split": spec.base_split,
                "query_split": spec.query_split,
                "groundtruth_url": spec.groundtruth_url,
                "groundtruth_scope": spec.groundtruth_scope,
                "groundtruth_exact": spec.groundtruth_exact,
                "groundtruth_topk": spec.groundtruth_topk,
            }
        )
    for spec in TEXMEX_DATASETS.values():
        rows.append(
            {
                "profile": spec.name,
                "dataset": spec.display_name,
                "family": "texmex",
                "metric": spec.metric,
                "slice_based": spec.slice_based,
                "slice_rule": "official_texmex_split",
                "base_split": spec.base_split,
                "query_split": spec.query_split,
                "groundtruth_url": spec.url,
                "groundtruth_scope": spec.groundtruth_scope,
                "groundtruth_exact": spec.groundtruth_exact,
                "groundtruth_topk": next(
                    file_spec.dim
                    for file_spec in spec.files
                    if file_spec.kind == "ivecs"
                ),
            }
        )
    return rows


def validate_workload_contracts() -> dict[str, object]:
    rows = workload_contract_rows()
    errors = []
    for row in rows:
        profile = row["profile"]
        if row["metric"] != "L2":
            errors.append(f"{profile}: metric must be L2, found {row['metric']}")
        if not row["base_split"]:
            errors.append(f"{profile}: base split contract is empty")
        if not row["query_split"]:
            errors.append(f"{profile}: query split contract is empty")
        if not row["groundtruth_exact"]:
            errors.append(f"{profile}: ground truth must be exact")
        if not row["groundtruth_scope"]:
            errors.append(f"{profile}: ground-truth scope is empty")
        if row["slice_based"] and "slice" not in str(row["groundtruth_scope"]):
            errors.append(f"{profile}: slice workload must use slice-scoped GT")
        if row["slice_based"] and "prefix" not in str(row["base_split"]):
            errors.append(f"{profile}: slice workload must use a canonical prefix base")
    return {"ok": not errors, "errors": errors, "workloads": rows}
