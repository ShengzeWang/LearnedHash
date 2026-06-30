"""Flatten benchmark-pack manifests into stable summary metrics."""

from __future__ import annotations

from typing import Any


RECALL_VALIDATION_TOLERANCE = 1.0e-6


def _numeric(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _recall_validation_tolerance(eval_out: dict[str, Any]) -> float:
    """Allow the smallest representable recall step when query/k counts are known."""
    query_count = eval_out.get("query_count")
    k = eval_out.get("k")
    if isinstance(query_count, int) and isinstance(k, int) and query_count > 0 and k > 0:
        return max(RECALL_VALIDATION_TOLERANCE, 1.0 / float(query_count * k))
    return RECALL_VALIDATION_TOLERANCE


def flatten_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["profile"] = manifest["profile"]
    out["preset"] = manifest["preset"]
    out["summary_schema_version"] = manifest.get("summary_schema_version", 2)
    out["legacy_metric_aliases"] = manifest.get("legacy_metric_aliases", False)
    out["git_commit"] = manifest["git"]["commit"]
    out["git_dirty"] = manifest["git"]["dirty"]
    out["base_count"] = manifest["dataset"]["base"]["count"]
    out["query_count"] = manifest["dataset"]["query"]["count"]
    out["dim"] = manifest["dataset"]["base"]["dim"]
    dataset_manifest = manifest["dataset"].get("manifest") or {}
    out["dataset_manifest_available"] = dataset_manifest.get("available")
    out["dataset_manifest_required"] = dataset_manifest.get("required")
    out["dataset_manifest_path"] = dataset_manifest.get("path")
    out["dataset_manifest_sha256"] = dataset_manifest.get("sha256")
    out["dataset_manifest_profile"] = dataset_manifest.get("profile")
    out["dataset_name"] = dataset_manifest.get("dataset")
    out["dataset_metric"] = dataset_manifest.get("metric")
    out["dataset_source_family"] = dataset_manifest.get("source_family")
    out["dataset_base_split"] = dataset_manifest.get("base_split")
    out["dataset_query_split"] = dataset_manifest.get("query_split")
    out["dataset_groundtruth_scope"] = dataset_manifest.get("groundtruth_scope")
    out["dataset_groundtruth_exact"] = dataset_manifest.get("groundtruth_exact")
    out["dataset_raw_validation_ok"] = dataset_manifest.get("raw_validation_ok")
    out["dataset_conversion_available"] = dataset_manifest.get("conversion_available")
    out["dataset_conversion_validated"] = dataset_manifest.get("conversion_validated")
    out["dataset_converted_at_utc"] = dataset_manifest.get("converted_at_utc")
    files = dataset_manifest.get("files") or {}
    for role in ("base", "query", "groundtruth"):
        file_info = files.get(role) or {}
        out[f"dataset_{role}_manifest_sha256"] = file_info.get("sha256")
        out[f"dataset_{role}_manifest_bytes"] = file_info.get("bytes")
        out[f"dataset_{role}_manifest_count"] = file_info.get("count")
        out[f"dataset_{role}_manifest_dim"] = file_info.get("dim")
        out[f"dataset_{role}_dtype_cast"] = file_info.get("dtype_cast")
    for name, result in manifest.get("commands", {}).items():
        out[f"{name}_wall_seconds"] = result.get("wall_seconds")
        out[f"{name}_peak_rss_kib"] = result.get("peak_rss_kib")
    out.update(manifest.get("selector_summary", {}))
    out["materialized_role_requested"] = manifest.get("materialized_role_requested")
    out["materialized_role"] = manifest.get("materialized_role")
    build = manifest.get("materialized_model") or manifest.get("best_model", {})
    for key in (
        "train_threads",
        "train_threads_fallback",
        "model_file_bytes",
        "codegen_model_bin_bytes",
        "codegen_centroid_index_bytes",
        "codegen_package_bytes",
    ):
        out[key] = build.get(key)
    cold = manifest.get("codegen_eval", {}).get("codegen_cold_build_load", {}).get("parsed_output", {})
    warm = manifest.get("codegen_eval", {}).get("codegen_warm_load", {}).get("parsed_output", {})
    eval_out = manifest.get("codegen_eval", {}).get("vortex_eval", {}).get("parsed_output", {})
    out["codegen_cold_load_time_ms"] = cold.get("load_time_ms")
    out["codegen_cold_library_status"] = cold.get("codegen_library_status")
    out["codegen_warm_load_time_ms"] = warm.get("load_time_ms")
    out["codegen_warm_library_status"] = warm.get("codegen_library_status")
    out["eval_load_time_ms"] = eval_out.get("load_time_ms")
    out["eval_codegen_library_status"] = eval_out.get("codegen_library_status")
    for field in (
        "read_base_time_ms",
        "read_query_time_ms",
        "hash_base_time_ms",
        "sort_base_time_ms",
        "rank_index_time_ms",
        "hash_index_cache",
        "hash_index_cache_load_time_ms",
        "hash_index_cache_write_time_ms",
        "truth_cache",
        "truth_cache_load_time_ms",
        "truth_cache_write_time_ms",
        "truth_groundtruth_load_time_ms",
        "truth_search_time_ms",
        "eval_query_time_ms",
    ):
        out[f"eval_{field}"] = eval_out.get(field)
    out["eval_recall_at_k_in_window"] = eval_out.get("recall@k_in_window")
    selector_recall = _numeric(out.get("materialized_recall_at_k_in_window"))
    eval_recall = _numeric(out["eval_recall_at_k_in_window"])
    recall_validation_tolerance = _recall_validation_tolerance(eval_out)
    if selector_recall is not None and eval_recall is not None:
        recall_delta = eval_recall - selector_recall
        out["materialized_selector_vs_eval_recall_delta"] = recall_delta
        out["materialized_selector_vs_eval_recall_abs_delta"] = abs(recall_delta)
        out["materialized_recall_validation_tolerance"] = recall_validation_tolerance
        out["materialized_recall_validation_status"] = (
            "matched" if abs(recall_delta) <= recall_validation_tolerance else "mismatch"
        )
    else:
        out["materialized_selector_vs_eval_recall_delta"] = None
        out["materialized_selector_vs_eval_recall_abs_delta"] = None
        out["materialized_recall_validation_tolerance"] = recall_validation_tolerance
        out["materialized_recall_validation_status"] = "unavailable"
    out["eval_mean_rank_distance_norm"] = eval_out.get("mean_rank_distance_norm")
    out["eval_node_count_values"] = eval_out.get("node_count_values")
    out["eval_overlay_match_score"] = eval_out.get("overlay_match_score")
    out["eval_overlay_match_p05"] = eval_out.get("overlay_match_p05")
    out["eval_overlay_match_p50"] = eval_out.get("overlay_match_p50")
    out["eval_overlay_match_p95"] = eval_out.get("overlay_match_p95")
    out["eval_node_locality_score"] = eval_out.get("node_locality_score")
    out["hash_latency_avg_ms"] = eval_out.get("latency_avg_ms")
    out["hash_latency_p95_ms"] = eval_out.get("latency_p95_ms")
    out["hash_latency_p99_ms"] = eval_out.get("latency_p99_ms")
    for prefix in (
        "nearest",
        "hash_with_centroid",
        "ann",
        "hash_ann",
        "ann_cached",
        "hash_ann_cached",
    ):
        out[f"{prefix}_latency_available"] = eval_out.get(f"{prefix}_latency_available")
        out[f"{prefix}_latency_avg_ms"] = eval_out.get(f"{prefix}_latency_avg_ms")
        out[f"{prefix}_latency_p95_ms"] = eval_out.get(f"{prefix}_latency_p95_ms")
        out[f"{prefix}_latency_p99_ms"] = eval_out.get(f"{prefix}_latency_p99_ms")
    out["dominant_cost"] = manifest.get("dominant_cost", {}).get("name")
    out["dominant_cost_wall_seconds"] = manifest.get("dominant_cost", {}).get("wall_seconds")
    return out
