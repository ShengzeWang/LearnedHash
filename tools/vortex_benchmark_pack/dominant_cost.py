"""Dominant-cost attribution for benchmark-pack runs."""

from __future__ import annotations

from typing import Any

from .common import fmt_bytes


def metric_value(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def dominant_cost(metrics: dict[str, Any]) -> dict[str, Any]:
    candidates = {
        "selector_search": metrics.get("selector_wall_seconds"),
        "train_codegen_materialization": metrics.get("train_codegen_wall_seconds"),
        "generated_cold_build_load": metrics.get("codegen_cold_build_load_wall_seconds"),
        "eval_recall_hash_latency": metrics.get("vortex_eval_wall_seconds"),
    }
    numeric = {k: float(v) for k, v in candidates.items() if isinstance(v, (int, float))}
    if not numeric:
        return {"name": None, "wall_seconds": None, "recommendation": "No timing data available."}
    name, value = max(numeric.items(), key=lambda item: item[1])
    recommendations = {
        "selector_search": "Optimize selector scheduling/data reuse before touching hash kernels.",
        "train_codegen_materialization": "Measure training substeps and codegen emission before changing selector policy.",
        "generated_cold_build_load": "Optimize generated-library build/cache invalidation and load path.",
        "eval_recall_hash_latency": "Profile centroid-distance/hash inference hot paths after confirming recall is stable.",
    }
    recommendation = recommendations[name]
    largest_substage = None
    if name == "selector_search":
        nearest_cache_build_split_available = any(
            isinstance(metrics.get(key), (int, float))
            for key in (
                "selector_eval_nearest_cache_sampled_build_ms",
                "selector_eval_nearest_cache_calibration_build_ms",
            )
        )
        nearest_cache_wait_split_available = any(
            isinstance(metrics.get(key), (int, float))
            for key in (
                "selector_eval_nearest_cache_sampled_wait_ms",
                "selector_eval_nearest_cache_calibration_wait_ms",
            )
        )
        selector_substages = {
            "actual_training_total": metrics.get("selector_train_time_ms_sum"),
            "training_base_build": metrics.get("selector_training_base_cache_build_ms"),
            "training_order_build": metrics.get("selector_training_order_cache_build_ms"),
            "training_cdf_cache_build": metrics.get("selector_training_cdf_cache_build_ms"),
            "training_gather_skeleton": metrics.get("selector_train_gather_skeleton_ms_sum"),
            "training_cluster_assign": metrics.get("selector_train_cluster_assign_ms_sum"),
            "training_assign_centroids": metrics.get("selector_train_assign_centroids_ms_sum"),
            "training_centroid_order": metrics.get("selector_train_centroid_order_ms_sum"),
            "training_range_alloc": metrics.get("selector_train_range_alloc_ms_sum"),
            "training_cdf_fit": metrics.get("selector_train_cdf_fit_ms_sum"),
            "training_model_assembly": metrics.get("selector_train_model_assembly_ms_sum"),
            "base_hashing": metric_value(metrics,
                                         "selector_candidate_eval_hash_base_ms_sum",
                                         "selector_eval_hash_base_ms_sum"),
            "hash_sort": metric_value(metrics,
                                      "selector_candidate_eval_sort_base_ms_sum",
                                      "selector_eval_sort_base_ms_sum"),
            "rank_index": metric_value(metrics,
                                       "selector_candidate_eval_rank_index_ms_sum",
                                       "selector_eval_rank_index_ms_sum"),
            "query_scoring": metric_value(metrics,
                                          "selector_candidate_eval_query_ms_sum",
                                          "selector_eval_query_ms_sum"),
            "latency_sampling": metric_value(metrics,
                                             "selector_candidate_eval_latency_ms_sum",
                                             "selector_eval_latency_ms_sum"),
            "serialized_training_lane_wait": metrics.get("selector_training_lane_wait_ms"),
            "nearest_centroid_cache_prewarm": metrics.get(
                "selector_eval_nearest_cache_prewarm_ms"),
            "nearest_centroid_cache_build": (
                None if nearest_cache_build_split_available
                else metrics.get("selector_eval_nearest_cache_build_ms")),
            "nearest_centroid_cache_sampled_build": metrics.get(
                "selector_eval_nearest_cache_sampled_build_ms"),
            "nearest_centroid_cache_calibration_build": metrics.get(
                "selector_eval_nearest_cache_calibration_build_ms"),
            "nearest_centroid_cache_wait": (
                None if nearest_cache_wait_split_available
                else metrics.get("selector_eval_nearest_cache_wait_ms")),
            "nearest_centroid_cache_sampled_wait": metrics.get(
                "selector_eval_nearest_cache_sampled_wait_ms"),
            "nearest_centroid_cache_calibration_wait": metrics.get(
                "selector_eval_nearest_cache_calibration_wait_ms"),
            "base_rank_cache_hash": metrics.get("selector_eval_base_rank_cache_hash_ms"),
            "base_rank_cache_sort": metrics.get("selector_eval_base_rank_cache_sort_ms"),
            "base_rank_cache_rank_index": metrics.get(
                "selector_eval_base_rank_cache_rank_index_ms"),
            "base_rank_cache_overhead": metrics.get("selector_eval_base_rank_cache_overhead_ms"),
            "base_rank_cache_wait": metrics.get("selector_eval_base_rank_cache_wait_ms"),
        }
        numeric_substages = {
            key: float(subvalue)
            for key, subvalue in selector_substages.items()
            if isinstance(subvalue, (int, float))
        }
        if numeric_substages:
            substage, substage_ms = max(numeric_substages.items(), key=lambda item: item[1])
            largest_substage = substage
            recommendation = (
                f"{recommendation} Largest selector substage: "
                f"{substage} ({substage_ms / 1000.0:.2f}s aggregate)."
            )
        lane_wait_ms = metrics.get("selector_training_lane_wait_ms")
        if (isinstance(lane_wait_ms, (int, float)) and lane_wait_ms > 0 and
                largest_substage != "serialized_training_lane_wait"):
            recommendation = (
                f"{recommendation} Aggregate serialized training-lane wait: "
                f"{lane_wait_ms / 1000.0:.2f}s across blocked workers."
            )
        base_rank_cache_bytes = metrics.get("selector_eval_base_rank_cache_memory_bytes")
        base_rank_cache_entries = metrics.get("selector_eval_base_rank_cache_entries")
        base_rank_cache_evictions = metrics.get("selector_eval_base_rank_cache_evictions")
        base_rank_cache_evicted_bytes = metrics.get(
            "selector_eval_base_rank_cache_evicted_bytes")
        if isinstance(base_rank_cache_bytes, (int, float)) and base_rank_cache_bytes > 0:
            recommendation = (
                f"{recommendation} Exact base-rank cache retained "
                f"{fmt_bytes(base_rank_cache_bytes)}"
                f" across {base_rank_cache_entries or 0} entries."
            )
        if (isinstance(base_rank_cache_evictions, (int, float)) and
                base_rank_cache_evictions > 0):
            recommendation = (
                f"{recommendation} Evicted {int(base_rank_cache_evictions)} "
                f"base-rank entries"
                f" ({fmt_bytes(base_rank_cache_evicted_bytes or 0)})."
            )
        top_base_rank_family = metrics.get("selector_eval_base_rank_cache_top_hash_family")
        top_base_rank_misses = metrics.get("selector_eval_base_rank_cache_top_hash_family_misses")
        top_base_rank_hash_ms = metrics.get("selector_eval_base_rank_cache_top_hash_family_hash_ms")
        if top_base_rank_family:
            recommendation = (
                f"{recommendation} Top aggregate base-rank hash family: "
                f"{top_base_rank_family} "
                f"({int(top_base_rank_misses or 0)} misses, "
                f"{float(top_base_rank_hash_ms or 0.0) / 1000.0:.2f}s hash)."
            )
        compacted = metrics.get("selector_strategy_compacted_candidates")
        frontier = metrics.get("selector_strategy_frontier_candidates")
        if isinstance(compacted, (int, float)) and compacted > 0:
            recommendation = (
                f"{recommendation} Attribution-guided compaction pruned "
                f"{int(compacted)} of {int(frontier or 0)} beam/frontier candidates."
            )
        nearest_cache_bytes = metrics.get("selector_eval_nearest_cache_memory_bytes")
        nearest_cache_entries = metrics.get("selector_eval_nearest_cache_entries")
        nearest_cache_evictions = metrics.get("selector_eval_nearest_cache_evictions")
        nearest_cache_bypasses = metrics.get(
            "selector_eval_nearest_cache_inflight_bypasses")
        nearest_cache_sampled_wait_ms = metrics.get(
            "selector_eval_nearest_cache_sampled_wait_ms")
        nearest_cache_calibration_wait_ms = metrics.get(
            "selector_eval_nearest_cache_calibration_wait_ms")
        if isinstance(nearest_cache_bytes, (int, float)) and nearest_cache_bytes > 0:
            recommendation = (
                f"{recommendation} Exact nearest cache retained "
                f"{fmt_bytes(nearest_cache_bytes)}"
                f" across {nearest_cache_entries or 0} entries."
            )
        if (isinstance(nearest_cache_evictions, (int, float)) and
                nearest_cache_evictions > 0):
            recommendation = (
                f"{recommendation} Evicted {int(nearest_cache_evictions)} "
                "nearest-cache entries."
            )
        if (isinstance(nearest_cache_bypasses, (int, float)) and
                nearest_cache_bypasses > 0):
            recommendation = (
                f"{recommendation} Bypassed {int(nearest_cache_bypasses)} "
                "in-flight sampled nearest-cache waits."
            )
        nearest_cache_prewarm_requests = metrics.get(
            "selector_eval_nearest_cache_prewarm_requests")
        nearest_cache_prewarm_ms = metrics.get("selector_eval_nearest_cache_prewarm_ms")
        nearest_cache_prewarm_threads = metrics.get(
            "selector_eval_nearest_cache_prewarm_threads")
        if (isinstance(nearest_cache_prewarm_requests, (int, float)) and
                nearest_cache_prewarm_requests > 0):
            recommendation = (
                f"{recommendation} Prewarmed {int(nearest_cache_prewarm_requests)} "
                f"calibration nearest-cache requests in "
                f"{float(nearest_cache_prewarm_ms or 0.0) / 1000.0:.2f}s"
                f" using up to {int(nearest_cache_prewarm_threads or 0)} threads."
            )
        if (isinstance(nearest_cache_calibration_wait_ms, (int, float)) and
                isinstance(nearest_cache_sampled_wait_ms, (int, float)) and
                nearest_cache_calibration_wait_ms > nearest_cache_sampled_wait_ms):
            recommendation = (
                f"{recommendation} Nearest-cache wait is calibration-dominated "
                f"({nearest_cache_calibration_wait_ms / 1000.0:.2f}s calibration, "
                f"{nearest_cache_sampled_wait_ms / 1000.0:.2f}s sampled aggregate)."
            )
        cdf_hits = metrics.get("selector_training_cdf_cache_hits")
        cdf_misses = metrics.get("selector_training_cdf_cache_misses")
        cdf_failures = metrics.get("selector_training_cdf_cache_failures")
        if any(isinstance(value, (int, float)) and value > 0
               for value in (cdf_hits, cdf_misses, cdf_failures)):
            recommendation = (
                f"{recommendation} Training CDF cache: {int(cdf_hits or 0)} hits, "
                f"{int(cdf_misses or 0)} misses, {int(cdf_failures or 0)} failures."
            )
        cdf_error_count = metrics.get("selector_candidate_error_cdf_fit_count")
        if isinstance(cdf_error_count, (int, float)) and cdf_error_count > 0:
            recommendation = (
                f"{recommendation} Candidate CDF-fit errors: {int(cdf_error_count)}."
            )
    return {
        "name": name,
        "wall_seconds": value,
        "recommendation": recommendation,
        "largest_selector_substage": largest_substage,
    }
