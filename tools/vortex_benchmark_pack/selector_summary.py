"""Selector JSON summarization helpers for benchmark-pack reports."""

from __future__ import annotations

from typing import Any


def same_config(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    lhs_config = lhs.get("config") if isinstance(lhs, dict) else None
    rhs_config = rhs.get("config") if isinstance(rhs, dict) else None
    return isinstance(lhs_config, dict) and lhs_config == rhs_config

def base_rank_attribution_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = selection.get("eval_base_rank_cache_attribution") or []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]

def base_rank_attribution_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    misses = row.get("misses")
    hash_ms = row.get("hash_ms")
    build_ms = row.get("build_ms")
    return (
        float(hash_ms) if isinstance(hash_ms, (int, float)) else 0.0,
        float(misses) if isinstance(misses, (int, float)) else 0.0,
        float(build_ms) if isinstance(build_ms, (int, float)) else 0.0,
    )

def base_rank_attribution_config_label(row: dict[str, Any]) -> str:
    roles = row.get("roles")
    role_label = ",".join(str(role) for role in roles) if isinstance(roles, list) else ""
    role_suffix = f", roles={role_label}" if role_label else ""
    return (
        f"{row.get('phase')}:ts={row.get('target_skeleton')},"
        f"nodes={row.get('skeleton_node_count')},K={row.get('K')},"
        f"knn={row.get('centroid_knn')},cdf={row.get('cdf_models')},"
        f"branch={row.get('cdf_branch')},graph={row.get('graph_centroid_order')}"
        f"{role_suffix}"
    )

def base_rank_attribution_family_label(row: dict[str, Any]) -> str:
    return (
        f"{row.get('phase')}:ts={row.get('target_skeleton')},"
        f"nodes={row.get('skeleton_node_count')},K={row.get('K')}"
    )

def aggregate_base_rank_attribution(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        aggregate = grouped.setdefault(
            key,
            {field: row.get(field) for field in fields},
        )
        for field in ("hits", "misses", "builds", "evictions"):
            value = row.get(field)
            if isinstance(value, (int, float)):
                aggregate[field] = int(aggregate.get(field) or 0) + int(value)
        for field in (
            "wait_ms",
            "build_ms",
            "nearest_ms",
            "hash_ms",
            "sort_ms",
            "rank_index_ms",
            "overhead_ms",
        ):
            value = row.get(field)
            if isinstance(value, (int, float)):
                aggregate[field] = float(aggregate.get(field) or 0.0) + float(value)
        for field in ("added_memory_bytes", "retained_memory_bytes", "evicted_bytes"):
            value = row.get(field)
            if isinstance(value, (int, float)):
                aggregate[field] = int(aggregate.get(field) or 0) + int(value)
    return sorted(
        grouped.values(),
        key=base_rank_attribution_sort_key,
        reverse=True,
    )

def summarize_selector(selector_json: dict[str, Any],
                       materialized_role: str,
                       materialized_candidate: dict[str, Any],
                       legacy_metric_aliases: bool = False) -> dict[str, Any]:
    candidates = selector_json.get("candidates", [])
    phase_counts: dict[str, int] = {}
    timing_fields = (
        "train_time_ms",
        "train_base_cache_wait_ms",
        "train_order_cache_wait_ms",
        "train_base_build_ms",
        "train_order_build_ms",
        "train_gather_skeleton_ms",
        "train_cluster_assign_ms",
        "train_assign_centroids_ms",
        "train_centroid_order_ms",
        "train_range_alloc_ms",
        "train_cdf_fit_ms",
        "train_model_assembly_ms",
        "eval_total_ms",
        "eval_hash_base_ms",
        "eval_sort_base_ms",
        "eval_rank_index_ms",
        "eval_query_ms",
        "eval_latency_ms",
    )
    eval_timing_by_phase: dict[str, dict[str, float]] = {
        field: {} for field in timing_fields
    }
    ok_count = 0
    error_stage_counts: dict[str, int] = {}
    for item in candidates:
        if item.get("ok"):
            ok_count += 1
        phase = str(item.get("phase", "unknown"))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if not item.get("ok"):
            stage = str(item.get("error_stage") or "unknown")
            error_stage_counts[stage] = error_stage_counts.get(stage, 0) + 1
            continue
        for field in timing_fields:
            value = item.get(field)
            if isinstance(value, (int, float)):
                phase_totals = eval_timing_by_phase[field]
                phase_totals[phase] = phase_totals.get(phase, 0.0) + float(value)
    best = selector_json.get("best") or {}
    selection = selector_json.get("selection") or {}
    calibration_probes = selection.get("final_calibration_progressive_probes") or []
    last_calibration_probe = (
        calibration_probes[-1]
        if isinstance(calibration_probes, list) and calibration_probes
        else {}
    )
    probe_by_name = {
        str(item.get("name")): item
        for item in calibration_probes
        if isinstance(item, dict)
    } if isinstance(calibration_probes, list) else {}
    sampled_to_screen_probe = probe_by_name.get("sampled_to_final_calibration_screen", {})
    screen_to_final_probe = probe_by_name.get("screen_to_final_calibration", {})
    base_rank_rows = base_rank_attribution_rows(selection)
    top_base_rank_rows = sorted(
        base_rank_rows,
        key=base_rank_attribution_sort_key,
        reverse=True,
    )
    top_base_rank = top_base_rank_rows[0] if top_base_rank_rows else {}
    top_base_rank_phase_rows = aggregate_base_rank_attribution(
        base_rank_rows,
        ("phase",),
    )
    top_base_rank_phase = top_base_rank_phase_rows[0] if top_base_rank_phase_rows else {}
    top_base_rank_family_rows = aggregate_base_rank_attribution(
        base_rank_rows,
        ("phase", "target_skeleton", "skeleton_node_count", "K"),
    )
    top_base_rank_family = top_base_rank_family_rows[0] if top_base_rank_family_rows else {}
    summary: dict[str, Any] = {
        "candidate_count": len(candidates),
        "candidate_ok_count": ok_count,
        "candidate_error_count": len(candidates) - ok_count,
        "candidate_error_stage_counts": error_stage_counts,
        "selector_candidate_error_count": selection.get("candidate_error_count"),
        "selector_candidate_error_launch_guard_count": selection.get(
            "candidate_error_launch_guard_count"),
        "selector_candidate_error_training_base_count": selection.get(
            "candidate_error_training_base_count"),
        "selector_candidate_error_centroid_order_count": selection.get(
            "candidate_error_centroid_order_count"),
        "selector_candidate_error_cdf_fit_count": selection.get(
            "candidate_error_cdf_fit_count"),
        "selector_candidate_error_eval_count": selection.get(
            "candidate_error_eval_count"),
        "selector_candidate_error_unknown_count": selection.get(
            "candidate_error_unknown_count"),
        "phase_counts": phase_counts,
        "selector_elapsed_search_seconds": selection.get("elapsed_search_seconds"),
        "selector_use_full_dataset_for_assignments": selection.get(
            "use_full_dataset_for_assignments"),
        "selector_auto_assignment_sample_limit": selection.get(
            "auto_assignment_sample_limit"),
        "selector_assignment_sample_limit": selection.get("assignment_sample_limit"),
        "selector_assignment_sample_policy": selection.get("assignment_sample_policy"),
        "selector_graph_centroid_order": selection.get("graph_centroid_order"),
        "selector_graph_centroid_order_evidence": selection.get(
            "graph_centroid_order_evidence"),
        "selector_centroid_order_policy": selection.get("centroid_order_policy"),
        "selector_model_trains_started": selection.get("model_trains_started"),
        "selector_train_cache_hits": selection.get("train_cache_hits"),
        "selector_train_cache_misses": selection.get("train_cache_misses"),
        "selector_train_cache_wait_ms": selection.get("train_cache_wait_ms"),
        "selector_training_lane_wait_ms": selection.get("training_lane_wait_ms"),
        "selector_training_base_cache_hits": selection.get("training_base_cache_hits"),
        "selector_training_base_cache_misses": selection.get("training_base_cache_misses"),
        "selector_training_order_cache_hits": selection.get("training_order_cache_hits"),
        "selector_training_order_cache_misses": selection.get("training_order_cache_misses"),
        "selector_training_cdf_cache_hits": selection.get("training_cdf_cache_hits"),
        "selector_training_cdf_cache_misses": selection.get("training_cdf_cache_misses"),
        "selector_training_cdf_cache_failures": selection.get(
            "training_cdf_cache_failures"),
        "selector_training_base_cache_wait_ms": selection.get(
            "training_base_cache_wait_ms"),
        "selector_training_order_cache_wait_ms": selection.get(
            "training_order_cache_wait_ms"),
        "selector_training_cdf_cache_wait_ms": selection.get(
            "training_cdf_cache_wait_ms"),
        "selector_training_base_cache_build_ms": selection.get(
            "training_base_cache_build_ms"),
        "selector_training_order_cache_build_ms": selection.get(
            "training_order_cache_build_ms"),
        "selector_training_cdf_cache_build_ms": selection.get(
            "training_cdf_cache_build_ms"),
        "selector_training_base_cache_memory_bytes": selection.get(
            "training_base_cache_memory_bytes"),
        "selector_training_order_cache_memory_bytes": selection.get(
            "training_order_cache_memory_bytes"),
        "selector_training_cdf_cache_memory_bytes": selection.get(
            "training_cdf_cache_memory_bytes"),
        "selector_train_time_ms_sum": selection.get("train_time_ms_sum"),
        "selector_train_gather_skeleton_ms_sum": selection.get(
            "train_gather_skeleton_ms_sum"),
        "selector_train_cluster_assign_ms_sum": selection.get(
            "train_cluster_assign_ms_sum"),
        "selector_train_assign_centroids_ms_sum": selection.get(
            "train_assign_centroids_ms_sum"),
        "selector_train_centroid_order_ms_sum": selection.get(
            "train_centroid_order_ms_sum"),
        "selector_train_range_alloc_ms_sum": selection.get("train_range_alloc_ms_sum"),
        "selector_train_cdf_fit_ms_sum": selection.get("train_cdf_fit_ms_sum"),
        "selector_train_model_assembly_ms_sum": selection.get(
            "train_model_assembly_ms_sum"),
        "selector_scheduler_ready_cached_candidates": selection.get(
            "scheduler_ready_cached_candidates"),
        "selector_scheduler_in_flight_candidates": selection.get(
            "scheduler_in_flight_candidates"),
        "selector_scheduler_missing_candidates": selection.get(
            "scheduler_missing_candidates"),
        "selector_scheduler_single_admission_batches": selection.get(
            "scheduler_single_admission_batches"),
        "selector_scheduler_reuse_prioritized_candidates": selection.get(
            "scheduler_reuse_prioritized_candidates"),
        "selector_scheduler_fresh_base_candidates": selection.get(
            "scheduler_fresh_base_candidates"),
        "selector_eval_nearest_cache_hits": selection.get("eval_nearest_cache_hits"),
        "selector_eval_nearest_cache_misses": selection.get("eval_nearest_cache_misses"),
        "selector_eval_nearest_cache_inflight_bypasses": selection.get(
            "eval_nearest_cache_inflight_bypasses"),
        "selector_eval_nearest_cache_prewarm_requests": selection.get(
            "eval_nearest_cache_prewarm_requests"),
        "selector_eval_nearest_cache_prewarm_ready_models": selection.get(
            "eval_nearest_cache_prewarm_ready_models"),
        "selector_eval_nearest_cache_prewarm_skipped": selection.get(
            "eval_nearest_cache_prewarm_skipped"),
        "selector_eval_nearest_cache_prewarm_failures": selection.get(
            "eval_nearest_cache_prewarm_failures"),
        "selector_eval_nearest_cache_prewarm_threads": selection.get(
            "eval_nearest_cache_prewarm_threads"),
        "selector_eval_nearest_cache_prewarm_ms": selection.get(
            "eval_nearest_cache_prewarm_ms"),
        "selector_eval_nearest_cache_wait_ms": selection.get("eval_nearest_cache_wait_ms"),
        "selector_eval_nearest_cache_sampled_wait_ms": selection.get(
            "eval_nearest_cache_sampled_wait_ms"),
        "selector_eval_nearest_cache_calibration_wait_ms": selection.get(
            "eval_nearest_cache_calibration_wait_ms"),
        "selector_eval_nearest_cache_build_ms": selection.get("eval_nearest_cache_build_ms"),
        "selector_eval_nearest_cache_sampled_build_ms": selection.get(
            "eval_nearest_cache_sampled_build_ms"),
        "selector_eval_nearest_cache_calibration_build_ms": selection.get(
            "eval_nearest_cache_calibration_build_ms"),
        "selector_eval_nearest_cache_distance_terms": selection.get(
            "eval_nearest_cache_distance_terms"
        ),
        "selector_eval_nearest_cache_skipped_distance_terms": selection.get(
            "eval_nearest_cache_skipped_distance_terms"
        ),
        "selector_eval_nearest_cache_early_abandon_builds": selection.get(
            "eval_nearest_cache_early_abandon_builds"
        ),
        "selector_eval_nearest_cache_entries": selection.get("eval_nearest_cache_entries"),
        "selector_eval_nearest_cache_memory_bytes": selection.get(
            "eval_nearest_cache_memory_bytes"),
        "selector_eval_nearest_cache_sampled_budget_bytes": selection.get(
            "eval_nearest_cache_sampled_budget_bytes"),
        "selector_eval_nearest_cache_sampled_memory_bytes": selection.get(
            "eval_nearest_cache_sampled_memory_bytes"),
        "selector_eval_nearest_cache_calibration_memory_bytes": selection.get(
            "eval_nearest_cache_calibration_memory_bytes"),
        "selector_eval_nearest_cache_evictions": selection.get(
            "eval_nearest_cache_evictions"),
        "selector_eval_nearest_cache_evicted_bytes": selection.get(
            "eval_nearest_cache_evicted_bytes"),
        "selector_eval_base_rank_cache_hits": selection.get("eval_base_rank_cache_hits"),
        "selector_eval_base_rank_cache_misses": selection.get("eval_base_rank_cache_misses"),
        "selector_eval_base_rank_cache_wait_ms": selection.get("eval_base_rank_cache_wait_ms"),
        "selector_eval_base_rank_cache_build_ms": selection.get("eval_base_rank_cache_build_ms"),
        "selector_eval_base_rank_cache_nearest_ms": selection.get(
            "eval_base_rank_cache_nearest_ms"),
        "selector_eval_base_rank_cache_hash_ms": selection.get(
            "eval_base_rank_cache_hash_ms"),
        "selector_eval_base_rank_cache_sort_ms": selection.get(
            "eval_base_rank_cache_sort_ms"),
        "selector_eval_base_rank_cache_rank_index_ms": selection.get(
            "eval_base_rank_cache_rank_index_ms"),
        "selector_eval_base_rank_cache_overhead_ms": selection.get(
            "eval_base_rank_cache_overhead_ms"),
        "selector_eval_base_rank_cache_entries": selection.get("eval_base_rank_cache_entries"),
        "selector_eval_base_rank_cache_memory_bytes": selection.get(
            "eval_base_rank_cache_memory_bytes"),
        "selector_eval_base_rank_cache_sampled_budget_bytes": selection.get(
            "eval_base_rank_cache_sampled_budget_bytes"),
        "selector_eval_base_rank_cache_sampled_memory_bytes": selection.get(
            "eval_base_rank_cache_sampled_memory_bytes"),
        "selector_eval_base_rank_cache_calibration_memory_bytes": selection.get(
            "eval_base_rank_cache_calibration_memory_bytes"),
        "selector_eval_base_rank_cache_evictions": selection.get(
            "eval_base_rank_cache_evictions"),
        "selector_eval_base_rank_cache_evicted_bytes": selection.get(
            "eval_base_rank_cache_evicted_bytes"),
        "selector_eval_base_rank_cache_attribution_count": len(base_rank_rows),
        "selector_eval_base_rank_cache_top_hash_config": (
            base_rank_attribution_config_label(top_base_rank) if top_base_rank else None
        ),
        "selector_eval_base_rank_cache_top_hash_phase": top_base_rank_phase.get("phase"),
        "selector_eval_base_rank_cache_top_hash_phase_hash_ms": top_base_rank_phase.get(
            "hash_ms"),
        "selector_eval_base_rank_cache_top_hash_phase_misses": top_base_rank_phase.get(
            "misses"),
        "selector_eval_base_rank_cache_top_hash_family": (
            base_rank_attribution_family_label(top_base_rank_family)
            if top_base_rank_family else None
        ),
        "selector_eval_base_rank_cache_top_hash_family_hash_ms": (
            top_base_rank_family.get("hash_ms")
        ),
        "selector_eval_base_rank_cache_top_hash_family_misses": (
            top_base_rank_family.get("misses")
        ),
        "selector_eval_base_rank_cache_top_miss_family": (
            base_rank_attribution_config_label(top_base_rank) if top_base_rank else None
        ),
        "selector_eval_base_rank_cache_top_miss_phase": top_base_rank.get("phase"),
        "selector_eval_base_rank_cache_top_miss_target_skeleton": top_base_rank.get(
            "target_skeleton"),
        "selector_eval_base_rank_cache_top_miss_skeleton_nodes": top_base_rank.get(
            "skeleton_node_count"),
        "selector_eval_base_rank_cache_top_miss_K": top_base_rank.get("K"),
        "selector_eval_base_rank_cache_top_miss_centroid_knn": top_base_rank.get(
            "centroid_knn"),
        "selector_eval_base_rank_cache_top_miss_cdf_models": top_base_rank.get(
            "cdf_models"),
        "selector_eval_base_rank_cache_top_miss_cdf_branch": top_base_rank.get(
            "cdf_branch"),
        "selector_eval_base_rank_cache_top_miss_roles": top_base_rank.get("roles"),
        "selector_eval_base_rank_cache_top_miss_hits": top_base_rank.get("hits"),
        "selector_eval_base_rank_cache_top_miss_misses": top_base_rank.get("misses"),
        "selector_eval_base_rank_cache_top_miss_hash_ms": top_base_rank.get("hash_ms"),
        "selector_eval_base_rank_cache_top_miss_build_ms": top_base_rank.get("build_ms"),
        "selector_eval_base_rank_cache_top_miss_retained_bytes": top_base_rank.get(
            "retained_memory_bytes"),
        "selector_model_train_budget_hit": selection.get("model_train_budget_hit"),
        "selector_search_time_budget_hit": selection.get("search_time_budget_hit"),
        "selector_remaining_time_guard_hit": selection.get("remaining_time_guard_hit"),
        "selector_final_calibration_reserve_seconds": selection.get(
            "final_calibration_reserve_seconds"),
        "selector_final_calibration_reserve_hit": selection.get(
            "final_calibration_reserve_hit"),
        "selector_stopped_reason": selection.get("stopped_reason"),
        "selector_plateau_rounds": selection.get("plateau_rounds"),
        "selector_best_quality": selection.get("best_quality"),
        "selector_best_margin": selection.get("best_margin"),
        "selector_capacity_policy": selection.get("selector_capacity_policy"),
        "selector_contexts_requested": selection.get("target_skeleton_contexts_requested"),
        "selector_contexts_unique": selection.get("target_skeleton_contexts_unique"),
        "selector_contexts_deduplicated": selection.get("target_skeleton_contexts_deduplicated"),
        "selector_contexts_evaluated": selection.get("target_skeleton_contexts_evaluated"),
        "selector_contexts_selected": selection.get("target_skeleton_contexts_selected"),
        "selector_attribution_guided_compaction": selection.get(
            "attribution_guided_compaction"),
        "selector_strategy_frontier_candidates": selection.get(
            "strategy_frontier_candidates"),
        "selector_strategy_compacted_candidates": selection.get(
            "strategy_compacted_candidates"),
        "selector_post_exploration_evaluated": selection.get("post_exploration_evaluated"),
        "selector_final_calibration_seed_count": selection.get("final_calibration_seed_count"),
        "selector_effective_final_calibration_full_count": selection.get(
            "effective_final_calibration_full_count"),
        "selector_final_calibration_screening": selection.get("final_calibration_screening"),
        "selector_final_calibration_screen_candidate_count": selection.get(
            "final_calibration_screen_candidate_count"),
        "selector_effective_final_calibration_screen_candidate_count": selection.get(
            "effective_final_calibration_screen_candidate_count"),
        "selector_final_calibration_screen_query_limit": selection.get(
            "final_calibration_screen_query_limit"),
        "selector_effective_final_calibration_screen_query_limit": selection.get(
            "effective_final_calibration_screen_query_limit"),
        "selector_final_calibration_screen_evaluated": selection.get(
            "final_calibration_screen_evaluated"),
        "selector_final_calibration_screen_promoted": selection.get(
            "final_calibration_screen_promoted"),
        "selector_final_calibration_role_promoted": selection.get(
            "final_calibration_role_promoted"),
        "selector_final_calibration_recall_promoted": selection.get(
            "final_calibration_recall_promoted"),
        "selector_final_calibration_screen_status": selection.get(
            "final_calibration_screen_status"),
        "selector_final_calibration_evaluated": selection.get("final_calibration_evaluated"),
        "selector_final_calibration_reused": selection.get("final_calibration_reused"),
        "selector_final_calibration_progressive_probe_count": (
            len(calibration_probes) if isinstance(calibration_probes, list) else 0
        ),
        "selector_final_calibration_progressive_peak_match": (
            last_calibration_probe.get("peak_recall_match")
            if isinstance(last_calibration_probe, dict) else None
        ),
        "selector_final_calibration_progressive_all_roles_match": (
            last_calibration_probe.get("all_roles_match")
            if isinstance(last_calibration_probe, dict) else None
        ),
        "selector_final_calibration_progressive_matched_role_count": (
            last_calibration_probe.get("matched_role_count")
            if isinstance(last_calibration_probe, dict) else None
        ),
        "selector_final_calibration_progressive_probe_peak_recall": (
            last_calibration_probe.get("probe_peak_recall")
            if isinstance(last_calibration_probe, dict) else None
        ),
        "selector_final_calibration_progressive_final_peak_recall": (
            last_calibration_probe.get("final_peak_recall")
            if isinstance(last_calibration_probe, dict) else None
        ),
        "selector_final_calibration_sampled_to_screen_peak_match": (
            sampled_to_screen_probe.get("peak_recall_match")
            if isinstance(sampled_to_screen_probe, dict) else None
        ),
        "selector_final_calibration_sampled_to_screen_all_roles_match": (
            sampled_to_screen_probe.get("all_roles_match")
            if isinstance(sampled_to_screen_probe, dict) else None
        ),
        "selector_final_calibration_sampled_to_screen_matched_role_count": (
            sampled_to_screen_probe.get("matched_role_count")
            if isinstance(sampled_to_screen_probe, dict) else None
        ),
        "selector_final_calibration_screen_to_final_peak_match": (
            screen_to_final_probe.get("peak_recall_match")
            if isinstance(screen_to_final_probe, dict) else None
        ),
        "selector_final_calibration_screen_to_final_all_roles_match": (
            screen_to_final_probe.get("all_roles_match")
            if isinstance(screen_to_final_probe, dict) else None
        ),
        "selector_final_calibration_screen_to_final_matched_role_count": (
            screen_to_final_probe.get("matched_role_count")
            if isinstance(screen_to_final_probe, dict) else None
        ),
        "selector_final_calibration_screen_to_final_probe_peak_recall": (
            screen_to_final_probe.get("probe_peak_recall")
            if isinstance(screen_to_final_probe, dict) else None
        ),
        "selector_final_calibration_screen_to_final_final_peak_recall": (
            screen_to_final_probe.get("final_peak_recall")
            if isinstance(screen_to_final_probe, dict) else None
        ),
        "selector_peak_parallelism": selection.get("selector_peak_parallelism"),
        "selector_peak_threads_per_candidate": selection.get("selector_peak_threads_per_candidate"),
        "selector_memory_budget_limited": selection.get("selector_memory_budget_limited"),
        "selector_effective_memory_budget_bytes": selection.get("effective_selector_memory_budget_bytes"),
        "selector_eval_windows": selection.get("eval_windows"),
        "selector_eval_node_counts": selection.get("eval_node_counts"),
        "best_recall_at_k_in_window": best.get("recall_at_k_in_window"),
        "best_recall_auc_log_window": best.get("recall_auc_log_window"),
        "best_query_recall_p05": best.get("query_recall_p05"),
        "best_rank_distance_norm_p95": best.get("rank_distance_norm_p95"),
        "best_overlay_match_score": best.get("overlay_match_score"),
        "best_query_overlay_match_p05": best.get("query_overlay_match_p05"),
        "best_node_locality_score": best.get("node_locality_score"),
        "best_locality_quality_score": best.get("locality_quality_score"),
        "best_min_window_for_recall_target": best.get("min_window_for_recall_target"),
        "best_latency_avg_ms": best.get("latency_avg_ms"),
        "best_latency_p95_ms": best.get("latency_p95_ms"),
        "best_latency_p99_ms": best.get("latency_p99_ms"),
        "best_train_time_ms": best.get("train_time_ms"),
        "best_model_size_bytes": best.get("model_size_bytes"),
        "best_model_memory_bytes": best.get("model_memory_bytes"),
        "best_eval_hash_base_threads": best.get("eval_hash_base_threads"),
        "best_eval_rank_index_threads": best.get("eval_rank_index_threads"),
        "materialized_role": materialized_role,
        "materialized_matches_best": same_config(materialized_candidate, best),
        "materialized_recall_at_k_in_window": materialized_candidate.get(
            "recall_at_k_in_window"),
        "materialized_recall_auc_log_window": materialized_candidate.get(
            "recall_auc_log_window"),
        "materialized_query_recall_p05": materialized_candidate.get("query_recall_p05"),
        "materialized_overlay_match_score": materialized_candidate.get(
            "overlay_match_score"),
        "materialized_query_overlay_match_p05": materialized_candidate.get(
            "query_overlay_match_p05"),
        "materialized_node_locality_score": materialized_candidate.get(
            "node_locality_score"),
        "materialized_locality_quality_score": materialized_candidate.get(
            "locality_quality_score"),
        "materialized_latency_avg_ms": materialized_candidate.get("latency_avg_ms"),
        "materialized_model_size_bytes": materialized_candidate.get("model_size_bytes"),
        "materialized_model_memory_bytes": materialized_candidate.get("model_memory_bytes"),
    }
    for field in timing_fields:
        per_phase = eval_timing_by_phase[field]
        candidate_sum = sum(per_phase.values())
        summary[f"selector_{field}_by_phase"] = per_phase
        summary[f"selector_candidate_{field}_by_phase"] = per_phase
        summary[f"selector_candidate_{field}_sum"] = candidate_sum
        legacy_sum_key = f"selector_{field}_sum"
        if legacy_metric_aliases and summary.get(legacy_sum_key) is None:
            summary[legacy_sum_key] = candidate_sum
        summary[f"selector_final_calibration_{field}"] = per_phase.get("final_calibration", 0.0)
        summary[f"best_{field}"] = best.get(field)
    config = best.get("config") or {}
    for key, value in config.items():
        summary[f"best_config_{key}"] = value
    roles = selector_json.get("recommendation_roles") or {}
    if isinstance(roles, dict):
        for role_name in ("peak_recall", "knee", "fast", "small"):
            role = roles.get(role_name) or {}
            if isinstance(role, dict):
                summary[f"role_{role_name}_quality"] = role.get("locality_quality_score")
                summary[f"role_{role_name}_overlay_match"] = role.get("overlay_match_score")
                summary[f"role_{role_name}_node_locality"] = role.get("node_locality_score")
                summary[f"role_{role_name}_recall"] = role.get("recall_at_k_in_window")
                summary[f"role_{role_name}_auc"] = role.get("recall_auc_log_window")
                summary[f"role_{role_name}_latency_avg_ms"] = role.get("latency_avg_ms")
                summary[f"role_{role_name}_model_size_bytes"] = role.get("model_size_bytes")
                summary[f"materialized_matches_{role_name}"] = same_config(
                    materialized_candidate, role)
        peak = roles.get("peak_recall") or {}
        if isinstance(peak, dict):
            best_recall = best.get("recall_at_k_in_window")
            peak_recall = peak.get("recall_at_k_in_window")
            if isinstance(best_recall, (int, float)) and isinstance(peak_recall, (int, float)):
                summary["peak_recall_delta_vs_best"] = peak_recall - best_recall
            best_latency = best.get("latency_avg_ms")
            peak_latency = peak.get("latency_avg_ms")
            if isinstance(best_latency, (int, float)) and isinstance(peak_latency, (int, float)):
                summary["peak_latency_delta_ms_vs_best"] = peak_latency - best_latency
    best_target = config.get("target_skeleton")
    materialized_config = materialized_candidate.get("config") or {}
    if isinstance(materialized_config, dict):
        for key, value in materialized_config.items():
            summary[f"materialized_config_{key}"] = value
    for row in selection.get("context_stats", []) or []:
        if row.get("target_skeleton") == best_target:
            summary["best_context_effective_K_values"] = row.get("effective_K_values")
            summary["best_context_pruned_K_values"] = row.get("pruned_K_values")
            summary["best_context_min_vectors_per_centroid"] = row.get("min_vectors_per_centroid")
            summary["best_context_min_skeleton_nodes_per_centroid"] = row.get(
                "min_skeleton_nodes_per_centroid")
            break
    return summary
