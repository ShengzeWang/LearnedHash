#include "selector_report_internal.h"

#include <cstdint>
#include <set>
#include <utility>

namespace vortex::selector_internal {

rm_model::json::Value selector_selection_to_json(const SelectorResult& result,
                                                 const SelectorOptions& report_options,
                                                 const SelectorOptions& raw_options,
                                                 const SelectorJsonOptions& json_options) {
  const SelectorOptions& options = raw_options;
  const bool include_flat_selection_compatibility_fields =
      json_options.include_flat_selection_compatibility_fields;
  rm_model::json::Value::Object selection_obj;
  selection_obj.emplace_back("schema_version",
                             rm_model::json::Value(
                                 include_flat_selection_compatibility_fields
                                     ? uint64_t{1}
                                     : uint64_t{2}));
  selection_obj.emplace_back("flat_compatibility_fields",
                             rm_model::json::Value(
                                 include_flat_selection_compatibility_fields));
  selection_obj.emplace_back("phase1_keep", rm_model::json::Value(static_cast<uint64_t>(report_options.phase1_keep)));
  selection_obj.emplace_back("max_phase2_candidates",
                             rm_model::json::Value(static_cast<uint64_t>(report_options.max_phase2_candidates)));
  selection_obj.emplace_back("attribution_guided_compaction",
                             rm_model::json::Value(report_options.attribution_guided_compaction));
  selection_obj.emplace_back("use_full_dataset_for_assignments",
                             rm_model::json::Value(report_options.use_full_dataset_for_assignments));
  selection_obj.emplace_back("auto_assignment_sample_limit",
                             rm_model::json::Value(report_options.auto_assignment_sample_limit));
  selection_obj.emplace_back("assignment_sample_limit",
                             rm_model::json::Value(report_options.assignment_sample_limit));
  std::string assignment_policy = "skeleton_only";
  if (report_options.use_full_dataset_for_assignments) {
    assignment_policy = report_options.assignment_sample_limit > 0 ? "bounded_sample" : "exact";
  }
  selection_obj.emplace_back(
      "assignment_sample_policy",
      rm_model::json::Value(assignment_policy));
  selection_obj.emplace_back("graph_centroid_order",
                             rm_model::json::Value(report_options.enable_graph_centroid_order));
  selection_obj.emplace_back("graph_centroid_order_evidence",
                             graph_centroid_order_evidence_to_json(result));
  selection_obj.emplace_back("recommend_count",
                             rm_model::json::Value(static_cast<uint64_t>(options.recommend_count)));
  selection_obj.emplace_back("selector_parallelism",
                             rm_model::json::Value(static_cast<uint64_t>(options.selector_parallelism)));
  selection_obj.emplace_back("selector_profile",
                             rm_model::json::Value(options.selector_profile));
  selection_obj.emplace_back("effective_selector_profile",
                             rm_model::json::Value(result.effective_selector_profile));
  selection_obj.emplace_back("objective_score_scope",
                             rm_model::json::Value("per-phase"));
  selection_obj.emplace_back(
      "objective_score_compare_rule",
      rm_model::json::Value(
          "Compare objective_score only among candidates with the same phase; "
          "phase1 uses coarse evaluation, phase2/post_exploration use sampled "
          "full evaluation, and final_calibration uses full-base calibration."));
  selection_obj.emplace_back("diagnostics_schema_version",
                             rm_model::json::Value(static_cast<uint64_t>(1)));
  selection_obj.emplace_back("diagnostics",
                             selector_diagnostics_to_json(result, options));
  selection_obj.emplace_back("eval_node_counts",
                             uint32_array_to_json(result.effective_eval_node_counts));
  if (result.dataset_count > 0) {
    selection_obj.emplace_back("dataset_count", rm_model::json::Value(result.dataset_count));
  }
  if (include_flat_selection_compatibility_fields) {
    selection_obj.emplace_back("target_skeleton_contexts_total",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.target_skeleton_contexts_total)));
    selection_obj.emplace_back("target_skeleton_contexts_requested",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.target_skeleton_contexts_requested)));
    selection_obj.emplace_back("target_skeleton_contexts_unique",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.target_skeleton_contexts_unique)));
    selection_obj.emplace_back("target_skeleton_contexts_deduplicated",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.target_skeleton_contexts_deduplicated)));
    selection_obj.emplace_back("target_skeleton_contexts_selected",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.target_skeleton_contexts_selected)));
    selection_obj.emplace_back("target_skeleton_contexts_evaluated",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.target_skeleton_contexts_evaluated)));
    selection_obj.emplace_back("max_target_skeleton_contexts",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(options.max_target_skeleton_contexts)));
    selection_obj.emplace_back("max_model_trains",
                               rm_model::json::Value(static_cast<uint64_t>(result.max_model_trains)));
    selection_obj.emplace_back("model_trains_started",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.model_trains_started)));
    selection_obj.emplace_back("train_cache_hits",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.train_cache_hits)));
    selection_obj.emplace_back("train_cache_misses",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.train_cache_misses)));
    selection_obj.emplace_back("train_cache_wait_ms",
                               rm_model::json::Value(result.train_cache_wait_ms));
    selection_obj.emplace_back("training_lane_wait_ms",
                               rm_model::json::Value(result.training_lane_wait_ms));
    selection_obj.emplace_back("training_base_cache_hits",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.training_base_cache_hits)));
    selection_obj.emplace_back("training_base_cache_misses",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.training_base_cache_misses)));
    selection_obj.emplace_back("training_order_cache_hits",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.training_order_cache_hits)));
    selection_obj.emplace_back("training_order_cache_misses",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.training_order_cache_misses)));
    selection_obj.emplace_back("training_cdf_cache_hits",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.training_cdf_cache_hits)));
    selection_obj.emplace_back("training_cdf_cache_misses",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.training_cdf_cache_misses)));
    selection_obj.emplace_back("training_cdf_cache_failures",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.training_cdf_cache_failures)));
    selection_obj.emplace_back("training_base_cache_wait_ms",
                               rm_model::json::Value(result.training_base_cache_wait_ms));
    selection_obj.emplace_back("training_order_cache_wait_ms",
                               rm_model::json::Value(result.training_order_cache_wait_ms));
    selection_obj.emplace_back("training_cdf_cache_wait_ms",
                               rm_model::json::Value(result.training_cdf_cache_wait_ms));
    selection_obj.emplace_back("training_base_cache_build_ms",
                               rm_model::json::Value(result.training_base_cache_build_ms));
    selection_obj.emplace_back("training_order_cache_build_ms",
                               rm_model::json::Value(result.training_order_cache_build_ms));
    selection_obj.emplace_back("training_cdf_cache_build_ms",
                               rm_model::json::Value(result.training_cdf_cache_build_ms));
    selection_obj.emplace_back("training_base_cache_memory_bytes",
                               rm_model::json::Value(
                                   result.training_base_cache_memory_bytes));
    selection_obj.emplace_back("training_order_cache_memory_bytes",
                               rm_model::json::Value(
                                   result.training_order_cache_memory_bytes));
    selection_obj.emplace_back("training_cdf_cache_memory_bytes",
                               rm_model::json::Value(
                                   result.training_cdf_cache_memory_bytes));
    selection_obj.emplace_back("candidate_error_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.candidate_error_count)));
    selection_obj.emplace_back("candidate_error_launch_guard_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.candidate_error_launch_guard_count)));
    selection_obj.emplace_back("candidate_error_training_base_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.candidate_error_training_base_count)));
    selection_obj.emplace_back("candidate_error_centroid_order_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.candidate_error_centroid_order_count)));
    selection_obj.emplace_back("candidate_error_cdf_fit_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.candidate_error_cdf_fit_count)));
    selection_obj.emplace_back("candidate_error_eval_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.candidate_error_eval_count)));
    selection_obj.emplace_back("candidate_error_unknown_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.candidate_error_unknown_count)));
    selection_obj.emplace_back("train_time_ms_sum",
                               rm_model::json::Value(result.train_time_ms_sum));
    selection_obj.emplace_back("train_gather_skeleton_ms_sum",
                               rm_model::json::Value(result.train_gather_skeleton_ms_sum));
    selection_obj.emplace_back("train_cluster_assign_ms_sum",
                               rm_model::json::Value(result.train_cluster_assign_ms_sum));
    selection_obj.emplace_back("train_assign_centroids_ms_sum",
                               rm_model::json::Value(result.train_assign_centroids_ms_sum));
    selection_obj.emplace_back("train_centroid_order_ms_sum",
                               rm_model::json::Value(result.train_centroid_order_ms_sum));
    selection_obj.emplace_back("train_range_alloc_ms_sum",
                               rm_model::json::Value(result.train_range_alloc_ms_sum));
    selection_obj.emplace_back("train_cdf_fit_ms_sum",
                               rm_model::json::Value(result.train_cdf_fit_ms_sum));
    selection_obj.emplace_back("train_model_assembly_ms_sum",
                               rm_model::json::Value(result.train_model_assembly_ms_sum));
    selection_obj.emplace_back("scheduler_ready_cached_candidates",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.scheduler_ready_cached_candidates)));
    selection_obj.emplace_back("scheduler_in_flight_candidates",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.scheduler_in_flight_candidates)));
    selection_obj.emplace_back("scheduler_missing_candidates",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.scheduler_missing_candidates)));
    selection_obj.emplace_back("scheduler_single_admission_batches",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.scheduler_single_admission_batches)));
    selection_obj.emplace_back("scheduler_reuse_prioritized_candidates",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.scheduler_reuse_prioritized_candidates)));
    selection_obj.emplace_back("scheduler_fresh_base_candidates",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.scheduler_fresh_base_candidates)));
    selection_obj.emplace_back("strategy_frontier_candidates",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.strategy_frontier_candidates)));
    selection_obj.emplace_back("strategy_compacted_candidates",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.strategy_compacted_candidates)));
    selection_obj.emplace_back("eval_nearest_cache_hits",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.eval_nearest_cache_hits)));
    selection_obj.emplace_back("eval_nearest_cache_misses",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.eval_nearest_cache_misses)));
    selection_obj.emplace_back("eval_nearest_cache_inflight_bypasses",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(
                                       result.eval_nearest_cache_inflight_bypasses)));
    selection_obj.emplace_back("eval_nearest_cache_prewarm_requests",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(
                                       result.eval_nearest_cache_prewarm_requests)));
    selection_obj.emplace_back("eval_nearest_cache_prewarm_ready_models",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(
                                       result.eval_nearest_cache_prewarm_ready_models)));
    selection_obj.emplace_back("eval_nearest_cache_prewarm_skipped",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(
                                       result.eval_nearest_cache_prewarm_skipped)));
    selection_obj.emplace_back("eval_nearest_cache_prewarm_failures",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(
                                       result.eval_nearest_cache_prewarm_failures)));
    selection_obj.emplace_back("eval_nearest_cache_prewarm_threads",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(
                                       result.eval_nearest_cache_prewarm_threads)));
    selection_obj.emplace_back("eval_nearest_cache_prewarm_ms",
                               rm_model::json::Value(
                                   result.eval_nearest_cache_prewarm_ms));
    selection_obj.emplace_back("eval_nearest_cache_wait_ms",
                               rm_model::json::Value(result.eval_nearest_cache_wait_ms));
    selection_obj.emplace_back(
        "eval_nearest_cache_sampled_wait_ms",
        rm_model::json::Value(result.eval_nearest_cache_sampled_wait_ms));
    selection_obj.emplace_back(
        "eval_nearest_cache_calibration_wait_ms",
        rm_model::json::Value(result.eval_nearest_cache_calibration_wait_ms));
    selection_obj.emplace_back("eval_nearest_cache_build_ms",
                               rm_model::json::Value(result.eval_nearest_cache_build_ms));
    selection_obj.emplace_back(
        "eval_nearest_cache_sampled_build_ms",
        rm_model::json::Value(result.eval_nearest_cache_sampled_build_ms));
    selection_obj.emplace_back(
        "eval_nearest_cache_calibration_build_ms",
        rm_model::json::Value(result.eval_nearest_cache_calibration_build_ms));
    selection_obj.emplace_back(
        "eval_nearest_cache_distance_terms",
        rm_model::json::Value(
            static_cast<uint64_t>(result.eval_nearest_cache_distance_terms)));
    selection_obj.emplace_back(
        "eval_nearest_cache_skipped_distance_terms",
        rm_model::json::Value(
            static_cast<uint64_t>(result.eval_nearest_cache_skipped_distance_terms)));
    selection_obj.emplace_back(
        "eval_nearest_cache_early_abandon_builds",
        rm_model::json::Value(
            static_cast<uint64_t>(result.eval_nearest_cache_early_abandon_builds)));
    selection_obj.emplace_back("eval_nearest_cache_entries",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.eval_nearest_cache_entries)));
    selection_obj.emplace_back("eval_nearest_cache_memory_bytes",
                               rm_model::json::Value(
                                   result.eval_nearest_cache_memory_bytes));
    selection_obj.emplace_back("eval_nearest_cache_sampled_budget_bytes",
                               rm_model::json::Value(
                                   result.eval_nearest_cache_sampled_budget_bytes));
    selection_obj.emplace_back("eval_nearest_cache_sampled_memory_bytes",
                               rm_model::json::Value(
                                   result.eval_nearest_cache_sampled_memory_bytes));
    selection_obj.emplace_back("eval_nearest_cache_calibration_memory_bytes",
                               rm_model::json::Value(
                                   result.eval_nearest_cache_calibration_memory_bytes));
    selection_obj.emplace_back("eval_nearest_cache_evictions",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.eval_nearest_cache_evictions)));
    selection_obj.emplace_back("eval_nearest_cache_evicted_bytes",
                               rm_model::json::Value(
                                   result.eval_nearest_cache_evicted_bytes));
    selection_obj.emplace_back("eval_base_rank_cache_hits",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.eval_base_rank_cache_hits)));
    selection_obj.emplace_back("eval_base_rank_cache_misses",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.eval_base_rank_cache_misses)));
    selection_obj.emplace_back("eval_base_rank_cache_wait_ms",
                               rm_model::json::Value(result.eval_base_rank_cache_wait_ms));
    selection_obj.emplace_back("eval_base_rank_cache_build_ms",
                               rm_model::json::Value(result.eval_base_rank_cache_build_ms));
    selection_obj.emplace_back("eval_base_rank_cache_nearest_ms",
                               rm_model::json::Value(result.eval_base_rank_cache_nearest_ms));
    selection_obj.emplace_back("eval_base_rank_cache_hash_ms",
                               rm_model::json::Value(result.eval_base_rank_cache_hash_ms));
    selection_obj.emplace_back("eval_base_rank_cache_sort_ms",
                               rm_model::json::Value(result.eval_base_rank_cache_sort_ms));
    selection_obj.emplace_back("eval_base_rank_cache_rank_index_ms",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_rank_index_ms));
    selection_obj.emplace_back("eval_base_rank_cache_overhead_ms",
                               rm_model::json::Value(result.eval_base_rank_cache_overhead_ms));
    selection_obj.emplace_back("eval_base_rank_cache_entries",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.eval_base_rank_cache_entries)));
    selection_obj.emplace_back("eval_base_rank_cache_memory_bytes",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_memory_bytes));
    selection_obj.emplace_back("eval_base_rank_cache_sampled_budget_bytes",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_sampled_budget_bytes));
    selection_obj.emplace_back("eval_base_rank_cache_sampled_memory_bytes",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_sampled_memory_bytes));
    selection_obj.emplace_back("eval_base_rank_cache_calibration_memory_bytes",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_calibration_memory_bytes));
    selection_obj.emplace_back("eval_base_rank_cache_evictions",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.eval_base_rank_cache_evictions)));
    selection_obj.emplace_back("eval_base_rank_cache_evicted_bytes",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_evicted_bytes));
    rm_model::json::Value::Array base_rank_attribution;
    base_rank_attribution.reserve(result.eval_base_rank_cache_attribution.size());
    for (const auto& row : result.eval_base_rank_cache_attribution) {
      base_rank_attribution.push_back(base_rank_cache_attribution_to_json(row));
    }
    selection_obj.emplace_back("eval_base_rank_cache_attribution",
                               rm_model::json::Value(std::move(base_rank_attribution)));
    selection_obj.emplace_back("model_train_budget_hit",
                               rm_model::json::Value(result.model_train_budget_hit));
    selection_obj.emplace_back("max_search_seconds",
                               rm_model::json::Value(static_cast<uint64_t>(options.max_search_seconds)));
    selection_obj.emplace_back("effective_max_search_seconds",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.effective_max_search_seconds)));
    selection_obj.emplace_back("elapsed_search_seconds",
                               rm_model::json::Value(result.elapsed_search_seconds));
    selection_obj.emplace_back("search_time_budget_hit",
                               rm_model::json::Value(result.search_time_budget_hit));
    selection_obj.emplace_back("remaining_time_guard_hit",
                               rm_model::json::Value(result.remaining_time_guard_hit));
    selection_obj.emplace_back("final_calibration_reserve_seconds",
                               rm_model::json::Value(
                                   result.final_calibration_reserve_seconds));
    selection_obj.emplace_back("final_calibration_reserve_hit",
                               rm_model::json::Value(result.final_calibration_reserve_hit));
    selection_obj.emplace_back("stopped_reason",
                               rm_model::json::Value(result.stopped_reason));
    selection_obj.emplace_back("plateau_rounds",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.plateau_rounds)));
    selection_obj.emplace_back("best_quality",
                               rm_model::json::Value(result.best_quality));
    selection_obj.emplace_back("best_margin",
                               rm_model::json::Value(result.best_margin));
    selection_obj.emplace_back("max_stagnant_contexts",
                               rm_model::json::Value(static_cast<uint64_t>(options.max_stagnant_contexts)));
    selection_obj.emplace_back("min_recall_improvement",
                               rm_model::json::Value(options.min_recall_improvement));
    selection_obj.emplace_back("min_quality_improvement",
                               rm_model::json::Value(options.min_quality_improvement));
    selection_obj.emplace_back("contexts_skipped_low_potential",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.contexts_skipped_low_potential)));
    selection_obj.emplace_back("skeleton_capacity_scaling",
                               rm_model::json::Value(options.skeleton_capacity_scaling));
    selection_obj.emplace_back("selector_capacity_policy",
                               rm_model::json::Value(
                                   "scale_aware_recall_first_config_families"));
    selection_obj.emplace_back("centroid_order_policy",
                               rm_model::json::Value(
                                   options.enable_graph_centroid_order
                                       ? "euclidean_mst_2opt_plus_nsw_graph_affinity"
                                       : "euclidean_mst_2opt"));
    selection_obj.emplace_back("advanced_search", rm_model::json::Value(options.advanced_search));
    selection_obj.emplace_back("optimize_for_recall",
                               rm_model::json::Value(options.optimize_for_recall));
    selection_obj.emplace_back("max_recall_refine_rounds",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(options.max_recall_refine_rounds)));
    selection_obj.emplace_back("beam_width",
                               rm_model::json::Value(static_cast<uint64_t>(options.beam_width)));
    selection_obj.emplace_back("beam_rounds",
                               rm_model::json::Value(static_cast<uint64_t>(options.beam_rounds)));
    selection_obj.emplace_back("hill_climb_steps",
                               rm_model::json::Value(static_cast<uint64_t>(options.hill_climb_steps)));
    selection_obj.emplace_back("beam_neighbor_limit",
                               rm_model::json::Value(static_cast<uint64_t>(options.beam_neighbor_limit)));
    selection_obj.emplace_back("strategy_refine_evaluated",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.strategy_refine_evaluated)));
    selection_obj.emplace_back("strategy_refine_rounds",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.strategy_refine_rounds)));
    selection_obj.emplace_back("recall_refine_evaluated",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.recall_refine_evaluated)));
    selection_obj.emplace_back("recall_refine_rounds",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.recall_refine_rounds)));
    selection_obj.emplace_back("post_exploration_exploitation",
                               rm_model::json::Value(options.post_exploration_exploitation));
    selection_obj.emplace_back("post_exploration_candidates",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(options.post_exploration_candidates)));
    selection_obj.emplace_back("post_exploration_evaluated",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.post_exploration_evaluated)));
    selection_obj.emplace_back("post_exploration_rounds",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.post_exploration_rounds)));
    selection_obj.emplace_back("final_calibration_count",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(options.final_calibration_count)));
    selection_obj.emplace_back("final_calibration_full_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   options.final_calibration_full_count)));
    selection_obj.emplace_back("effective_final_calibration_full_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.effective_final_calibration_full_count)));
    selection_obj.emplace_back("final_calibration_seed_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.final_calibration_seed_count)));
    selection_obj.emplace_back("final_calibration_screening",
                               rm_model::json::Value(options.final_calibration_screening));
    selection_obj.emplace_back("final_calibration_screen_candidate_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   options.final_calibration_screen_candidate_count)));
    selection_obj.emplace_back("effective_final_calibration_screen_candidate_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.effective_final_calibration_screen_candidate_count)));
    selection_obj.emplace_back("final_calibration_screen_query_limit",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   options.final_calibration_screen_query_limit)));
    selection_obj.emplace_back("effective_final_calibration_screen_query_limit",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.effective_final_calibration_screen_query_limit)));
    selection_obj.emplace_back("final_calibration_screen_evaluated",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.final_calibration_screen_evaluated)));
    selection_obj.emplace_back("final_calibration_screen_promoted",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.final_calibration_screen_promoted)));
    selection_obj.emplace_back("final_calibration_role_promoted",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.final_calibration_role_promoted)));
    selection_obj.emplace_back("final_calibration_recall_promoted",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.final_calibration_recall_promoted)));
    selection_obj.emplace_back("final_calibration_screen_status",
                               rm_model::json::Value(result.final_calibration_screen_status));
    selection_obj.emplace_back("final_calibration_query_limit",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(options.final_calibration_query_limit)));
    selection_obj.emplace_back("final_calibration_evaluated",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.final_calibration_evaluated)));
    selection_obj.emplace_back("final_calibration_reused",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.final_calibration_reused)));
    rm_model::json::Value::Array calibration_probes;
    calibration_probes.reserve(result.final_calibration_progressive_probes.size());
    for (const auto& probe : result.final_calibration_progressive_probes) {
      calibration_probes.push_back(calibration_probe_to_json(probe));
    }
    selection_obj.emplace_back("final_calibration_progressive_probes",
                               rm_model::json::Value(std::move(calibration_probes)));
    if (options.recall_target.has_value()) {
      selection_obj.emplace_back("recall_target", rm_model::json::Value(*options.recall_target));
      selection_obj.emplace_back("recall_target_met",
                                 rm_model::json::Value(result.recall_target_met));
      selection_obj.emplace_back("recall_target_met_recommended",
                                 rm_model::json::Value(
                                     static_cast<uint64_t>(result.recall_target_met_recommended)));
    }
  }
  if (!options.target_skeleton_values.empty()) {
    rm_model::json::Value::Array target_values;
    target_values.reserve(options.target_skeleton_values.size());
    for (uint64_t value : normalize_target_skeleton_values(options.target_skeleton_values)) {
      target_values.push_back(rm_model::json::Value(value));
    }
    selection_obj.emplace_back("target_skeleton_values", rm_model::json::Value(std::move(target_values)));
  }
  if (!options.target_skeleton_percentages.empty()) {
    rm_model::json::Value::Array target_percentages;
    target_percentages.reserve(options.target_skeleton_percentages.size());
    for (double value : normalize_target_skeleton_percentages(options.target_skeleton_percentages)) {
      target_percentages.push_back(rm_model::json::Value(value));
    }
    selection_obj.emplace_back("target_skeleton_percentages",
                               rm_model::json::Value(std::move(target_percentages)));
  }
  if (uses_auto_target_skeleton_policy(options)) {
    selection_obj.emplace_back("target_skeleton_policy",
                               rm_model::json::Value(std::string("auto_scale")));
  }
  std::set<uint64_t> resolved_target_skeleton_values;
  for (const auto& item : result.candidates) {
    if (item.config.target_skeleton > 0) {
      resolved_target_skeleton_values.insert(item.config.target_skeleton);
    }
  }
  if (!resolved_target_skeleton_values.empty()) {
    rm_model::json::Value::Array resolved_values;
    resolved_values.reserve(resolved_target_skeleton_values.size());
    for (uint64_t value : resolved_target_skeleton_values) {
      resolved_values.push_back(rm_model::json::Value(value));
    }
    selection_obj.emplace_back("resolved_target_skeleton_values",
                               rm_model::json::Value(std::move(resolved_values)));
  }
  if (options.nsw_layer.has_value()) {
    selection_obj.emplace_back("nsw_layer", rm_model::json::Value(static_cast<int64_t>(*options.nsw_layer)));
  }
  selection_obj.emplace_back("eval_k", rm_model::json::Value(static_cast<uint64_t>(options.eval_k)));
  selection_obj.emplace_back("eval_window", rm_model::json::Value(options.eval_window));
  if (!result.effective_eval_windows.empty()) {
    selection_obj.emplace_back("eval_windows",
                               uint64_array_to_json(result.effective_eval_windows));
  } else if (!options.eval_window_values.empty()) {
    selection_obj.emplace_back("eval_windows",
                               uint64_array_to_json(options.eval_window_values));
  }
  selection_obj.emplace_back("coarse_base_limit",
                             rm_model::json::Value(static_cast<uint64_t>(options.coarse_base_limit)));
  selection_obj.emplace_back("coarse_query_limit",
                             rm_model::json::Value(static_cast<uint64_t>(options.coarse_query_limit)));
  selection_obj.emplace_back("eval_base_limit",
                             rm_model::json::Value(static_cast<uint64_t>(options.eval_base_limit)));
  selection_obj.emplace_back("eval_query_limit",
                             rm_model::json::Value(static_cast<uint64_t>(options.eval_query_limit)));
  selection_obj.emplace_back("coarse_latency_iterations",
                             rm_model::json::Value(static_cast<uint64_t>(options.coarse_latency_iterations)));
  selection_obj.emplace_back("latency_iterations",
                             rm_model::json::Value(static_cast<uint64_t>(options.latency_iterations)));
  const SelectorWeights& effective_weights = result.effective_weights;
  selection_obj.emplace_back("weights",
                             rm_model::json::Value(rm_model::json::Value::Object{
                                 {"recall", rm_model::json::Value(effective_weights.recall)},
                                 {"rank_distance", rm_model::json::Value(effective_weights.rank_distance)},
                                 {"latency", rm_model::json::Value(effective_weights.latency)},
                                 {"model_size", rm_model::json::Value(effective_weights.model_size)},
                                 {"train_time", rm_model::json::Value(effective_weights.train_time)},
                             }));
  selection_obj.emplace_back("requested_weights",
                             rm_model::json::Value(rm_model::json::Value::Object{
                                 {"recall", rm_model::json::Value(options.weights.recall)},
                                 {"rank_distance", rm_model::json::Value(options.weights.rank_distance)},
                                 {"latency", rm_model::json::Value(options.weights.latency)},
                                 {"model_size", rm_model::json::Value(options.weights.model_size)},
                                 {"train_time", rm_model::json::Value(options.weights.train_time)},
                             }));
  selection_obj.emplace_back("auto_dataset_weighting",
                             rm_model::json::Value(options.auto_dataset_weighting));
  selection_obj.emplace_back("auto_dataset_weighting_applied",
                             rm_model::json::Value(result.auto_dataset_weighting_applied));
  if (include_flat_selection_compatibility_fields) {
    selection_obj.emplace_back("selector_memory_budget_bytes",
                               rm_model::json::Value(options.selector_memory_budget_bytes));
    selection_obj.emplace_back("effective_selector_memory_budget_bytes",
                               rm_model::json::Value(result.effective_selector_memory_budget_bytes));
    selection_obj.emplace_back("selector_peak_parallelism",
                               rm_model::json::Value(
                                   static_cast<uint64_t>(result.selector_peak_parallelism)));
    selection_obj.emplace_back("selector_peak_threads_per_candidate",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.selector_peak_threads_per_candidate)));
    selection_obj.emplace_back("selector_memory_budget_limited",
                               rm_model::json::Value(result.selector_memory_budget_limited));
    if (!result.timings.empty()) {
      rm_model::json::Value::Array timings;
      timings.reserve(result.timings.size());
      for (const auto& timing : result.timings) {
        rm_model::json::Value::Object row;
        row.emplace_back("name", rm_model::json::Value(timing.name));
        row.emplace_back("elapsed_ms", rm_model::json::Value(timing.elapsed_ms));
        timings.push_back(rm_model::json::Value(std::move(row)));
      }
      selection_obj.emplace_back("timings", rm_model::json::Value(std::move(timings)));
    }
    if (!result.context_stats.empty()) {
      rm_model::json::Value::Array context_stats;
      context_stats.reserve(result.context_stats.size());
      for (const auto& stats : result.context_stats) {
        rm_model::json::Value::Object row;
        row.emplace_back("target_skeleton", rm_model::json::Value(stats.target_skeleton));
        if (!stats.requested_target_skeletons.empty()) {
          rm_model::json::Value::Array requested_targets;
          requested_targets.reserve(stats.requested_target_skeletons.size());
          for (uint64_t target : stats.requested_target_skeletons) {
            requested_targets.push_back(rm_model::json::Value(target));
          }
          row.emplace_back("requested_target_skeletons",
                           rm_model::json::Value(std::move(requested_targets)));
        }
        row.emplace_back("skeleton_node_count",
                         rm_model::json::Value(static_cast<uint64_t>(stats.skeleton_node_count)));
        row.emplace_back("skeleton_layer", rm_model::json::Value(static_cast<int64_t>(stats.skeleton_layer)));
        row.emplace_back("context_model_train_budget",
                         rm_model::json::Value(static_cast<uint64_t>(stats.context_model_train_budget)));
        row.emplace_back("model_trains_started_before",
                         rm_model::json::Value(static_cast<uint64_t>(stats.model_trains_started_before)));
        row.emplace_back("model_trains_started_after",
                         rm_model::json::Value(static_cast<uint64_t>(stats.model_trains_started_after)));
        row.emplace_back("context_model_train_budget_hit",
                         rm_model::json::Value(stats.context_model_train_budget_hit));
        row.emplace_back("phase1_candidates",
                         rm_model::json::Value(static_cast<uint64_t>(stats.phase1_candidates)));
        row.emplace_back("phase1_ok",
                         rm_model::json::Value(static_cast<uint64_t>(stats.phase1_ok)));
        row.emplace_back("phase2_candidates",
                         rm_model::json::Value(static_cast<uint64_t>(stats.phase2_candidates)));
        row.emplace_back("phase2_ok",
                         rm_model::json::Value(static_cast<uint64_t>(stats.phase2_ok)));
        row.emplace_back("phase2_candidate_limit",
                         rm_model::json::Value(
                             static_cast<uint64_t>(stats.phase2_candidate_limit)));
        row.emplace_back("phase2_refine_evaluated",
                         rm_model::json::Value(
                             static_cast<uint64_t>(stats.phase2_refine_evaluated)));
        row.emplace_back("phase2_refine_rounds",
                         rm_model::json::Value(
                             static_cast<uint64_t>(stats.phase2_refine_rounds)));
        row.emplace_back("phase2_frontier_candidates",
                         rm_model::json::Value(
                             static_cast<uint64_t>(stats.phase2_frontier_candidates)));
        row.emplace_back("phase2_compacted_candidates",
                         rm_model::json::Value(
                             static_cast<uint64_t>(stats.phase2_compacted_candidates)));
        row.emplace_back("phase2_plateau_rounds",
                         rm_model::json::Value(
                             static_cast<uint64_t>(stats.phase2_plateau_rounds)));
        row.emplace_back("budget_failures",
                         rm_model::json::Value(static_cast<uint64_t>(stats.budget_failures)));
        row.emplace_back("effective_K_values",
                         uint32_array_to_json(stats.effective_K_values));
        row.emplace_back("pruned_K_values",
                         uint32_array_to_json(stats.pruned_K_values));
        row.emplace_back("effective_centroid_knn_values",
                         uint32_array_to_json(stats.effective_centroid_knn_values));
        row.emplace_back("effective_cdf_branching_values",
                         uint64_array_to_json(stats.effective_cdf_branching_values));
        row.emplace_back("min_vectors_per_centroid",
                         rm_model::json::Value(stats.min_vectors_per_centroid));
        row.emplace_back("max_vectors_per_centroid",
                         rm_model::json::Value(stats.max_vectors_per_centroid));
        row.emplace_back("min_skeleton_nodes_per_centroid",
                         rm_model::json::Value(stats.min_skeleton_nodes_per_centroid));
        row.emplace_back("max_skeleton_nodes_per_centroid",
                         rm_model::json::Value(stats.max_skeleton_nodes_per_centroid));
        row.emplace_back("best_phase1_recall",
                         rm_model::json::Value(stats.best_phase1_recall));
        row.emplace_back("best_phase2_recall",
                         rm_model::json::Value(stats.best_phase2_recall));
        row.emplace_back("best_phase1_quality",
                         rm_model::json::Value(stats.best_phase1_quality));
        row.emplace_back("best_phase2_quality",
                         rm_model::json::Value(stats.best_phase2_quality));
        row.emplace_back("phase2_quality_gain",
                         rm_model::json::Value(stats.phase2_quality_gain));
        row.emplace_back("promotion_decision",
                         rm_model::json::Value(stats.promotion_decision));
        row.emplace_back("phase2_stopped_reason",
                         rm_model::json::Value(stats.phase2_stopped_reason));
        row.emplace_back("elapsed_ms", rm_model::json::Value(stats.elapsed_ms));
        context_stats.push_back(rm_model::json::Value(std::move(row)));
      }
      selection_obj.emplace_back("context_stats",
                                 rm_model::json::Value(std::move(context_stats)));

      rm_model::json::Value::Array promotion_history;
      promotion_history.reserve(result.context_stats.size());
      for (const auto& stats : result.context_stats) {
        rm_model::json::Value::Object row;
        row.emplace_back("target_skeleton", rm_model::json::Value(stats.target_skeleton));
        row.emplace_back("skeleton_node_count",
                         rm_model::json::Value(static_cast<uint64_t>(stats.skeleton_node_count)));
        row.emplace_back("promotion_decision",
                         rm_model::json::Value(stats.promotion_decision));
        row.emplace_back("phase2_candidate_limit",
                         rm_model::json::Value(
                             static_cast<uint64_t>(stats.phase2_candidate_limit)));
        row.emplace_back("phase2_refine_evaluated",
                         rm_model::json::Value(
                             static_cast<uint64_t>(stats.phase2_refine_evaluated)));
        row.emplace_back("phase2_refine_rounds",
                         rm_model::json::Value(
                             static_cast<uint64_t>(stats.phase2_refine_rounds)));
        row.emplace_back("phase2_plateau_rounds",
                         rm_model::json::Value(
                             static_cast<uint64_t>(stats.phase2_plateau_rounds)));
        row.emplace_back("best_phase1_quality",
                         rm_model::json::Value(stats.best_phase1_quality));
        row.emplace_back("best_phase2_quality",
                         rm_model::json::Value(stats.best_phase2_quality));
        row.emplace_back("phase2_quality_gain",
                         rm_model::json::Value(stats.phase2_quality_gain));
        row.emplace_back("phase2_stopped_reason",
                         rm_model::json::Value(stats.phase2_stopped_reason));
        promotion_history.push_back(rm_model::json::Value(std::move(row)));
      }
      selection_obj.emplace_back("promotion_history",
                                 rm_model::json::Value(std::move(promotion_history)));
    }
  }
  if (report_options.size_budget_bytes.has_value()) {
    selection_obj.emplace_back("size_budget_bytes",
                               rm_model::json::Value(*report_options.size_budget_bytes));
  }
  return rm_model::json::Value(std::move(selection_obj));
}

} // namespace vortex::selector_internal
