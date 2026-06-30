#ifndef VORTEX_V1_MODEL_SELECTOR_H
#define VORTEX_V1_MODEL_SELECTOR_H

#include "rm_model/json.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vortex {

struct SelectorCandidateConfig {
  uint64_t target_skeleton = 0;
  uint32_t K = 0;
  uint32_t centroid_knn = 0;
  std::string cdf_model_spec = "linear,linear";
  uint64_t cdf_branching_factor = 32;
  bool enable_2opt = true;
  uint32_t two_opt_iterations = 8;
  bool enable_graph_centroid_order = true;
};

struct SelectorWeights {
  double recall = 0.55;
  double rank_distance = 0.10;
  double latency = 0.20;
  double model_size = 0.10;
  double train_time = 0.05;
};

struct SelectorOptions {
  std::filesystem::path dataset_path;
  std::filesystem::path nsw_path;
  std::optional<std::filesystem::path> nsw_index_path;
  std::vector<uint64_t> target_skeleton_values;
  std::vector<double> target_skeleton_percentages;
  std::optional<int> nsw_layer;
  std::optional<std::filesystem::path> query_path;
  uint32_t hash_bits = 64;
  uint32_t threads = 1;
  uint32_t selector_parallelism = 0;
  uint64_t seed = 42;
  uint64_t dataset_count_hint = 0;
  std::string selector_profile = "auto";
  uint32_t max_target_skeleton_contexts = 0;
  uint32_t max_model_trains = 0;

  std::vector<uint32_t> K_values;
  std::vector<uint32_t> centroid_knn_values;
  std::vector<uint64_t> cdf_branching_values;
  std::vector<std::string> cdf_model_specs;
  std::vector<uint32_t> two_opt_iterations_values;
  bool include_disable_2opt = true;
  bool include_disable_graph_centroid_order = true;

  uint32_t phase1_keep = 8;
  uint32_t max_phase2_candidates = 24;
  uint32_t recommend_count = 8;

  uint32_t eval_k = 10;
  uint64_t eval_window = 1000;
  std::vector<uint64_t> eval_window_values;
  std::vector<uint32_t> eval_node_count_values;

  uint32_t coarse_base_limit = 4000;
  uint32_t coarse_query_limit = 64;
  uint32_t coarse_latency_iterations = 256;
  uint32_t coarse_latency_warmup = 64;

  uint32_t eval_base_limit = 12000;
  uint32_t eval_query_limit = 256;
  uint32_t latency_iterations = 1024;
  uint32_t latency_warmup = 128;

  std::optional<uint64_t> size_budget_bytes;
  SelectorWeights weights;
  bool auto_dataset_weighting = true;
  std::optional<double> recall_target;
  bool skeleton_capacity_scaling = true;
  uint32_t max_search_seconds = 0;
  uint32_t max_stagnant_contexts = 0;
  double min_recall_improvement = 0.002;
  double min_quality_improvement = 0.002;
  bool optimize_for_recall = false;
  uint32_t max_recall_refine_rounds = 2;
  bool post_exploration_exploitation = true;
  uint32_t post_exploration_candidates = 0;
  uint32_t final_calibration_count = 0;
  uint32_t final_calibration_full_count = 0;
  bool final_calibration_screening = true;
  uint32_t final_calibration_screen_candidate_count = 0;
  uint32_t final_calibration_screen_query_limit = 0;
  uint32_t final_calibration_query_limit = 0;
  uint64_t selector_memory_budget_bytes = 0;
  bool use_full_dataset_for_assignments = true;
  bool auto_assignment_sample_limit = true;
  uint64_t assignment_sample_limit = 0;
  // Global kill switch for graph-aware centroid ordering. When true, the
  // selector may still compare graph-aware and Euclidean-only candidate orders.
  bool enable_graph_centroid_order = true;
  bool advanced_search = true;
  bool attribution_guided_compaction = true;
  uint32_t beam_width = 6;
  uint32_t beam_rounds = 2;
  uint32_t hill_climb_steps = 2;
  uint32_t beam_neighbor_limit = 64;
};

struct SelectorJsonOptions {
  // Keep the historical flat selection.* telemetry by default. New consumers
  // can disable it and read the grouped selection.diagnostics schema instead.
  bool include_flat_selection_compatibility_fields = true;
};

struct SelectorCandidateMetrics {
  SelectorCandidateConfig config;
  std::string nsw_path;
  uint32_t skeleton_node_count = 0;
  int skeleton_layer = -1;

  bool ok = false;
  std::string error;
  std::string error_stage;
  std::string phase;

  double objective_score = -1.0;
  double train_time_ms = 0.0;
  bool train_base_cache_hit = false;
  bool train_order_cache_hit = false;
  bool train_cdf_cache_hit = false;
  double train_base_cache_wait_ms = 0.0;
  double train_order_cache_wait_ms = 0.0;
  double train_cdf_cache_wait_ms = 0.0;
  double train_base_build_ms = 0.0;
  double train_order_build_ms = 0.0;
  double train_cdf_cache_build_ms = 0.0;
  double train_gather_skeleton_ms = 0.0;
  double train_cluster_assign_ms = 0.0;
  double train_assign_centroids_ms = 0.0;
  double train_centroid_order_ms = 0.0;
  double train_range_alloc_ms = 0.0;
  double train_cdf_fit_ms = 0.0;
  double train_model_assembly_ms = 0.0;

  uint64_t model_size_bytes = 0;
  uint64_t model_memory_bytes = 0;
  uint32_t centroid_count = 0;
  uint32_t active_centroid_count = 0;

  double recall_at_k_in_window = 0.0;
  std::vector<uint64_t> recall_windows;
  std::vector<double> recall_at_k_by_window;
  double recall_auc_log_window = 0.0;
  uint64_t min_window_for_recall_target = 0;
  bool recall_target_met_by_curve = false;
  double query_recall_p05 = 0.0;
  double query_recall_p50 = 0.0;
  double query_recall_p95 = 0.0;
  double mean_rank_distance_norm = 0.0;
  double rank_distance_norm_p50 = 0.0;
  double rank_distance_norm_p95 = 0.0;
  double rank_distance_norm_p99 = 0.0;
  std::vector<uint32_t> node_counts;
  std::vector<double> same_node_hit_by_count;
  std::vector<double> near_1_node_hit_by_count;
  std::vector<double> near_2_node_hit_by_count;
  std::vector<double> near_4_node_hit_by_count;
  std::vector<double> mean_node_distance_norm_by_count;
  std::vector<double> overlay_match_by_count;
  double node_locality_score = 0.0;
  double overlay_match_score = 0.0;
  double query_overlay_match_p05 = 0.0;
  double query_overlay_match_p50 = 0.0;
  double query_overlay_match_p95 = 0.0;
  double locality_quality_score = 0.0;
  double latency_avg_ms = 0.0;
  double latency_p95_ms = 0.0;
  double latency_p99_ms = 0.0;
  double eval_total_ms = 0.0;
  double eval_hash_base_ms = 0.0;
  double eval_sort_base_ms = 0.0;
  double eval_rank_index_ms = 0.0;
  double eval_query_ms = 0.0;
  double eval_latency_ms = 0.0;
  uint32_t eval_hash_base_threads = 1;
  uint32_t eval_rank_index_threads = 1;
};

struct SelectorPhaseTiming {
  std::string name;
  double elapsed_ms = 0.0;
};

struct SelectorContextStats {
  uint64_t target_skeleton = 0;
  std::vector<uint64_t> requested_target_skeletons;
  uint32_t skeleton_node_count = 0;
  int skeleton_layer = -1;
  uint32_t context_model_train_budget = 0;
  uint32_t model_trains_started_before = 0;
  uint32_t model_trains_started_after = 0;
  bool context_model_train_budget_hit = false;
  uint32_t phase1_candidates = 0;
  uint32_t phase1_ok = 0;
  uint32_t phase2_candidates = 0;
  uint32_t phase2_ok = 0;
  uint32_t phase2_candidate_limit = 0;
  uint32_t phase2_refine_evaluated = 0;
  uint32_t phase2_refine_rounds = 0;
  uint32_t phase2_frontier_candidates = 0;
  uint32_t phase2_compacted_candidates = 0;
  uint32_t phase2_plateau_rounds = 0;
  uint32_t budget_failures = 0;
  std::vector<uint32_t> effective_K_values;
  std::vector<uint32_t> pruned_K_values;
  std::vector<uint32_t> effective_centroid_knn_values;
  std::vector<uint64_t> effective_cdf_branching_values;
  double min_vectors_per_centroid = 0.0;
  double max_vectors_per_centroid = 0.0;
  double min_skeleton_nodes_per_centroid = 0.0;
  double max_skeleton_nodes_per_centroid = 0.0;
  double best_phase1_recall = -1.0;
  double best_phase2_recall = -1.0;
  double best_phase1_quality = -1.0;
  double best_phase2_quality = -1.0;
  double phase2_quality_gain = 0.0;
  std::string promotion_decision;
  std::string phase2_stopped_reason;
  double elapsed_ms = 0.0;
};

struct SelectorCalibrationProbe {
  std::string name;
  uint64_t probe_base_count = 0;
  uint64_t probe_query_count = 0;
  uint64_t final_base_count = 0;
  uint64_t final_query_count = 0;
  uint32_t evaluated = 0;
  double probe_peak_recall = 0.0;
  double final_peak_recall = 0.0;
  bool peak_recall_match = false;
  bool knee_match = false;
  bool fast_match = false;
  bool small_match = false;
  uint32_t matched_role_count = 0;
};

// Measurement-only breakdown of exact base-rank cache work. Rows are grouped by
// the selector phase and candidate family that accessed or built a cache entry.
struct SelectorBaseRankCacheAttribution {
  std::string phase;
  std::string nsw_path;
  uint64_t target_skeleton = 0;
  uint32_t skeleton_node_count = 0;
  int skeleton_layer = -1;
  uint32_t K = 0;
  uint32_t centroid_knn = 0;
  std::string cdf_model_spec;
  uint64_t cdf_branching_factor = 0;
  bool enable_2opt = false;
  uint32_t two_opt_iterations = 0;
  bool enable_graph_centroid_order = false;
  uint64_t base_count = 0;
  uint64_t query_count = 0;
  bool calibration_tier = false;
  std::vector<std::string> roles;
  uint32_t hits = 0;
  uint32_t misses = 0;
  uint32_t builds = 0;
  double wait_ms = 0.0;
  double build_ms = 0.0;
  double nearest_ms = 0.0;
  double hash_ms = 0.0;
  double sort_ms = 0.0;
  double rank_index_ms = 0.0;
  double overhead_ms = 0.0;
  uint64_t added_memory_bytes = 0;
  uint64_t retained_memory_bytes = 0;
  uint32_t evictions = 0;
  uint64_t evicted_bytes = 0;
};

struct SelectorResult {
  std::vector<SelectorCandidateMetrics> candidates;
  std::vector<SelectorCandidateMetrics> pareto_front;
  std::vector<SelectorCandidateMetrics> recommended;
  std::optional<SelectorCandidateMetrics> best;
  std::optional<SelectorCandidateMetrics> recommended_peak_recall;
  std::optional<SelectorCandidateMetrics> recommended_knee;
  std::optional<SelectorCandidateMetrics> recommended_fast;
  std::optional<SelectorCandidateMetrics> recommended_small;
  SelectorWeights effective_weights;
  uint64_t dataset_count = 0;
  std::vector<uint64_t> effective_eval_windows;
  std::vector<uint32_t> effective_eval_node_counts;
  bool auto_dataset_weighting_applied = false;
  std::string effective_selector_profile = "auto";
  uint32_t target_skeleton_contexts_total = 0;
  uint32_t target_skeleton_contexts_requested = 0;
  uint32_t target_skeleton_contexts_unique = 0;
  uint32_t target_skeleton_contexts_deduplicated = 0;
  uint32_t target_skeleton_contexts_selected = 0;
  uint32_t max_model_trains = 0;
  uint32_t model_trains_started = 0;
  uint32_t train_cache_hits = 0;
  uint32_t train_cache_misses = 0;
  double train_cache_wait_ms = 0.0;
  double training_lane_wait_ms = 0.0;
  uint32_t training_base_cache_hits = 0;
  uint32_t training_base_cache_misses = 0;
  uint32_t training_order_cache_hits = 0;
  uint32_t training_order_cache_misses = 0;
  uint32_t training_cdf_cache_hits = 0;
  uint32_t training_cdf_cache_misses = 0;
  uint32_t training_cdf_cache_failures = 0;
  double training_base_cache_wait_ms = 0.0;
  double training_order_cache_wait_ms = 0.0;
  double training_cdf_cache_wait_ms = 0.0;
  double training_base_cache_build_ms = 0.0;
  double training_order_cache_build_ms = 0.0;
  double training_cdf_cache_build_ms = 0.0;
  uint64_t training_base_cache_memory_bytes = 0;
  uint64_t training_order_cache_memory_bytes = 0;
  uint64_t training_cdf_cache_memory_bytes = 0;
  uint32_t candidate_error_count = 0;
  uint32_t candidate_error_launch_guard_count = 0;
  uint32_t candidate_error_training_base_count = 0;
  uint32_t candidate_error_centroid_order_count = 0;
  uint32_t candidate_error_cdf_fit_count = 0;
  uint32_t candidate_error_eval_count = 0;
  uint32_t candidate_error_unknown_count = 0;
  double train_time_ms_sum = 0.0;
  double train_gather_skeleton_ms_sum = 0.0;
  double train_cluster_assign_ms_sum = 0.0;
  double train_assign_centroids_ms_sum = 0.0;
  double train_centroid_order_ms_sum = 0.0;
  double train_range_alloc_ms_sum = 0.0;
  double train_cdf_fit_ms_sum = 0.0;
  double train_model_assembly_ms_sum = 0.0;
  uint32_t scheduler_ready_cached_candidates = 0;
  uint32_t scheduler_in_flight_candidates = 0;
  uint32_t scheduler_missing_candidates = 0;
  uint32_t scheduler_single_admission_batches = 0;
  uint32_t scheduler_reuse_prioritized_candidates = 0;
  uint32_t scheduler_fresh_base_candidates = 0;
  uint32_t eval_nearest_cache_hits = 0;
  uint32_t eval_nearest_cache_misses = 0;
  uint32_t eval_nearest_cache_inflight_bypasses = 0;
  uint32_t eval_nearest_cache_prewarm_requests = 0;
  uint32_t eval_nearest_cache_prewarm_ready_models = 0;
  uint32_t eval_nearest_cache_prewarm_skipped = 0;
  uint32_t eval_nearest_cache_prewarm_failures = 0;
  uint32_t eval_nearest_cache_prewarm_threads = 0;
  double eval_nearest_cache_prewarm_ms = 0.0;
  double eval_nearest_cache_wait_ms = 0.0;
  double eval_nearest_cache_sampled_wait_ms = 0.0;
  double eval_nearest_cache_calibration_wait_ms = 0.0;
  double eval_nearest_cache_build_ms = 0.0;
  double eval_nearest_cache_sampled_build_ms = 0.0;
  double eval_nearest_cache_calibration_build_ms = 0.0;
  uint64_t eval_nearest_cache_distance_terms = 0;
  uint64_t eval_nearest_cache_skipped_distance_terms = 0;
  uint32_t eval_nearest_cache_early_abandon_builds = 0;
  uint32_t eval_nearest_cache_entries = 0;
  uint64_t eval_nearest_cache_memory_bytes = 0;
  uint64_t eval_nearest_cache_sampled_budget_bytes = 0;
  uint64_t eval_nearest_cache_sampled_memory_bytes = 0;
  uint64_t eval_nearest_cache_calibration_memory_bytes = 0;
  uint32_t eval_nearest_cache_evictions = 0;
  uint64_t eval_nearest_cache_evicted_bytes = 0;
  uint32_t eval_base_rank_cache_hits = 0;
  uint32_t eval_base_rank_cache_misses = 0;
  double eval_base_rank_cache_wait_ms = 0.0;
  double eval_base_rank_cache_build_ms = 0.0;
  double eval_base_rank_cache_nearest_ms = 0.0;
  double eval_base_rank_cache_hash_ms = 0.0;
  double eval_base_rank_cache_sort_ms = 0.0;
  double eval_base_rank_cache_rank_index_ms = 0.0;
  double eval_base_rank_cache_overhead_ms = 0.0;
  uint32_t eval_base_rank_cache_entries = 0;
  uint64_t eval_base_rank_cache_memory_bytes = 0;
  uint64_t eval_base_rank_cache_sampled_budget_bytes = 0;
  uint64_t eval_base_rank_cache_sampled_memory_bytes = 0;
  uint64_t eval_base_rank_cache_calibration_memory_bytes = 0;
  uint32_t eval_base_rank_cache_evictions = 0;
  uint64_t eval_base_rank_cache_evicted_bytes = 0;
  std::vector<SelectorBaseRankCacheAttribution> eval_base_rank_cache_attribution;
  bool model_train_budget_hit = false;
  uint32_t target_skeleton_contexts_evaluated = 0;
  uint32_t contexts_skipped_low_potential = 0;
  uint32_t effective_max_search_seconds = 0;
  double elapsed_search_seconds = 0.0;
  bool search_time_budget_hit = false;
  bool remaining_time_guard_hit = false;
  double final_calibration_reserve_seconds = 0.0;
  bool final_calibration_reserve_hit = false;
  std::string stopped_reason;
  uint32_t plateau_rounds = 0;
  double best_quality = -1.0;
  double best_margin = 0.0;
  bool recall_target_met = false;
  uint32_t recall_target_met_recommended = 0;
  uint32_t strategy_refine_evaluated = 0;
  uint32_t strategy_refine_rounds = 0;
  uint32_t strategy_frontier_candidates = 0;
  uint32_t strategy_compacted_candidates = 0;
  uint32_t recall_refine_evaluated = 0;
  uint32_t recall_refine_rounds = 0;
  uint32_t post_exploration_evaluated = 0;
  uint32_t post_exploration_rounds = 0;
  uint32_t final_calibration_seed_count = 0;
  uint32_t effective_final_calibration_full_count = 0;
  uint32_t effective_final_calibration_screen_candidate_count = 0;
  uint32_t effective_final_calibration_screen_query_limit = 0;
  uint32_t final_calibration_screen_evaluated = 0;
  uint32_t final_calibration_screen_promoted = 0;
  uint32_t final_calibration_role_promoted = 0;
  uint32_t final_calibration_recall_promoted = 0;
  std::string final_calibration_screen_status;
  uint32_t final_calibration_evaluated = 0;
  uint32_t final_calibration_reused = 0;
  std::vector<SelectorCalibrationProbe> final_calibration_progressive_probes;
  uint64_t effective_selector_memory_budget_bytes = 0;
  uint32_t selector_peak_parallelism = 1;
  uint32_t selector_peak_threads_per_candidate = 1;
  bool selector_memory_budget_limited = false;
  std::vector<SelectorPhaseTiming> timings;
  std::vector<SelectorContextStats> context_stats;
};

std::vector<SelectorCandidateConfig> build_phase1_selector_candidates(const SelectorOptions& options);

SelectorResult select_vortex_models(const SelectorOptions& options);

rm_model::json::Value selector_result_to_json(const SelectorResult& result,
                                              const SelectorOptions& options);
rm_model::json::Value selector_result_to_json(const SelectorResult& result,
                                              const SelectorOptions& options,
                                              const SelectorJsonOptions& json_options);

} // namespace vortex

#endif // VORTEX_V1_MODEL_SELECTOR_H
