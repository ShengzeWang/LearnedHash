#include "selector_report_internal.h"

#include <cstdint>
#include <utility>

namespace vortex::selector_internal {
namespace {

rm_model::json::Value cache_summary_to_json(uint32_t hits,
                                            uint32_t misses,
                                            double wait_ms,
                                            double build_ms,
                                            uint64_t memory_bytes) {
  rm_model::json::Value::Object obj;
  obj.emplace_back("hits", rm_model::json::Value(static_cast<uint64_t>(hits)));
  obj.emplace_back("misses", rm_model::json::Value(static_cast<uint64_t>(misses)));
  obj.emplace_back("wait_ms", rm_model::json::Value(wait_ms));
  obj.emplace_back("build_ms", rm_model::json::Value(build_ms));
  obj.emplace_back("memory_bytes", rm_model::json::Value(memory_bytes));
  return rm_model::json::Value(std::move(obj));
}

} // namespace

rm_model::json::Value selector_diagnostics_to_json(const SelectorResult& result,
                                                   const SelectorOptions& options) {
  rm_model::json::Value::Object diagnostics;
  diagnostics.emplace_back("schema_version",
                           rm_model::json::Value(static_cast<uint64_t>(1)));

  rm_model::json::Value::Object contexts;
  contexts.emplace_back("total",
                        rm_model::json::Value(
                            static_cast<uint64_t>(result.target_skeleton_contexts_total)));
  contexts.emplace_back("requested",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.target_skeleton_contexts_requested)));
  contexts.emplace_back("unique",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.target_skeleton_contexts_unique)));
  contexts.emplace_back("deduplicated",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.target_skeleton_contexts_deduplicated)));
  contexts.emplace_back("selected",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.target_skeleton_contexts_selected)));
  contexts.emplace_back("evaluated",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.target_skeleton_contexts_evaluated)));
  contexts.emplace_back("max",
                        rm_model::json::Value(static_cast<uint64_t>(
                            options.max_target_skeleton_contexts)));
  contexts.emplace_back("skipped_low_potential",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.contexts_skipped_low_potential)));

  rm_model::json::Value::Object search;
  search.emplace_back("selector_profile", rm_model::json::Value(options.selector_profile));
  search.emplace_back("effective_selector_profile",
                      rm_model::json::Value(result.effective_selector_profile));
  search.emplace_back("max_search_seconds",
                      rm_model::json::Value(
                          static_cast<uint64_t>(options.max_search_seconds)));
  search.emplace_back("effective_max_search_seconds",
                      rm_model::json::Value(static_cast<uint64_t>(
                          result.effective_max_search_seconds)));
  search.emplace_back("elapsed_search_seconds",
                      rm_model::json::Value(result.elapsed_search_seconds));
  search.emplace_back("stopped_reason",
                      rm_model::json::Value(result.stopped_reason));
  search.emplace_back("model_train_budget_hit",
                      rm_model::json::Value(result.model_train_budget_hit));
  search.emplace_back("search_time_budget_hit",
                      rm_model::json::Value(result.search_time_budget_hit));
  search.emplace_back("remaining_time_guard_hit",
                      rm_model::json::Value(result.remaining_time_guard_hit));
  search.emplace_back("final_calibration_reserve_seconds",
                      rm_model::json::Value(result.final_calibration_reserve_seconds));
  search.emplace_back("final_calibration_reserve_hit",
                      rm_model::json::Value(result.final_calibration_reserve_hit));
  search.emplace_back("plateau_rounds",
                      rm_model::json::Value(static_cast<uint64_t>(result.plateau_rounds)));
  search.emplace_back("best_quality", rm_model::json::Value(result.best_quality));
  search.emplace_back("best_margin", rm_model::json::Value(result.best_margin));
  if (options.recall_target.has_value()) {
    search.emplace_back("recall_target", rm_model::json::Value(*options.recall_target));
    search.emplace_back("recall_target_met",
                        rm_model::json::Value(result.recall_target_met));
    search.emplace_back("recall_target_met_recommended",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.recall_target_met_recommended)));
  }
  search.emplace_back("contexts", rm_model::json::Value(std::move(contexts)));
  diagnostics.emplace_back("search", rm_model::json::Value(std::move(search)));

  rm_model::json::Value::Object training;
  training.emplace_back("model_trains_started",
                        rm_model::json::Value(
                            static_cast<uint64_t>(result.model_trains_started)));
  training.emplace_back("max_model_trains",
                        rm_model::json::Value(
                            static_cast<uint64_t>(result.max_model_trains)));
  training.emplace_back("lane_wait_ms",
                        rm_model::json::Value(result.training_lane_wait_ms));
  training.emplace_back("model_cache",
                        cache_summary_to_json(result.train_cache_hits,
                                              result.train_cache_misses,
                                              result.train_cache_wait_ms,
                                              0.0,
                                              0));
  training.emplace_back("base_cache",
                        cache_summary_to_json(result.training_base_cache_hits,
                                              result.training_base_cache_misses,
                                              result.training_base_cache_wait_ms,
                                              result.training_base_cache_build_ms,
                                              result.training_base_cache_memory_bytes));
  training.emplace_back("order_cache",
                        cache_summary_to_json(result.training_order_cache_hits,
                                              result.training_order_cache_misses,
                                              result.training_order_cache_wait_ms,
                                              result.training_order_cache_build_ms,
                                              result.training_order_cache_memory_bytes));
  training.emplace_back("cdf_cache",
                        cache_summary_to_json(result.training_cdf_cache_hits,
                                              result.training_cdf_cache_misses,
                                              result.training_cdf_cache_wait_ms,
                                              result.training_cdf_cache_build_ms,
                                              result.training_cdf_cache_memory_bytes));
  training.emplace_back("cdf_cache_failures",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.training_cdf_cache_failures)));
  training.emplace_back("stage_ms_sum",
                        rm_model::json::Value(rm_model::json::Value::Object{
                            {"total", rm_model::json::Value(result.train_time_ms_sum)},
                            {"gather_skeleton",
                             rm_model::json::Value(result.train_gather_skeleton_ms_sum)},
                            {"cluster_assign",
                             rm_model::json::Value(result.train_cluster_assign_ms_sum)},
                            {"assign_centroids",
                             rm_model::json::Value(result.train_assign_centroids_ms_sum)},
                            {"centroid_order",
                             rm_model::json::Value(result.train_centroid_order_ms_sum)},
                            {"range_alloc",
                             rm_model::json::Value(result.train_range_alloc_ms_sum)},
                            {"cdf_fit", rm_model::json::Value(result.train_cdf_fit_ms_sum)},
                            {"model_assembly",
                             rm_model::json::Value(result.train_model_assembly_ms_sum)},
                        }));
  diagnostics.emplace_back("training", rm_model::json::Value(std::move(training)));

  rm_model::json::Value::Object scheduler;
  scheduler.emplace_back("ready_cached_candidates",
                         rm_model::json::Value(static_cast<uint64_t>(
                             result.scheduler_ready_cached_candidates)));
  scheduler.emplace_back("in_flight_candidates",
                         rm_model::json::Value(static_cast<uint64_t>(
                             result.scheduler_in_flight_candidates)));
  scheduler.emplace_back("missing_candidates",
                         rm_model::json::Value(static_cast<uint64_t>(
                             result.scheduler_missing_candidates)));
  scheduler.emplace_back("single_admission_batches",
                         rm_model::json::Value(static_cast<uint64_t>(
                             result.scheduler_single_admission_batches)));
  scheduler.emplace_back("reuse_prioritized_candidates",
                         rm_model::json::Value(static_cast<uint64_t>(
                             result.scheduler_reuse_prioritized_candidates)));
  scheduler.emplace_back("fresh_base_candidates",
                         rm_model::json::Value(static_cast<uint64_t>(
                             result.scheduler_fresh_base_candidates)));
  diagnostics.emplace_back("scheduler", rm_model::json::Value(std::move(scheduler)));

  rm_model::json::Value::Object nearest_cache;
  nearest_cache.emplace_back("hits",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_hits)));
  nearest_cache.emplace_back("misses",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_misses)));
  nearest_cache.emplace_back("inflight_bypasses",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_inflight_bypasses)));
  nearest_cache.emplace_back("prewarm_requests",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_prewarm_requests)));
  nearest_cache.emplace_back("prewarm_ready_models",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_prewarm_ready_models)));
  nearest_cache.emplace_back("prewarm_skipped",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_prewarm_skipped)));
  nearest_cache.emplace_back("prewarm_failures",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_prewarm_failures)));
  nearest_cache.emplace_back("prewarm_threads",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_prewarm_threads)));
  nearest_cache.emplace_back("prewarm_ms",
                             rm_model::json::Value(
                                 result.eval_nearest_cache_prewarm_ms));
  nearest_cache.emplace_back("wait_ms",
                             rm_model::json::Value(result.eval_nearest_cache_wait_ms));
  nearest_cache.emplace_back(
      "sampled_wait_ms",
      rm_model::json::Value(result.eval_nearest_cache_sampled_wait_ms));
  nearest_cache.emplace_back(
      "calibration_wait_ms",
      rm_model::json::Value(result.eval_nearest_cache_calibration_wait_ms));
  nearest_cache.emplace_back("build_ms",
                             rm_model::json::Value(result.eval_nearest_cache_build_ms));
  nearest_cache.emplace_back(
      "sampled_build_ms",
      rm_model::json::Value(result.eval_nearest_cache_sampled_build_ms));
  nearest_cache.emplace_back(
      "calibration_build_ms",
      rm_model::json::Value(result.eval_nearest_cache_calibration_build_ms));
  nearest_cache.emplace_back("entries",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_entries)));
  nearest_cache.emplace_back("memory_bytes",
                             rm_model::json::Value(
                                 result.eval_nearest_cache_memory_bytes));
  nearest_cache.emplace_back("sampled_budget_bytes",
                             rm_model::json::Value(
                                 result.eval_nearest_cache_sampled_budget_bytes));
  nearest_cache.emplace_back("sampled_memory_bytes",
                             rm_model::json::Value(
                                 result.eval_nearest_cache_sampled_memory_bytes));
  nearest_cache.emplace_back("calibration_memory_bytes",
                             rm_model::json::Value(
                                 result.eval_nearest_cache_calibration_memory_bytes));
  nearest_cache.emplace_back("evictions",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_evictions)));
  nearest_cache.emplace_back("evicted_bytes",
                             rm_model::json::Value(result.eval_nearest_cache_evicted_bytes));
  nearest_cache.emplace_back("distance_terms",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_distance_terms)));
  nearest_cache.emplace_back("skipped_distance_terms",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_skipped_distance_terms)));
  nearest_cache.emplace_back("early_abandon_builds",
                             rm_model::json::Value(static_cast<uint64_t>(
                                 result.eval_nearest_cache_early_abandon_builds)));

  rm_model::json::Value::Object base_rank_cache;
  base_rank_cache.emplace_back("hits",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.eval_base_rank_cache_hits)));
  base_rank_cache.emplace_back("misses",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.eval_base_rank_cache_misses)));
  base_rank_cache.emplace_back("wait_ms",
                               rm_model::json::Value(result.eval_base_rank_cache_wait_ms));
  base_rank_cache.emplace_back("build_ms",
                               rm_model::json::Value(result.eval_base_rank_cache_build_ms));
  base_rank_cache.emplace_back("nearest_ms",
                               rm_model::json::Value(result.eval_base_rank_cache_nearest_ms));
  base_rank_cache.emplace_back("hash_ms",
                               rm_model::json::Value(result.eval_base_rank_cache_hash_ms));
  base_rank_cache.emplace_back("sort_ms",
                               rm_model::json::Value(result.eval_base_rank_cache_sort_ms));
  base_rank_cache.emplace_back("rank_index_ms",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_rank_index_ms));
  base_rank_cache.emplace_back("overhead_ms",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_overhead_ms));
  base_rank_cache.emplace_back("entries",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.eval_base_rank_cache_entries)));
  base_rank_cache.emplace_back("memory_bytes",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_memory_bytes));
  base_rank_cache.emplace_back("sampled_budget_bytes",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_sampled_budget_bytes));
  base_rank_cache.emplace_back("sampled_memory_bytes",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_sampled_memory_bytes));
  base_rank_cache.emplace_back("calibration_memory_bytes",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_calibration_memory_bytes));
  base_rank_cache.emplace_back("evictions",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.eval_base_rank_cache_evictions)));
  base_rank_cache.emplace_back("evicted_bytes",
                               rm_model::json::Value(
                                   result.eval_base_rank_cache_evicted_bytes));
  base_rank_cache.emplace_back("attribution_count",
                               rm_model::json::Value(static_cast<uint64_t>(
                                   result.eval_base_rank_cache_attribution.size())));

  rm_model::json::Value::Object evaluation;
  evaluation.emplace_back("nearest_cache", rm_model::json::Value(std::move(nearest_cache)));
  evaluation.emplace_back("base_rank_cache",
                          rm_model::json::Value(std::move(base_rank_cache)));
  diagnostics.emplace_back("evaluation", rm_model::json::Value(std::move(evaluation)));

  rm_model::json::Value::Object strategy;
  strategy.emplace_back("advanced_search", rm_model::json::Value(options.advanced_search));
  strategy.emplace_back("optimize_for_recall",
                        rm_model::json::Value(options.optimize_for_recall));
  strategy.emplace_back("primary_metric",
                        rm_model::json::Value("locality_quality_score"));
  strategy.emplace_back("deployment_metric",
                        rm_model::json::Value("overlay_match_score"));
  strategy.emplace_back("compatibility_locality_metric",
                        rm_model::json::Value("node_locality_score"));
  strategy.emplace_back("quality_score_formula",
                        rm_model::json::Value(
                            "0.50*overlay_match_score+0.20*node_locality_score+"
                            "0.15*recall_auc_log_window+0.10*query_overlay_match_p05+"
                            "0.05*(1-rank_distance_norm_p50)"));
  strategy.emplace_back("attribution_guided_compaction",
                        rm_model::json::Value(options.attribution_guided_compaction));
  strategy.emplace_back("skeleton_capacity_scaling",
                        rm_model::json::Value(options.skeleton_capacity_scaling));
  strategy.emplace_back("capacity_policy",
                        rm_model::json::Value(
                            "scale_aware_recall_first_config_families"));
  strategy.emplace_back("centroid_order_policy",
                        rm_model::json::Value(
                            options.enable_graph_centroid_order
                                ? "euclidean_mst_2opt_plus_nsw_graph_affinity"
                                : "euclidean_mst_2opt"));
  strategy.emplace_back("beam_width",
                        rm_model::json::Value(static_cast<uint64_t>(options.beam_width)));
  strategy.emplace_back("beam_rounds",
                        rm_model::json::Value(static_cast<uint64_t>(options.beam_rounds)));
  strategy.emplace_back("hill_climb_steps",
                        rm_model::json::Value(
                            static_cast<uint64_t>(options.hill_climb_steps)));
  strategy.emplace_back("beam_neighbor_limit",
                        rm_model::json::Value(
                            static_cast<uint64_t>(options.beam_neighbor_limit)));
  strategy.emplace_back("strategy_refine_evaluated",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.strategy_refine_evaluated)));
  strategy.emplace_back("strategy_refine_rounds",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.strategy_refine_rounds)));
  strategy.emplace_back("frontier_candidates",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.strategy_frontier_candidates)));
  strategy.emplace_back("compacted_candidates",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.strategy_compacted_candidates)));
  strategy.emplace_back("recall_refine_evaluated",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.recall_refine_evaluated)));
  strategy.emplace_back("recall_refine_rounds",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.recall_refine_rounds)));
  strategy.emplace_back("post_exploration_exploitation",
                        rm_model::json::Value(options.post_exploration_exploitation));
  strategy.emplace_back("post_exploration_candidates",
                        rm_model::json::Value(static_cast<uint64_t>(
                            options.post_exploration_candidates)));
  strategy.emplace_back("post_exploration_evaluated",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.post_exploration_evaluated)));
  strategy.emplace_back("post_exploration_rounds",
                        rm_model::json::Value(static_cast<uint64_t>(
                            result.post_exploration_rounds)));
  diagnostics.emplace_back("strategy", rm_model::json::Value(std::move(strategy)));

  rm_model::json::Value::Object calibration;
  calibration.emplace_back("count",
                           rm_model::json::Value(static_cast<uint64_t>(
                               options.final_calibration_count)));
  calibration.emplace_back("full_count",
                           rm_model::json::Value(static_cast<uint64_t>(
                               options.final_calibration_full_count)));
  calibration.emplace_back("effective_full_count",
                           rm_model::json::Value(static_cast<uint64_t>(
                               result.effective_final_calibration_full_count)));
  calibration.emplace_back("seed_count",
                           rm_model::json::Value(static_cast<uint64_t>(
                               result.final_calibration_seed_count)));
  calibration.emplace_back("screening",
                           rm_model::json::Value(options.final_calibration_screening));
  calibration.emplace_back("screen_candidate_count",
                           rm_model::json::Value(static_cast<uint64_t>(
                               options.final_calibration_screen_candidate_count)));
  calibration.emplace_back("effective_screen_candidate_count",
                           rm_model::json::Value(static_cast<uint64_t>(
                               result.effective_final_calibration_screen_candidate_count)));
  calibration.emplace_back("screen_query_limit",
                           rm_model::json::Value(static_cast<uint64_t>(
                               options.final_calibration_screen_query_limit)));
  calibration.emplace_back("effective_screen_query_limit",
                           rm_model::json::Value(static_cast<uint64_t>(
                               result.effective_final_calibration_screen_query_limit)));
  calibration.emplace_back("screen_evaluated",
                           rm_model::json::Value(static_cast<uint64_t>(
                               result.final_calibration_screen_evaluated)));
  calibration.emplace_back("screen_promoted",
                           rm_model::json::Value(static_cast<uint64_t>(
                               result.final_calibration_screen_promoted)));
  calibration.emplace_back("role_promoted",
                           rm_model::json::Value(static_cast<uint64_t>(
                               result.final_calibration_role_promoted)));
  calibration.emplace_back("recall_promoted",
                           rm_model::json::Value(static_cast<uint64_t>(
                               result.final_calibration_recall_promoted)));
  calibration.emplace_back("screen_status",
                           rm_model::json::Value(result.final_calibration_screen_status));
  calibration.emplace_back("query_limit",
                           rm_model::json::Value(static_cast<uint64_t>(
                               options.final_calibration_query_limit)));
  calibration.emplace_back("evaluated",
                           rm_model::json::Value(static_cast<uint64_t>(
                               result.final_calibration_evaluated)));
  calibration.emplace_back("reused",
                           rm_model::json::Value(static_cast<uint64_t>(
                               result.final_calibration_reused)));
  calibration.emplace_back("progressive_probe_count",
                           rm_model::json::Value(static_cast<uint64_t>(
                               result.final_calibration_progressive_probes.size())));
  diagnostics.emplace_back("final_calibration",
                           rm_model::json::Value(std::move(calibration)));

  rm_model::json::Value::Object memory;
  memory.emplace_back("selector_memory_budget_bytes",
                      rm_model::json::Value(options.selector_memory_budget_bytes));
  memory.emplace_back("effective_selector_memory_budget_bytes",
                      rm_model::json::Value(result.effective_selector_memory_budget_bytes));
  memory.emplace_back("selector_peak_parallelism",
                      rm_model::json::Value(static_cast<uint64_t>(
                          result.selector_peak_parallelism)));
  memory.emplace_back("selector_peak_threads_per_candidate",
                      rm_model::json::Value(static_cast<uint64_t>(
                          result.selector_peak_threads_per_candidate)));
  memory.emplace_back("selector_memory_budget_limited",
                      rm_model::json::Value(result.selector_memory_budget_limited));
  diagnostics.emplace_back("memory", rm_model::json::Value(std::move(memory)));

  rm_model::json::Value::Object errors;
  errors.emplace_back("total",
                      rm_model::json::Value(static_cast<uint64_t>(
                          result.candidate_error_count)));
  errors.emplace_back("launch_guard",
                      rm_model::json::Value(static_cast<uint64_t>(
                          result.candidate_error_launch_guard_count)));
  errors.emplace_back("training_base",
                      rm_model::json::Value(static_cast<uint64_t>(
                          result.candidate_error_training_base_count)));
  errors.emplace_back("centroid_order",
                      rm_model::json::Value(static_cast<uint64_t>(
                          result.candidate_error_centroid_order_count)));
  errors.emplace_back("cdf_fit",
                      rm_model::json::Value(static_cast<uint64_t>(
                          result.candidate_error_cdf_fit_count)));
  errors.emplace_back("evaluation",
                      rm_model::json::Value(static_cast<uint64_t>(
                          result.candidate_error_eval_count)));
  errors.emplace_back("unknown",
                      rm_model::json::Value(static_cast<uint64_t>(
                          result.candidate_error_unknown_count)));
  diagnostics.emplace_back("errors", rm_model::json::Value(std::move(errors)));

  return rm_model::json::Value(std::move(diagnostics));
}

} // namespace vortex::selector_internal
