#include "selector_report_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>

namespace vortex::selector_internal {
namespace {

std::string objective_score_scope_label(const SelectorCandidateMetrics& metrics) {
  if (!metrics.phase.empty()) {
    return metrics.phase;
  }
  return "unknown";
}

rm_model::json::Value recall_curve_to_json(const SelectorCandidateMetrics& metrics) {
  rm_model::json::Value::Array out;
  std::size_t count = std::min(metrics.recall_windows.size(),
                               metrics.recall_at_k_by_window.size());
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    rm_model::json::Value::Object row;
    row.emplace_back("window", rm_model::json::Value(metrics.recall_windows[i]));
    row.emplace_back("recall_at_k", rm_model::json::Value(metrics.recall_at_k_by_window[i]));
    out.push_back(rm_model::json::Value(std::move(row)));
  }
  return rm_model::json::Value(std::move(out));
}

rm_model::json::Value node_locality_curve_to_json(const SelectorCandidateMetrics& metrics) {
  rm_model::json::Value::Array out;
  std::size_t count = metrics.node_counts.size();
  count = std::min(count, metrics.same_node_hit_by_count.size());
  count = std::min(count, metrics.near_1_node_hit_by_count.size());
  count = std::min(count, metrics.near_2_node_hit_by_count.size());
  count = std::min(count, metrics.near_4_node_hit_by_count.size());
  count = std::min(count, metrics.mean_node_distance_norm_by_count.size());
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    rm_model::json::Value::Object row;
    row.emplace_back("nodes",
                     rm_model::json::Value(static_cast<uint64_t>(
                         metrics.node_counts[i])));
    row.emplace_back("same_node_hit_rate",
                     rm_model::json::Value(metrics.same_node_hit_by_count[i]));
    row.emplace_back("near_1_node_hit_rate",
                     rm_model::json::Value(metrics.near_1_node_hit_by_count[i]));
    row.emplace_back("near_2_node_hit_rate",
                     rm_model::json::Value(metrics.near_2_node_hit_by_count[i]));
    row.emplace_back("near_4_node_hit_rate",
                     rm_model::json::Value(metrics.near_4_node_hit_by_count[i]));
    row.emplace_back("mean_node_distance_norm",
                     rm_model::json::Value(
                         metrics.mean_node_distance_norm_by_count[i]));
    out.push_back(rm_model::json::Value(std::move(row)));
  }
  return rm_model::json::Value(std::move(out));
}

rm_model::json::Value overlay_match_curve_to_json(const SelectorCandidateMetrics& metrics) {
  rm_model::json::Value::Array out;
  std::size_t count = std::min(metrics.node_counts.size(),
                               metrics.overlay_match_by_count.size());
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    rm_model::json::Value::Object row;
    row.emplace_back("nodes",
                     rm_model::json::Value(static_cast<uint64_t>(
                         metrics.node_counts[i])));
    row.emplace_back("overlay_match_score",
                     rm_model::json::Value(metrics.overlay_match_by_count[i]));
    out.push_back(rm_model::json::Value(std::move(row)));
  }
  return rm_model::json::Value(std::move(out));
}


} // namespace

rm_model::json::Value selector_config_to_json(const SelectorCandidateConfig& config) {
  rm_model::json::Value::Object obj;
  obj.emplace_back("target_skeleton", rm_model::json::Value(config.target_skeleton));
  obj.emplace_back("K", rm_model::json::Value(static_cast<uint64_t>(config.K)));
  obj.emplace_back("centroid_knn", rm_model::json::Value(static_cast<uint64_t>(config.centroid_knn)));
  obj.emplace_back("cdf_models", rm_model::json::Value(config.cdf_model_spec));
  obj.emplace_back("cdf_branch", rm_model::json::Value(config.cdf_branching_factor));
  obj.emplace_back("enable_2opt", rm_model::json::Value(config.enable_2opt));
  obj.emplace_back("two_opt_iters", rm_model::json::Value(static_cast<uint64_t>(config.two_opt_iterations)));
  obj.emplace_back("graph_centroid_order",
                   rm_model::json::Value(config.enable_graph_centroid_order));
  return rm_model::json::Value(std::move(obj));
}
rm_model::json::Value selector_candidate_to_json(const SelectorCandidateMetrics& metrics,
                                                 const SelectorOptions& options) {
  rm_model::json::Value::Object obj;
  obj.emplace_back("phase", rm_model::json::Value(metrics.phase));
  obj.emplace_back("ok", rm_model::json::Value(metrics.ok));
  obj.emplace_back("config", selector_config_to_json(metrics.config));
  obj.emplace_back("nsw", rm_model::json::Value(metrics.nsw_path));
  obj.emplace_back("skeleton_node_count",
                   rm_model::json::Value(static_cast<uint64_t>(metrics.skeleton_node_count)));
  obj.emplace_back("skeleton_layer",
                   rm_model::json::Value(static_cast<int64_t>(metrics.skeleton_layer)));
  obj.emplace_back("objective_score", rm_model::json::Value(metrics.objective_score));
  obj.emplace_back("objective_score_scope",
                   rm_model::json::Value(objective_score_scope_label(metrics)));
  obj.emplace_back("train_time_ms", rm_model::json::Value(metrics.train_time_ms));
  obj.emplace_back("train_base_cache_hit",
                   rm_model::json::Value(metrics.train_base_cache_hit));
  obj.emplace_back("train_order_cache_hit",
                   rm_model::json::Value(metrics.train_order_cache_hit));
  obj.emplace_back("train_cdf_cache_hit",
                   rm_model::json::Value(metrics.train_cdf_cache_hit));
  obj.emplace_back("train_base_cache_wait_ms",
                   rm_model::json::Value(metrics.train_base_cache_wait_ms));
  obj.emplace_back("train_order_cache_wait_ms",
                   rm_model::json::Value(metrics.train_order_cache_wait_ms));
  obj.emplace_back("train_cdf_cache_wait_ms",
                   rm_model::json::Value(metrics.train_cdf_cache_wait_ms));
  obj.emplace_back("train_base_build_ms",
                   rm_model::json::Value(metrics.train_base_build_ms));
  obj.emplace_back("train_order_build_ms",
                   rm_model::json::Value(metrics.train_order_build_ms));
  obj.emplace_back("train_cdf_cache_build_ms",
                   rm_model::json::Value(metrics.train_cdf_cache_build_ms));
  obj.emplace_back("train_gather_skeleton_ms",
                   rm_model::json::Value(metrics.train_gather_skeleton_ms));
  obj.emplace_back("train_cluster_assign_ms",
                   rm_model::json::Value(metrics.train_cluster_assign_ms));
  obj.emplace_back("train_assign_centroids_ms",
                   rm_model::json::Value(metrics.train_assign_centroids_ms));
  obj.emplace_back("train_centroid_order_ms",
                   rm_model::json::Value(metrics.train_centroid_order_ms));
  obj.emplace_back("train_range_alloc_ms",
                   rm_model::json::Value(metrics.train_range_alloc_ms));
  obj.emplace_back("train_cdf_fit_ms",
                   rm_model::json::Value(metrics.train_cdf_fit_ms));
  obj.emplace_back("train_model_assembly_ms",
                   rm_model::json::Value(metrics.train_model_assembly_ms));
  obj.emplace_back("model_size_bytes", rm_model::json::Value(metrics.model_size_bytes));
  obj.emplace_back("model_memory_bytes", rm_model::json::Value(metrics.model_memory_bytes));
  obj.emplace_back("centroid_count", rm_model::json::Value(static_cast<uint64_t>(metrics.centroid_count)));
  obj.emplace_back("active_centroid_count",
                   rm_model::json::Value(static_cast<uint64_t>(metrics.active_centroid_count)));
  obj.emplace_back("recall_at_k_in_window", rm_model::json::Value(metrics.recall_at_k_in_window));
  obj.emplace_back("recall_curve", recall_curve_to_json(metrics));
  obj.emplace_back("recall_auc_log_window",
                   rm_model::json::Value(metrics.recall_auc_log_window));
  obj.emplace_back("min_window_for_recall_target",
                   metrics.min_window_for_recall_target == 0
                       ? rm_model::json::Value()
                       : rm_model::json::Value(metrics.min_window_for_recall_target));
  obj.emplace_back("recall_target_met_by_curve",
                   rm_model::json::Value(metrics.recall_target_met_by_curve));
  obj.emplace_back("query_recall_p05", rm_model::json::Value(metrics.query_recall_p05));
  obj.emplace_back("query_recall_p50", rm_model::json::Value(metrics.query_recall_p50));
  obj.emplace_back("query_recall_p95", rm_model::json::Value(metrics.query_recall_p95));
  obj.emplace_back("mean_rank_distance_norm", rm_model::json::Value(metrics.mean_rank_distance_norm));
  obj.emplace_back("rank_distance_norm_p50",
                   rm_model::json::Value(metrics.rank_distance_norm_p50));
  obj.emplace_back("rank_distance_norm_p95",
                   rm_model::json::Value(metrics.rank_distance_norm_p95));
  obj.emplace_back("rank_distance_norm_p99",
                   rm_model::json::Value(metrics.rank_distance_norm_p99));
  obj.emplace_back("node_locality_score",
                   rm_model::json::Value(metrics.node_locality_score));
  obj.emplace_back("node_locality_curve", node_locality_curve_to_json(metrics));
  obj.emplace_back("overlay_match_score",
                   rm_model::json::Value(metrics.overlay_match_score));
  obj.emplace_back("overlay_match_curve", overlay_match_curve_to_json(metrics));
  obj.emplace_back("query_overlay_match_p05",
                   rm_model::json::Value(metrics.query_overlay_match_p05));
  obj.emplace_back("query_overlay_match_p50",
                   rm_model::json::Value(metrics.query_overlay_match_p50));
  obj.emplace_back("query_overlay_match_p95",
                   rm_model::json::Value(metrics.query_overlay_match_p95));
  obj.emplace_back("locality_quality_score",
                   rm_model::json::Value(metrics.locality_quality_score));
  obj.emplace_back("latency_avg_ms", rm_model::json::Value(metrics.latency_avg_ms));
  obj.emplace_back("latency_p95_ms", rm_model::json::Value(metrics.latency_p95_ms));
  obj.emplace_back("latency_p99_ms", rm_model::json::Value(metrics.latency_p99_ms));
  obj.emplace_back("eval_total_ms", rm_model::json::Value(metrics.eval_total_ms));
  obj.emplace_back("eval_hash_base_ms", rm_model::json::Value(metrics.eval_hash_base_ms));
  obj.emplace_back("eval_sort_base_ms", rm_model::json::Value(metrics.eval_sort_base_ms));
  obj.emplace_back("eval_rank_index_ms", rm_model::json::Value(metrics.eval_rank_index_ms));
  obj.emplace_back("eval_query_ms", rm_model::json::Value(metrics.eval_query_ms));
  obj.emplace_back("eval_latency_ms", rm_model::json::Value(metrics.eval_latency_ms));
  obj.emplace_back("eval_hash_base_threads",
                   rm_model::json::Value(
                       static_cast<uint64_t>(metrics.eval_hash_base_threads)));
  obj.emplace_back("eval_rank_index_threads",
                   rm_model::json::Value(
                       static_cast<uint64_t>(metrics.eval_rank_index_threads)));
  if (!metrics.error.empty()) {
    obj.emplace_back("error", rm_model::json::Value(metrics.error));
    obj.emplace_back("error_stage", rm_model::json::Value(metrics.error_stage));
  }

  std::string nsw_for_train = metrics.nsw_path.empty() ? options.nsw_path.string() : metrics.nsw_path;
  std::string cmd;
  if (metrics.config.target_skeleton > 0 && options.nsw_index_path.has_value()) {
    std::string extracted_nsw = metrics.nsw_path;
    if (extracted_nsw.empty()) {
      std::string stem = options.dataset_path.stem().string();
      if (stem.empty()) {
        stem = "dataset";
      }
      extracted_nsw =
          (std::filesystem::path("vortex_v1_output") /
           (stem + "_nsw_ts" + std::to_string(metrics.config.target_skeleton) + ".csr"))
              .string();
    }
    cmd = "vortex_v1_cli extract_nsw --index \"" + options.nsw_index_path->string() +
          "\" --target_skeleton " + std::to_string(metrics.config.target_skeleton);
    if (options.nsw_layer.has_value()) {
      cmd += " --layer " + std::to_string(*options.nsw_layer);
    }
    cmd += " --output \"" + extracted_nsw + "\" && ";
    nsw_for_train = extracted_nsw;
  }
  cmd += "vortex_v1_cli train --dataset \"" + options.dataset_path.string() +
         "\" --nsw \"" + nsw_for_train + "\"" +
         " --K " + std::to_string(metrics.config.K) +
         " --hash_bits " + std::to_string(options.hash_bits) +
         " --cdf_models " + metrics.config.cdf_model_spec +
         " --cdf_branch " + std::to_string(metrics.config.cdf_branching_factor) +
         " --centroid_knn " + std::to_string(metrics.config.centroid_knn) +
         (metrics.config.enable_2opt ? " --enable_2opt " : " --disable_2opt ") +
         "--two_opt_iters " + std::to_string(metrics.config.two_opt_iterations) +
         " --seed " + std::to_string(options.seed) +
         " --threads " + std::to_string(options.threads);
  if (!options.use_full_dataset_for_assignments) {
    cmd += " --disable-full-dataset-assignments";
  }
  if (options.assignment_sample_limit > 0) {
    cmd += " --assignment-sample-limit " + std::to_string(options.assignment_sample_limit);
  }
  obj.emplace_back("train_command", rm_model::json::Value(cmd));

  return rm_model::json::Value(std::move(obj));
}

} // namespace vortex::selector_internal
