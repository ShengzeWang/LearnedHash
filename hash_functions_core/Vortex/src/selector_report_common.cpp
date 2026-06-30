#include "selector_report_internal.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace vortex::selector_internal {
namespace {

struct GraphCentroidOrderEvidenceRow {
  uint64_t candidates = 0;
  uint64_t ok = 0;
  bool has_ok = false;
  double best_locality_quality_score = 0.0;
  double best_overlay_match_score = 0.0;
  double best_recall_at_k_in_window = 0.0;
  double quality_selected_overlay_match_score = 0.0;
  double quality_selected_recall_at_k_in_window = 0.0;
  uint64_t quality_selected_model_size_bytes = 0;
};

void update_graph_centroid_order_evidence(
    GraphCentroidOrderEvidenceRow& row,
    const SelectorCandidateMetrics& metrics) {
  row.candidates += 1;
  if (!metrics.ok) {
    return;
  }
  row.ok += 1;
  row.best_overlay_match_score =
      std::max(row.best_overlay_match_score, metrics.overlay_match_score);
  row.best_recall_at_k_in_window =
      std::max(row.best_recall_at_k_in_window, metrics.recall_at_k_in_window);
  if (!row.has_ok ||
      metrics.locality_quality_score > row.best_locality_quality_score) {
    row.has_ok = true;
    row.best_locality_quality_score = metrics.locality_quality_score;
    row.quality_selected_overlay_match_score = metrics.overlay_match_score;
    row.quality_selected_recall_at_k_in_window = metrics.recall_at_k_in_window;
    row.quality_selected_model_size_bytes = metrics.model_size_bytes;
  }
}

rm_model::json::Value graph_centroid_order_evidence_row_to_json(
    const GraphCentroidOrderEvidenceRow& row) {
  rm_model::json::Value::Object obj;
  obj.emplace_back("candidates", rm_model::json::Value(row.candidates));
  obj.emplace_back("ok", rm_model::json::Value(row.ok));
  obj.emplace_back("has_ok", rm_model::json::Value(row.has_ok));
  obj.emplace_back("best_locality_quality_score",
                   rm_model::json::Value(row.best_locality_quality_score));
  obj.emplace_back("best_overlay_match_score",
                   rm_model::json::Value(row.best_overlay_match_score));
  obj.emplace_back("best_recall_at_k_in_window",
                   rm_model::json::Value(row.best_recall_at_k_in_window));
  obj.emplace_back("quality_selected_overlay_match_score",
                   rm_model::json::Value(
                       row.quality_selected_overlay_match_score));
  obj.emplace_back("quality_selected_recall_at_k_in_window",
                   rm_model::json::Value(
                       row.quality_selected_recall_at_k_in_window));
  obj.emplace_back("quality_selected_model_size_bytes",
                   rm_model::json::Value(row.quality_selected_model_size_bytes));
  return rm_model::json::Value(std::move(obj));
}


} // namespace

rm_model::json::Value graph_centroid_order_evidence_to_json(
    const SelectorResult& result) {
  GraphCentroidOrderEvidenceRow enabled;
  GraphCentroidOrderEvidenceRow disabled;
  for (const auto& item : result.candidates) {
    update_graph_centroid_order_evidence(
        item.config.enable_graph_centroid_order ? enabled : disabled,
        item);
  }
  rm_model::json::Value::Object obj;
  obj.emplace_back("enabled", graph_centroid_order_evidence_row_to_json(enabled));
  obj.emplace_back("disabled", graph_centroid_order_evidence_row_to_json(disabled));
  return rm_model::json::Value(std::move(obj));
}

rm_model::json::Value uint32_array_to_json(const std::vector<uint32_t>& values) {
  rm_model::json::Value::Array out;
  out.reserve(values.size());
  for (uint32_t value : values) {
    out.push_back(rm_model::json::Value(static_cast<uint64_t>(value)));
  }
  return rm_model::json::Value(std::move(out));
}

rm_model::json::Value uint64_array_to_json(const std::vector<uint64_t>& values) {
  rm_model::json::Value::Array out;
  out.reserve(values.size());
  for (uint64_t value : values) {
    out.push_back(rm_model::json::Value(value));
  }
  return rm_model::json::Value(std::move(out));
}

rm_model::json::Value string_array_to_json(const std::vector<std::string>& values) {
  rm_model::json::Value::Array out;
  out.reserve(values.size());
  for (const auto& value : values) {
    out.push_back(rm_model::json::Value(value));
  }
  return rm_model::json::Value(std::move(out));
}

rm_model::json::Value base_rank_cache_attribution_to_json(
    const SelectorBaseRankCacheAttribution& row) {
  rm_model::json::Value::Object obj;
  obj.emplace_back("phase", rm_model::json::Value(row.phase));
  obj.emplace_back("nsw", rm_model::json::Value(row.nsw_path));
  obj.emplace_back("target_skeleton", rm_model::json::Value(row.target_skeleton));
  obj.emplace_back("skeleton_node_count",
                   rm_model::json::Value(static_cast<uint64_t>(row.skeleton_node_count)));
  obj.emplace_back("skeleton_layer",
                   rm_model::json::Value(static_cast<int64_t>(row.skeleton_layer)));
  obj.emplace_back("K", rm_model::json::Value(static_cast<uint64_t>(row.K)));
  obj.emplace_back("centroid_knn",
                   rm_model::json::Value(static_cast<uint64_t>(row.centroid_knn)));
  obj.emplace_back("cdf_models", rm_model::json::Value(row.cdf_model_spec));
  obj.emplace_back("cdf_branch",
                   rm_model::json::Value(row.cdf_branching_factor));
  obj.emplace_back("enable_2opt", rm_model::json::Value(row.enable_2opt));
  obj.emplace_back("two_opt_iters",
                   rm_model::json::Value(static_cast<uint64_t>(row.two_opt_iterations)));
  obj.emplace_back("graph_centroid_order",
                   rm_model::json::Value(row.enable_graph_centroid_order));
  obj.emplace_back("base_count", rm_model::json::Value(row.base_count));
  obj.emplace_back("query_count", rm_model::json::Value(row.query_count));
  obj.emplace_back("calibration_tier", rm_model::json::Value(row.calibration_tier));
  obj.emplace_back("roles", string_array_to_json(row.roles));
  obj.emplace_back("hits", rm_model::json::Value(static_cast<uint64_t>(row.hits)));
  obj.emplace_back("misses", rm_model::json::Value(static_cast<uint64_t>(row.misses)));
  obj.emplace_back("builds", rm_model::json::Value(static_cast<uint64_t>(row.builds)));
  obj.emplace_back("wait_ms", rm_model::json::Value(row.wait_ms));
  obj.emplace_back("build_ms", rm_model::json::Value(row.build_ms));
  obj.emplace_back("nearest_ms", rm_model::json::Value(row.nearest_ms));
  obj.emplace_back("hash_ms", rm_model::json::Value(row.hash_ms));
  obj.emplace_back("sort_ms", rm_model::json::Value(row.sort_ms));
  obj.emplace_back("rank_index_ms", rm_model::json::Value(row.rank_index_ms));
  obj.emplace_back("overhead_ms", rm_model::json::Value(row.overhead_ms));
  obj.emplace_back("added_memory_bytes", rm_model::json::Value(row.added_memory_bytes));
  obj.emplace_back("retained_memory_bytes",
                   rm_model::json::Value(row.retained_memory_bytes));
  obj.emplace_back("evictions", rm_model::json::Value(static_cast<uint64_t>(row.evictions)));
  obj.emplace_back("evicted_bytes", rm_model::json::Value(row.evicted_bytes));
  return rm_model::json::Value(std::move(obj));
}
rm_model::json::Value calibration_probe_to_json(const SelectorCalibrationProbe& probe) {
  rm_model::json::Value::Object obj;
  obj.emplace_back("name", rm_model::json::Value(probe.name));
  obj.emplace_back("probe_base_count", rm_model::json::Value(probe.probe_base_count));
  obj.emplace_back("probe_query_count", rm_model::json::Value(probe.probe_query_count));
  obj.emplace_back("final_base_count", rm_model::json::Value(probe.final_base_count));
  obj.emplace_back("final_query_count", rm_model::json::Value(probe.final_query_count));
  obj.emplace_back("evaluated", rm_model::json::Value(static_cast<uint64_t>(probe.evaluated)));
  obj.emplace_back("probe_peak_recall", rm_model::json::Value(probe.probe_peak_recall));
  obj.emplace_back("final_peak_recall", rm_model::json::Value(probe.final_peak_recall));
  obj.emplace_back("peak_recall_match", rm_model::json::Value(probe.peak_recall_match));
  obj.emplace_back("knee_match", rm_model::json::Value(probe.knee_match));
  obj.emplace_back("fast_match", rm_model::json::Value(probe.fast_match));
  obj.emplace_back("small_match", rm_model::json::Value(probe.small_match));
  obj.emplace_back("matched_role_count",
                   rm_model::json::Value(static_cast<uint64_t>(
                       probe.matched_role_count)));
  obj.emplace_back("all_roles_match",
                   rm_model::json::Value(probe.peak_recall_match &&
                                         probe.knee_match &&
                                         probe.fast_match &&
                                         probe.small_match));
  return rm_model::json::Value(std::move(obj));
}

} // namespace vortex::selector_internal
