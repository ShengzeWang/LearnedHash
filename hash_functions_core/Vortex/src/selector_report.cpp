#include "vortex_v1/model_selector.h"

#include "selector_internal.h"
#include "selector_report_internal.h"

#include <cstdint>
#include <utility>

namespace vortex {

using namespace selector_internal;

rm_model::json::Value selector_result_to_json(const SelectorResult& result,
                                              const SelectorOptions& options) {
  return selector_result_to_json(result, options, SelectorJsonOptions{});
}

rm_model::json::Value selector_result_to_json(const SelectorResult& result,
                                              const SelectorOptions& options,
                                              const SelectorJsonOptions& json_options) {
  SelectorOptions report_options = options;
  if (!report_options.use_full_dataset_for_assignments) {
    report_options.assignment_sample_limit = 0;
  } else if (report_options.auto_assignment_sample_limit &&
             report_options.assignment_sample_limit == 0 &&
             result.dataset_count > 0) {
    report_options.assignment_sample_limit =
        default_assignment_sample_limit(report_options, result.dataset_count);
  }
  rm_model::json::Value::Object root;
  root.emplace_back("selector_json_schema_version",
                    rm_model::json::Value(
                        json_options.include_flat_selection_compatibility_fields
                            ? uint64_t{1}
                            : uint64_t{2}));
  root.emplace_back("selector_json_mode",
                    rm_model::json::Value(
                        json_options.include_flat_selection_compatibility_fields
                            ? "compat"
                            : "compact"));
  root.emplace_back("dataset", rm_model::json::Value(report_options.dataset_path.string()));
  root.emplace_back("nsw", rm_model::json::Value(report_options.nsw_path.string()));
  if (report_options.nsw_index_path.has_value()) {
    root.emplace_back("nsw_index", rm_model::json::Value(report_options.nsw_index_path->string()));
  }
  if (report_options.query_path.has_value()) {
    root.emplace_back("query", rm_model::json::Value(report_options.query_path->string()));
  } else {
    root.emplace_back("query", rm_model::json::Value(report_options.dataset_path.string()));
  }
  root.emplace_back("hash_bits", rm_model::json::Value(static_cast<uint64_t>(report_options.hash_bits)));
  root.emplace_back("threads", rm_model::json::Value(static_cast<uint64_t>(report_options.threads)));
  root.emplace_back("seed", rm_model::json::Value(report_options.seed));

  root.emplace_back("selection",
                    selector_selection_to_json(result, report_options, options, json_options));

  rm_model::json::Value::Array candidates;
  candidates.reserve(result.candidates.size());
  for (const auto& item : result.candidates) {
    candidates.push_back(selector_candidate_to_json(item, report_options));
  }
  root.emplace_back("candidates", rm_model::json::Value(std::move(candidates)));

  rm_model::json::Value::Array pareto;
  pareto.reserve(result.pareto_front.size());
  for (const auto& item : result.pareto_front) {
    pareto.push_back(selector_candidate_to_json(item, report_options));
  }
  root.emplace_back("pareto_front", rm_model::json::Value(std::move(pareto)));

  rm_model::json::Value::Array recommended;
  recommended.reserve(result.recommended.size());
  for (const auto& item : result.recommended) {
    recommended.push_back(selector_candidate_to_json(item, report_options));
  }
  root.emplace_back("recommended", rm_model::json::Value(std::move(recommended)));

  rm_model::json::Value::Object recommendation_roles;
  if (result.recommended_peak_recall.has_value()) {
    recommendation_roles.emplace_back(
        "peak_recall",
        selector_candidate_to_json(*result.recommended_peak_recall, report_options));
  }
  if (result.recommended_knee.has_value()) {
    recommendation_roles.emplace_back(
        "knee",
        selector_candidate_to_json(*result.recommended_knee, report_options));
  }
  if (result.recommended_fast.has_value()) {
    recommendation_roles.emplace_back(
        "fast",
        selector_candidate_to_json(*result.recommended_fast, report_options));
  }
  if (result.recommended_small.has_value()) {
    recommendation_roles.emplace_back(
        "small",
        selector_candidate_to_json(*result.recommended_small, report_options));
  }
  root.emplace_back("recommendation_roles",
                    rm_model::json::Value(std::move(recommendation_roles)));

  rm_model::json::Value::Array configs;
  configs.reserve(result.recommended.size());
  for (const auto& item : result.recommended) {
    configs.push_back(selector_config_to_json(item.config));
  }
  root.emplace_back("configs", rm_model::json::Value(std::move(configs)));

  if (result.best.has_value()) {
    root.emplace_back("best", selector_candidate_to_json(*result.best, report_options));
    root.emplace_back("selected_target_skeleton",
                      rm_model::json::Value(result.best->config.target_skeleton));
  } else {
    root.emplace_back("best", rm_model::json::Value());
    root.emplace_back("selected_target_skeleton", rm_model::json::Value());
  }

  return rm_model::json::Value(std::move(root));
}

} // namespace vortex
