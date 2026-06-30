#ifndef VORTEX_V1_SELECTOR_REPORT_INTERNAL_H
#define VORTEX_V1_SELECTOR_REPORT_INTERNAL_H

#include "selector_internal.h"

namespace vortex::selector_internal {

rm_model::json::Value graph_centroid_order_evidence_to_json(const SelectorResult& result);
rm_model::json::Value uint32_array_to_json(const std::vector<uint32_t>& values);
rm_model::json::Value uint64_array_to_json(const std::vector<uint64_t>& values);
rm_model::json::Value string_array_to_json(const std::vector<std::string>& values);
rm_model::json::Value base_rank_cache_attribution_to_json(
    const SelectorBaseRankCacheAttribution& row);
rm_model::json::Value calibration_probe_to_json(const SelectorCalibrationProbe& probe);
rm_model::json::Value selector_diagnostics_to_json(const SelectorResult& result,
                                                   const SelectorOptions& options);
rm_model::json::Value selector_selection_to_json(const SelectorResult& result,
                                                 const SelectorOptions& report_options,
                                                 const SelectorOptions& raw_options,
                                                 const SelectorJsonOptions& json_options);

} // namespace vortex::selector_internal

#endif // VORTEX_V1_SELECTOR_REPORT_INTERNAL_H
