#ifndef VORTEX_V1_TOOLS_VORTEX_MODEL_SELECTOR_REPORT_H
#define VORTEX_V1_TOOLS_VORTEX_MODEL_SELECTOR_REPORT_H

#include "vortex_v1/model_selector.h"

#include <filesystem>

namespace vortex::tools::model_selector_report {

std::filesystem::path default_html_report_path(const std::filesystem::path& json_output_path);

void write_html_report(const std::filesystem::path& report_path,
                       const vortex::SelectorResult& result,
                       const vortex::SelectorOptions& options,
                       const std::filesystem::path& json_output_path);

void print_summary(const vortex::SelectorResult& result,
                   const vortex::SelectorOptions& options);

} // namespace vortex::tools::model_selector_report

#endif // VORTEX_V1_TOOLS_VORTEX_MODEL_SELECTOR_REPORT_H
