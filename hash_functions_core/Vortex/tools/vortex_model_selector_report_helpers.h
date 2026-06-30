#ifndef VORTEX_V1_TOOLS_VORTEX_MODEL_SELECTOR_REPORT_HELPERS_H
#define VORTEX_V1_TOOLS_VORTEX_MODEL_SELECTOR_REPORT_HELPERS_H

#include "vortex_v1/model_selector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace vortex::tools::model_selector_report::detail {


inline std::string fmt_double(double value, int digits) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(digits) << value;
  return oss.str();
}

inline std::string fmt_bytes(uint64_t bytes) {
  static const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < (sizeof(kUnits) / sizeof(kUnits[0]))) {
    value /= 1024.0;
    ++unit;
  }
  std::ostringstream oss;
  if (unit == 0) {
    oss << bytes << " " << kUnits[unit];
  } else {
    oss << std::fixed << std::setprecision(2) << value << " " << kUnits[unit];
  }
  return oss.str();
}

inline std::string fmt_count(uint64_t value) {
  std::string digits = std::to_string(value);
  std::string out;
  out.reserve(digits.size() + digits.size() / 3);
  for (std::size_t i = 0; i < digits.size(); ++i) {
    if (i > 0 && ((digits.size() - i) % 3) == 0) {
      out.push_back(',');
    }
    out.push_back(digits[i]);
  }
  return out;
}

inline std::string fmt_score(double value, int digits = 4) {
  if (!std::isfinite(value) || value < -1e-12) {
    return "n/a";
  }
  return fmt_double(value, digits);
}

inline std::string fmt_ms(double value, int digits = 3) {
  if (!std::isfinite(value)) {
    return "n/a";
  }
  return fmt_double(value, digits);
}

inline std::string fmt_tick(double value) {
  double abs_value = std::abs(value);
  int digits = 2;
  if (abs_value < 1.0) {
    digits = 3;
  } else if (abs_value >= 1000.0) {
    digits = 0;
  }
  return fmt_double(value, digits);
}

template <typename T>
inline std::string format_list(const std::vector<T>& values, std::size_t limit = 8) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < values.size() && i < limit; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  if (values.size() > limit) {
    oss << ", ...";
  }
  return oss.str();
}

inline std::string format_config(const vortex::SelectorCandidateConfig& config) {
  std::ostringstream oss;
  if (config.target_skeleton > 0) {
    oss << "target_skeleton=" << config.target_skeleton << ", ";
  }
  oss << "K=" << config.K
      << ", knn=" << config.centroid_knn
      << ", cdf=" << config.cdf_model_spec
      << ", branch=" << config.cdf_branching_factor
      << ", 2opt=" << (config.enable_2opt ? "on" : "off")
      << ", 2opt_iters=" << config.two_opt_iterations
      << ", graph_order=" << (config.enable_graph_centroid_order ? "on" : "off");
  return oss.str();
}

inline std::string escape_html(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (char ch : input) {
    switch (ch) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      case '\'': out += "&#39;"; break;
      default: out.push_back(ch); break;
    }
  }
  return out;
}

inline std::string metric_key(const vortex::SelectorCandidateMetrics& item) {
  return item.phase + "|" + item.nsw_path + "|" + format_config(item.config);
}

inline std::string format_local_timestamp() {
  std::time_t now = std::time(nullptr);
  std::tm tm_now{};
#if defined(_WIN32)
  localtime_s(&tm_now, &now);
#else
  localtime_r(&now, &tm_now);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

struct PlotPoint {
  double x = 0.0;
  double y = 0.0;
  std::string label;
  bool highlight = false;
};

inline std::string build_scatter_svg(const std::string& title,
                              const std::string& x_label,
                              const std::string& y_label,
                              const std::vector<PlotPoint>& points,
                              std::optional<std::pair<double, double>> y_bounds = std::nullopt) {
  std::vector<PlotPoint> finite_points;
  finite_points.reserve(points.size());
  for (const auto& point : points) {
    if (std::isfinite(point.x) && std::isfinite(point.y)) {
      finite_points.push_back(point);
    }
  }
  if (finite_points.empty()) {
    return "<div class=\"empty-card\">No data available for this chart.</div>";
  }

  constexpr double width = 760.0;
  constexpr double height = 372.0;
  constexpr double margin_left = 76.0;
  constexpr double margin_right = 28.0;
  constexpr double margin_top = 50.0;
  constexpr double margin_bottom = 64.0;
  const double plot_width = width - margin_left - margin_right;
  const double plot_height = height - margin_top - margin_bottom;

  double min_x = finite_points.front().x;
  double max_x = finite_points.front().x;
  double min_y = finite_points.front().y;
  double max_y = finite_points.front().y;
  for (const auto& point : finite_points) {
    min_x = std::min(min_x, point.x);
    max_x = std::max(max_x, point.x);
    min_y = std::min(min_y, point.y);
    max_y = std::max(max_y, point.y);
  }
  if (max_x - min_x < 1e-12) {
    double delta = std::max(0.05, std::abs(min_x) * 0.05);
    min_x -= delta;
    max_x += delta;
  }
  if (max_y - min_y < 1e-12) {
    double delta = std::max(0.05, std::abs(min_y) * 0.05);
    min_y -= delta;
    max_y += delta;
  }
  const double x_pad = (max_x - min_x) * 0.05;
  min_x -= x_pad;
  max_x += x_pad;
  if (y_bounds.has_value()) {
    min_y = y_bounds->first;
    max_y = y_bounds->second;
  } else {
    const double y_pad = (max_y - min_y) * 0.05;
    min_y -= y_pad;
    max_y += y_pad;
  }

  auto x_to_svg = [&](double x) {
    return margin_left + ((x - min_x) / (max_x - min_x)) * plot_width;
  };
  auto y_to_svg = [&](double y) {
    return margin_top + (1.0 - ((y - min_y) / (max_y - min_y))) * plot_height;
  };

  std::ostringstream svg;
  svg << "<svg viewBox=\"0 0 " << width << " " << height
      << "\" role=\"img\" aria-label=\"" << escape_html(title) << "\">";
  svg << "<rect x=\"0\" y=\"0\" width=\"" << width
      << "\" height=\"" << height << "\" fill=\"#ffffff\" rx=\"8\"/>";
  svg << "<text x=\"" << margin_left << "\" y=\"26\" text-anchor=\"start\""
      << " fill=\"#0f172a\" font-size=\"14\" font-weight=\"700\">"
      << escape_html(title) << "</text>";
  svg << "<circle cx=\"" << (width - 156.0) << "\" cy=\"22\" r=\"4\" fill=\"#1d4ed8\"/>";
  svg << "<text x=\"" << (width - 146.0) << "\" y=\"26\" fill=\"#475569\" font-size=\"11\">candidate</text>";
  svg << "<circle cx=\"" << (width - 76.0) << "\" cy=\"22\" r=\"4.8\" fill=\"#b45309\"/>";
  svg << "<circle cx=\"" << (width - 76.0) << "\" cy=\"22\" r=\"7.2\" fill=\"none\" stroke=\"#b45309\" stroke-width=\"1.4\"/>";
  svg << "<text x=\"" << (width - 66.0) << "\" y=\"26\" fill=\"#475569\" font-size=\"11\">recommended</text>";

  for (int tick = 0; tick <= 5; ++tick) {
    double ratio = static_cast<double>(tick) / 5.0;
    double x = margin_left + ratio * plot_width;
    double y = margin_top + ratio * plot_height;
    svg << "<line x1=\"" << x << "\" y1=\"" << margin_top
        << "\" x2=\"" << x << "\" y2=\"" << (margin_top + plot_height)
        << "\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>";
    svg << "<line x1=\"" << margin_left << "\" y1=\"" << y
        << "\" x2=\"" << (margin_left + plot_width) << "\" y2=\"" << y
        << "\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>";

    double x_value = min_x + ratio * (max_x - min_x);
    double y_value = max_y - ratio * (max_y - min_y);
    svg << "<text x=\"" << x << "\" y=\"" << (margin_top + plot_height + 20)
        << "\" text-anchor=\"middle\" fill=\"#64748b\" font-size=\"11\">"
        << escape_html(fmt_tick(x_value)) << "</text>";
    svg << "<text x=\"" << (margin_left - 8) << "\" y=\"" << (y + 4)
        << "\" text-anchor=\"end\" fill=\"#64748b\" font-size=\"11\">"
        << escape_html(fmt_tick(y_value)) << "</text>";
  }

  for (const auto& point : finite_points) {
    double cx = x_to_svg(point.x);
    double cy = y_to_svg(point.y);
    const char* color = point.highlight ? "#b45309" : "#1d4ed8";
    double radius = point.highlight ? 4.8 : 3.3;
    if (point.highlight) {
      svg << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"7.2\""
          << " fill=\"none\" stroke=\"#b45309\" stroke-width=\"1.4\" opacity=\"0.85\"/>";
    }
    svg << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << radius
        << "\" fill=\"" << color << "\" opacity=\"0.92\">";
    svg << "<title>" << escape_html(point.label) << "</title>";
    svg << "</circle>";
  }

  svg << "<line x1=\"" << margin_left << "\" y1=\"" << (margin_top + plot_height)
      << "\" x2=\"" << (margin_left + plot_width) << "\" y2=\"" << (margin_top + plot_height)
      << "\" stroke=\"#475569\" stroke-width=\"1.25\"/>";
  svg << "<line x1=\"" << margin_left << "\" y1=\"" << margin_top
      << "\" x2=\"" << margin_left << "\" y2=\"" << (margin_top + plot_height)
      << "\" stroke=\"#475569\" stroke-width=\"1.25\"/>";
  svg << "<text x=\"" << (margin_left + plot_width / 2.0)
      << "\" y=\"" << (height - 14.0) << "\" text-anchor=\"middle\""
      << " fill=\"#334155\" font-size=\"12\">" << escape_html(x_label) << "</text>";
  svg << "<text x=\"18\" y=\"" << (margin_top + plot_height / 2.0)
      << "\" text-anchor=\"middle\" fill=\"#334155\" font-size=\"12\""
      << " transform=\"rotate(-90 18 " << (margin_top + plot_height / 2.0) << ")\">"
      << escape_html(y_label) << "</text>";
  svg << "</svg>";
  return svg.str();
}

inline std::string format_skeleton_target(const vortex::SelectorCandidateMetrics& item) {
  if (item.config.target_skeleton == 0) {
    return "fixed";
  }
  return std::to_string(item.config.target_skeleton);
}

inline std::string build_metrics_table(const std::vector<const vortex::SelectorCandidateMetrics*>& items,
                                std::size_t limit) {
  std::ostringstream html;
  html << "<table><thead><tr>"
       << "<th>#</th><th>Score</th><th>Quality</th><th>Match</th><th>Recall</th><th>AUC</th><th>Tail P05</th>"
       << "<th>Node Locality</th><th>Rank Dist</th><th>Rank P95</th><th>Lat Avg (ms)</th>"
       << "<th>P95 (ms)</th><th>Model Size</th><th>Train (ms)</th>"
       << "<th>target_skeleton</th><th>skeleton_nodes</th><th>Config</th>"
       << "</tr></thead><tbody>";
  std::size_t rows = std::min<std::size_t>(items.size(), limit);
  for (std::size_t i = 0; i < rows; ++i) {
    const auto& item = *items[i];
    html << "<tr><td>" << (i + 1) << "</td>"
         << "<td>" << escape_html(fmt_score(item.objective_score, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.locality_quality_score, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.overlay_match_score, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.recall_at_k_in_window, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.recall_auc_log_window, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.query_recall_p05, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.node_locality_score, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.mean_rank_distance_norm, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.rank_distance_norm_p95, 6)) << "</td>"
         << "<td>" << escape_html(fmt_ms(item.latency_avg_ms, 6)) << "</td>"
         << "<td>" << escape_html(fmt_ms(item.latency_p95_ms, 6)) << "</td>"
         << "<td>" << escape_html(fmt_bytes(item.model_size_bytes)) << "</td>"
         << "<td>" << escape_html(fmt_ms(item.train_time_ms, 3)) << "</td>"
         << "<td>" << escape_html(format_skeleton_target(item)) << "</td>"
         << "<td>" << escape_html(fmt_count(item.skeleton_node_count)) << "</td>"
         << "<td class=\"mono\">" << escape_html(format_config(item.config)) << "</td></tr>";
  }
  html << "</tbody></table>";
  return html.str();
}

inline std::string format_roles(const std::vector<std::string>& roles) {
  if (roles.empty()) {
    return "-";
  }
  return format_list(roles, roles.size());
}

inline std::string format_base_rank_attribution_config(
    const vortex::SelectorBaseRankCacheAttribution& row) {
  std::ostringstream oss;
  oss << "K=" << row.K
      << ", knn=" << row.centroid_knn
      << ", cdf=" << row.cdf_model_spec
      << ", branch=" << row.cdf_branching_factor
      << ", 2opt=" << (row.enable_2opt ? "on" : "off")
      << ", 2opt_iters=" << row.two_opt_iterations
      << ", graph_order=" << (row.enable_graph_centroid_order ? "on" : "off");
  return oss.str();
}

struct BaseRankAttributionAggregate {
  std::string phase;
  uint64_t target_skeleton = 0;
  uint32_t skeleton_node_count = 0;
  uint32_t K = 0;
  uint32_t hits = 0;
  uint32_t misses = 0;
  double hash_ms = 0.0;
  double build_ms = 0.0;
  uint64_t retained_memory_bytes = 0;
  uint32_t evictions = 0;
  uint64_t evicted_bytes = 0;
};

inline std::string base_rank_aggregate_key(
    const vortex::SelectorBaseRankCacheAttribution& row) {
  return row.phase + "|" + std::to_string(row.target_skeleton) + "|" +
         std::to_string(row.skeleton_node_count) + "|" + std::to_string(row.K);
}

inline std::vector<BaseRankAttributionAggregate> aggregate_base_rank_attribution(
    const std::vector<vortex::SelectorBaseRankCacheAttribution>& rows) {
  std::vector<BaseRankAttributionAggregate> aggregates;
  std::vector<std::string> keys;
  for (const auto& row : rows) {
    std::string key = base_rank_aggregate_key(row);
    auto found = std::find(keys.begin(), keys.end(), key);
    std::size_t index = 0;
    if (found == keys.end()) {
      keys.push_back(key);
      BaseRankAttributionAggregate aggregate;
      aggregate.phase = row.phase;
      aggregate.target_skeleton = row.target_skeleton;
      aggregate.skeleton_node_count = row.skeleton_node_count;
      aggregate.K = row.K;
      aggregates.push_back(aggregate);
      index = aggregates.size() - 1;
    } else {
      index = static_cast<std::size_t>(found - keys.begin());
    }
    auto& aggregate = aggregates[index];
    aggregate.hits += row.hits;
    aggregate.misses += row.misses;
    aggregate.hash_ms += row.hash_ms;
    aggregate.build_ms += row.build_ms;
    aggregate.retained_memory_bytes += row.retained_memory_bytes;
    aggregate.evictions += row.evictions;
    aggregate.evicted_bytes += row.evicted_bytes;
  }
  std::sort(aggregates.begin(), aggregates.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.hash_ms != rhs.hash_ms) return lhs.hash_ms > rhs.hash_ms;
    if (lhs.misses != rhs.misses) return lhs.misses > rhs.misses;
    if (lhs.build_ms != rhs.build_ms) return lhs.build_ms > rhs.build_ms;
    if (lhs.phase != rhs.phase) return lhs.phase < rhs.phase;
    if (lhs.target_skeleton != rhs.target_skeleton) {
      return lhs.target_skeleton < rhs.target_skeleton;
    }
    return lhs.K < rhs.K;
  });
  return aggregates;
}

inline std::string build_base_rank_aggregate_table(
    const std::vector<vortex::SelectorBaseRankCacheAttribution>& rows,
    std::size_t limit) {
  auto aggregates = aggregate_base_rank_attribution(rows);
  if (aggregates.empty()) {
    return "<div class=\"empty-card\">No base-rank cache attribution rows.</div>";
  }
  std::ostringstream html;
  html << "<table><thead><tr>"
       << "<th>#</th><th>Phase</th><th>target_skeleton</th><th>skeleton_nodes</th>"
       << "<th>K</th><th>Hits</th><th>Misses</th><th>Hash (ms)</th>"
       << "<th>Build (ms)</th><th>Retained</th><th>Evicted</th>"
       << "</tr></thead><tbody>";
  std::size_t count = std::min<std::size_t>(aggregates.size(), limit);
  for (std::size_t i = 0; i < count; ++i) {
    const auto& row = aggregates[i];
    html << "<tr><td>" << (i + 1) << "</td>"
         << "<td>" << escape_html(row.phase) << "</td>"
         << "<td>" << row.target_skeleton << "</td>"
         << "<td>" << row.skeleton_node_count << "</td>"
         << "<td>" << row.K << "</td>"
         << "<td>" << row.hits << "</td>"
         << "<td>" << row.misses << "</td>"
         << "<td>" << escape_html(fmt_double(row.hash_ms, 3)) << "</td>"
         << "<td>" << escape_html(fmt_double(row.build_ms, 3)) << "</td>"
         << "<td>" << escape_html(fmt_bytes(row.retained_memory_bytes)) << "</td>"
         << "<td>" << row.evictions << " / "
         << escape_html(fmt_bytes(row.evicted_bytes)) << "</td></tr>";
  }
  html << "</tbody></table>";
  return html.str();
}

inline std::string build_base_rank_attribution_table(
    const std::vector<vortex::SelectorBaseRankCacheAttribution>& rows,
    std::size_t limit) {
  if (rows.empty()) {
    return "<div class=\"empty-card\">No base-rank cache attribution rows.</div>";
  }
  std::ostringstream html;
  html << "<table><thead><tr>"
       << "<th>#</th><th>Phase</th><th>Roles</th><th>target_skeleton</th>"
       << "<th>skeleton_nodes</th><th>Base/Query</th><th>Hits</th><th>Misses</th>"
       << "<th>Hash (ms)</th><th>Build (ms)</th><th>Retained</th><th>Evicted</th>"
       << "<th>Config</th>"
       << "</tr></thead><tbody>";
  std::size_t count = std::min<std::size_t>(rows.size(), limit);
  for (std::size_t i = 0; i < count; ++i) {
    const auto& row = rows[i];
    html << "<tr><td>" << (i + 1) << "</td>"
         << "<td>" << escape_html(row.phase) << "</td>"
         << "<td>" << escape_html(format_roles(row.roles)) << "</td>"
         << "<td>" << row.target_skeleton << "</td>"
         << "<td>" << row.skeleton_node_count << "</td>"
         << "<td>" << row.base_count << "/" << row.query_count << "</td>"
         << "<td>" << row.hits << "</td>"
         << "<td>" << row.misses << "</td>"
         << "<td>" << escape_html(fmt_double(row.hash_ms, 3)) << "</td>"
         << "<td>" << escape_html(fmt_double(row.build_ms, 3)) << "</td>"
         << "<td>" << escape_html(fmt_bytes(row.retained_memory_bytes)) << "</td>"
         << "<td>" << row.evictions << " / "
         << escape_html(fmt_bytes(row.evicted_bytes)) << "</td>"
         << "<td class=\"mono\">"
         << escape_html(format_base_rank_attribution_config(row))
         << "</td></tr>";
  }
  html << "</tbody></table>";
  return html.str();
}

inline std::string build_recommendation_roles_table(const vortex::SelectorResult& result) {
  struct RoleRow {
    const char* name;
    const vortex::SelectorCandidateMetrics* item;
    const char* why;
  };
  std::vector<RoleRow> rows = {
      {"Peak Recall", result.recommended_peak_recall ? &*result.recommended_peak_recall : nullptr,
       "highest locality quality / recall curve"},
      {"Knee", result.recommended_knee ? &*result.recommended_knee : nullptr,
       "best quality-cost tradeoff"},
      {"Fast", result.recommended_fast ? &*result.recommended_fast : nullptr,
       "lowest latency near peak quality"},
      {"Small", result.recommended_small ? &*result.recommended_small : nullptr,
       "smallest model near peak quality"},
  };
  std::ostringstream html;
  html << "<table><thead><tr>"
       << "<th>Role</th><th>Why</th><th>Quality</th><th>Match</th><th>Node Locality</th><th>Recall</th><th>AUC</th>"
       << "<th>Lat Avg (ms)</th><th>Model Size</th><th>Config</th>"
       << "</tr></thead><tbody>";
  for (const auto& row : rows) {
    if (row.item == nullptr) {
      continue;
    }
    const auto& item = *row.item;
    html << "<tr><td>" << row.name << "</td>"
         << "<td>" << escape_html(row.why) << "</td>"
         << "<td>" << escape_html(fmt_score(item.locality_quality_score, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.overlay_match_score, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.node_locality_score, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.recall_at_k_in_window, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(item.recall_auc_log_window, 6)) << "</td>"
         << "<td>" << escape_html(fmt_ms(item.latency_avg_ms, 6)) << "</td>"
         << "<td>" << escape_html(fmt_bytes(item.model_size_bytes)) << "</td>"
         << "<td class=\"mono\">" << escape_html(format_config(item.config)) << "</td></tr>";
  }
  html << "</tbody></table>";
  return html.str();
}

inline std::string build_recall_curve_table(const vortex::SelectorResult& result) {
  struct RoleRow {
    const char* name;
    const vortex::SelectorCandidateMetrics* item;
  };
  std::vector<RoleRow> rows = {
      {"Peak Recall", result.recommended_peak_recall ? &*result.recommended_peak_recall : nullptr},
      {"Knee", result.recommended_knee ? &*result.recommended_knee : nullptr},
      {"Fast", result.recommended_fast ? &*result.recommended_fast : nullptr},
      {"Small", result.recommended_small ? &*result.recommended_small : nullptr},
  };
  std::vector<uint64_t> windows = result.effective_eval_windows;
  if (windows.empty()) {
    for (const auto& row : rows) {
      if (row.item != nullptr) {
        windows = row.item->recall_windows;
        break;
      }
    }
  }
  if (windows.empty()) {
    return "<div class=\"empty-card\">No recall-curve data available.</div>";
  }

  std::ostringstream html;
  html << "<table><thead><tr><th>Role</th>";
  for (uint64_t window : windows) {
    html << "<th>w=" << window << "</th>";
  }
  html << "<th>AUC</th><th>Config</th></tr></thead><tbody>";
  for (const auto& row : rows) {
    if (row.item == nullptr) {
      continue;
    }
    const auto& item = *row.item;
    html << "<tr><td>" << row.name << "</td>";
    for (uint64_t window : windows) {
      double recall = 0.0;
      for (std::size_t i = 0; i < item.recall_windows.size() &&
                              i < item.recall_at_k_by_window.size(); ++i) {
        if (item.recall_windows[i] == window) {
          recall = item.recall_at_k_by_window[i];
          break;
        }
      }
      html << "<td>" << escape_html(fmt_score(recall, 6)) << "</td>";
    }
    html << "<td>" << escape_html(fmt_score(item.recall_auc_log_window, 6)) << "</td>"
         << "<td class=\"mono\">" << escape_html(format_config(item.config)) << "</td></tr>";
  }
  html << "</tbody></table>";
  return html.str();
}

inline std::string build_skeleton_summary_table(const std::vector<const vortex::SelectorCandidateMetrics*>& candidates) {
  struct SkeletonSummary {
    std::size_t count = 0;
    double best_score = -1.0;
    double best_match = 0.0;
    double best_node_locality = 0.0;
    double best_recall = 0.0;
    double best_latency = std::numeric_limits<double>::infinity();
    uint64_t best_size = std::numeric_limits<uint64_t>::max();
  };

  std::map<uint64_t, SkeletonSummary> by_skeleton;
  for (const auto* item_ptr : candidates) {
    const auto& item = *item_ptr;
    if (!item.ok || item.config.target_skeleton == 0) {
      continue;
    }
    auto& summary = by_skeleton[item.config.target_skeleton];
    summary.count += 1;
    summary.best_score = std::max(summary.best_score, item.objective_score);
    summary.best_match = std::max(summary.best_match, item.overlay_match_score);
    summary.best_node_locality =
        std::max(summary.best_node_locality, item.node_locality_score);
    summary.best_recall = std::max(summary.best_recall, item.recall_at_k_in_window);
    summary.best_latency = std::min(summary.best_latency, item.latency_avg_ms);
    summary.best_size = std::min(summary.best_size, item.model_size_bytes);
  }
  if (by_skeleton.empty()) {
    return "<div class=\"empty-card\">No target_skeleton data available.</div>";
  }

  std::ostringstream html;
  html << "<table><thead><tr>"
       << "<th>target_skeleton</th><th>Configs</th><th>Best Score</th><th>Best Match</th><th>Best Node Locality</th><th>Best Recall</th>"
       << "<th>Best Lat Avg (ms)</th><th>Smallest Model</th>"
       << "</tr></thead><tbody>";
  for (const auto& [target, summary] : by_skeleton) {
    html << "<tr><td>" << target << "</td>"
         << "<td>" << summary.count << "</td>"
         << "<td>" << escape_html(fmt_score(summary.best_score, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(summary.best_match, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(summary.best_node_locality, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(summary.best_recall, 6)) << "</td>"
         << "<td>" << escape_html(fmt_ms(summary.best_latency, 6)) << "</td>"
         << "<td>" << escape_html(fmt_bytes(summary.best_size)) << "</td></tr>";
  }
  html << "</tbody></table>";
  return html.str();
}

inline std::string build_skeleton_benchmark_table(
    const std::vector<const vortex::SelectorCandidateMetrics*>& candidates,
    uint64_t dataset_count,
    const std::vector<double>& requested_percentages) {
  std::map<uint64_t, const vortex::SelectorCandidateMetrics*> best_by_target;
  for (const auto* item_ptr : candidates) {
    const auto& item = *item_ptr;
    if (!item.ok || item.config.target_skeleton == 0) {
      continue;
    }
    auto it = best_by_target.find(item.config.target_skeleton);
    if (it == best_by_target.end() || item.objective_score > it->second->objective_score) {
      best_by_target[item.config.target_skeleton] = &item;
    }
  }
  if (best_by_target.empty()) {
    return "<div class=\"empty-card\">No evaluated target_skeleton sweep data available.</div>";
  }

  std::ostringstream html;
  html << "<table><thead><tr>";
  if (!requested_percentages.empty()) {
    html << "<th>Requested %</th>";
  }
  html << "<th>Dataset %</th><th>target_skeleton</th><th>Score</th><th>Match</th><th>Node Locality</th><th>Recall</th>"
       << "<th>Lat Avg (ms)</th><th>Model Size</th><th>Config</th>"
       << "</tr></thead><tbody>";

  auto requested_percent_label = [&](uint64_t target_skeleton) -> std::string {
    if (requested_percentages.empty() || dataset_count == 0) {
      return "";
    }
    double actual_percent =
        (100.0 * static_cast<double>(target_skeleton)) / static_cast<double>(dataset_count);
    double best_diff = std::numeric_limits<double>::infinity();
    std::optional<double> matched;
    for (double requested : requested_percentages) {
      double diff = std::abs(requested - actual_percent);
      if (diff < best_diff) {
        best_diff = diff;
        matched = requested;
      }
    }
    if (!matched.has_value()) {
      return "";
    }
    return fmt_double(*matched, 2) + "%";
  };

  for (const auto& [target_skeleton, row] : best_by_target) {
    double actual_percent = dataset_count == 0
        ? 0.0
        : (100.0 * static_cast<double>(target_skeleton)) / static_cast<double>(dataset_count);
    html << "<tr>";
    if (!requested_percentages.empty()) {
      html << "<td>" << escape_html(requested_percent_label(target_skeleton)) << "</td>";
    }
    html << "<td>" << escape_html(fmt_double(actual_percent, 2)) << "%</td>"
         << "<td>" << escape_html(fmt_count(row->config.target_skeleton)) << "</td>"
         << "<td>" << escape_html(fmt_score(row->objective_score, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(row->overlay_match_score, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(row->node_locality_score, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(row->recall_at_k_in_window, 6)) << "</td>"
         << "<td>" << escape_html(fmt_ms(row->latency_avg_ms, 6)) << "</td>"
         << "<td>" << escape_html(fmt_bytes(row->model_size_bytes)) << "</td>"
         << "<td class=\"mono\">" << escape_html(format_config(row->config)) << "</td>"
         << "</tr>";
  }
  html << "</tbody></table>";
  return html.str();
}

inline std::string build_context_stats_table(const std::vector<vortex::SelectorContextStats>& stats) {
  if (stats.empty()) {
    return "<div class=\"empty-card\">No per-context selector telemetry available.</div>";
  }

  std::ostringstream html;
  html << "<table><thead><tr>"
       << "<th>target_skeleton</th><th>requested aliases</th><th>nodes</th><th>layer</th><th>Train Budget</th>"
       << "<th>Train Attempts</th><th>Budget Hit</th><th>Phase1 OK/Candidates</th>"
       << "<th>Phase2 OK/Candidates</th><th>Decision</th><th>Phase2 Limit</th>"
       << "<th>Beam Frontier</th><th>Compacted</th>"
       << "<th>Beam Stop</th><th>Beam Plateau</th><th>Best P1 Recall</th><th>Best P2 Recall</th>"
       << "<th>Best P1 Quality</th><th>Best P2 Quality</th><th>Quality Gain</th>"
       << "<th>K Search</th><th>Pruned K</th><th>Vectors/Centroid</th>"
       << "<th>Skeleton Nodes/Centroid</th><th>Budget Failures</th><th>Elapsed</th>"
       << "</tr></thead><tbody>";
  for (const auto& row : stats) {
    std::ostringstream aliases;
    for (std::size_t i = 0; i < row.requested_target_skeletons.size(); ++i) {
      if (i > 0) {
        aliases << ", ";
      }
      aliases << row.requested_target_skeletons[i];
    }
    html << "<tr><td>" << escape_html(fmt_count(row.target_skeleton)) << "</td>"
         << "<td>" << escape_html(aliases.str()) << "</td>"
         << "<td>" << escape_html(fmt_count(row.skeleton_node_count)) << "</td>"
         << "<td>" << row.skeleton_layer << "</td>"
         << "<td>" << row.context_model_train_budget << "</td>"
         << "<td>" << row.model_trains_started_before << " &rarr; "
         << row.model_trains_started_after << "</td>"
         << "<td>" << (row.context_model_train_budget_hit ? "yes" : "no") << "</td>"
         << "<td>" << row.phase1_ok << "/" << row.phase1_candidates << "</td>"
         << "<td>" << row.phase2_ok << "/" << row.phase2_candidates << "</td>"
         << "<td>" << escape_html(row.promotion_decision) << "</td>"
         << "<td>" << row.phase2_candidate_limit << "</td>"
         << "<td>" << row.phase2_frontier_candidates << "</td>"
         << "<td>" << row.phase2_compacted_candidates << "</td>"
         << "<td>" << escape_html(row.phase2_stopped_reason) << "</td>"
         << "<td>" << row.phase2_plateau_rounds << "</td>"
         << "<td>" << escape_html(fmt_score(row.best_phase1_recall, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(row.best_phase2_recall, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(row.best_phase1_quality, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(row.best_phase2_quality, 6)) << "</td>"
         << "<td>" << escape_html(fmt_score(row.phase2_quality_gain, 6)) << "</td>"
         << "<td class=\"mono\">" << escape_html(format_list(row.effective_K_values)) << "</td>"
         << "<td class=\"mono\">" << escape_html(format_list(row.pruned_K_values)) << "</td>"
         << "<td>" << escape_html(fmt_double(row.min_vectors_per_centroid, 2))
         << "&ndash;" << escape_html(fmt_double(row.max_vectors_per_centroid, 2)) << "</td>"
         << "<td>" << escape_html(fmt_double(row.min_skeleton_nodes_per_centroid, 2))
         << "&ndash;" << escape_html(fmt_double(row.max_skeleton_nodes_per_centroid, 2))
         << "</td>"
         << "<td>" << row.budget_failures << "</td>"
         << "<td>" << escape_html(fmt_double(row.elapsed_ms / 1000.0, 2)) << " s</td></tr>";
  }
  html << "</tbody></table>";
  return html.str();
}

inline std::string stat_card(const std::string& label,
                      const std::string& value,
                      const std::string& detail = "",
                      const std::string& tone = "") {
  std::ostringstream html;
  html << "<div class=\"stat";
  if (!tone.empty()) {
    html << " " << escape_html(tone);
  }
  html << "\"><div class=\"stat-label\">" << escape_html(label)
       << "</div><div class=\"stat-value\">" << escape_html(value) << "</div>";
  if (!detail.empty()) {
    html << "<div class=\"stat-detail\">" << escape_html(detail) << "</div>";
  }
  html << "</div>";
  return html.str();
}

inline std::string status_pill(const std::string& label, const std::string& tone) {
  return "<span class=\"pill " + escape_html(tone) + "\">" + escape_html(label) + "</span>";
}

inline std::string build_best_config_panel(const vortex::SelectorResult& result) {
  if (!result.best.has_value()) {
    return "<section class=\"panel primary\"><h2>Selected Best</h2>"
           "<div class=\"empty-card\">No best candidate was selected.</div></section>";
  }
  const auto& best = *result.best;
  std::ostringstream html;
  html << "<section class=\"panel primary\"><div class=\"section-head\">"
       << "<div><div class=\"eyebrow\">Selected best</div>"
       << "<h2>Deployment-locality winner</h2></div>"
       << status_pill(best.phase.empty() ? "unknown phase" : best.phase, "neutral")
       << "</div>"
       << "<div class=\"winner-grid\">"
       << stat_card("Quality", fmt_score(best.locality_quality_score, 6), "primary selector objective", "good")
       << stat_card("Overlay Match", fmt_score(best.overlay_match_score, 6), "32-1024 node locality")
       << stat_card("Recall", fmt_score(best.recall_at_k_in_window, 6), "fixed hash-window recall")
       << stat_card("Latency Avg", fmt_ms(best.latency_avg_ms, 6) + " ms", "hash inference")
       << stat_card("Model Size", fmt_bytes(best.model_size_bytes), "serialized runtime model")
       << stat_card("target_skeleton", best.config.target_skeleton == 0
                        ? std::string("fixed")
                        : fmt_count(best.config.target_skeleton),
                    fmt_count(best.skeleton_node_count) + " skeleton nodes")
       << "</div>"
       << "<pre class=\"config-line\">" << escape_html(format_config(best.config)) << "</pre>"
       << "</section>";
  return html.str();
}

inline std::string build_metric_guide() {
  return "<section class=\"panel guide\"><div class=\"section-head\">"
         "<div><div class=\"eyebrow\">How to read this report</div>"
         "<h2>Metrics and scope</h2></div></div>"
         "<div class=\"guide-grid\">"
         "<div><strong>Overlay Match</strong><span>Primary deployment signal. Higher means true neighbors land closer on the simulated 32-1024 node overlay.</span></div>"
         "<div><strong>Recall</strong><span>ANN-style fixed-window recall. Useful for quality studies, but not the only placement target.</span></div>"
         "<div><strong>Objective Score</strong><span>Phase-local score. Compare rows within the same phase, not across coarse and calibrated phases.</span></div>"
         "<div><strong>Screening vs Final</strong><span>Phase-1/phase-2 charts are search-screening evidence. Trust final-calibration rows and generated-code evaluation for selected artifacts.</span></div>"
         "<div><strong>Recommended Points</strong><span>Outlined chart points are selected recommendation roles: peak recall, knee, fast, or small.</span></div>"
         "</div></section>";
}


} // namespace vortex::tools::model_selector_report::detail

#endif // VORTEX_V1_TOOLS_VORTEX_MODEL_SELECTOR_REPORT_HELPERS_H
