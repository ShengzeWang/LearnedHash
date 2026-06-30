#include "vortex_model_selector_report.h"
#include "vortex_model_selector_report_helpers.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vortex::tools::model_selector_report {

using namespace detail;

namespace {

std::string build_html_report(const vortex::SelectorResult& result,
                              const vortex::SelectorOptions& options,
                              const std::filesystem::path& json_output_path) {
  std::vector<const vortex::SelectorCandidateMetrics*> phase2_ok;
  phase2_ok.reserve(result.candidates.size());
  for (const auto& item : result.candidates) {
    if (item.phase == "phase2" && item.ok) {
      phase2_ok.push_back(&item);
    }
  }

  std::set<std::string> recommended_keys;
  for (const auto& item : result.recommended) {
    recommended_keys.insert(metric_key(item));
  }

  std::vector<PlotPoint> recall_vs_latency;
  std::vector<PlotPoint> recall_vs_size;
  std::vector<PlotPoint> score_vs_skeleton;
  recall_vs_latency.reserve(phase2_ok.size());
  recall_vs_size.reserve(phase2_ok.size());
  score_vs_skeleton.reserve(phase2_ok.size());
  for (const auto* item_ptr : phase2_ok) {
    const auto& item = *item_ptr;
    std::string label = "score=" + fmt_double(item.objective_score, 6) +
                        ", recall=" + fmt_double(item.recall_at_k_in_window, 6) +
                        ", " + format_config(item.config);
    bool highlight = recommended_keys.count(metric_key(item)) > 0;
    recall_vs_latency.push_back(PlotPoint{
        item.latency_avg_ms, item.recall_at_k_in_window, label, highlight});
    recall_vs_size.push_back(PlotPoint{
        static_cast<double>(item.model_size_bytes) / 1024.0, item.recall_at_k_in_window, label, highlight});
    if (item.config.target_skeleton > 0) {
      score_vs_skeleton.push_back(PlotPoint{
          static_cast<double>(item.config.target_skeleton), item.objective_score, label, highlight});
    }
  }

  std::vector<const vortex::SelectorCandidateMetrics*> recommended_ptrs;
  recommended_ptrs.reserve(result.recommended.size());
  for (const auto& item : result.recommended) {
    recommended_ptrs.push_back(&item);
  }
  std::vector<const vortex::SelectorCandidateMetrics*> pareto_ptrs;
  pareto_ptrs.reserve(result.pareto_front.size());
  for (const auto& item : result.pareto_front) {
    pareto_ptrs.push_back(&item);
  }

  uint32_t ok_count = 0;
  uint32_t phase2_count = 0;
  for (const auto& item : result.candidates) {
    if (item.ok) {
      ++ok_count;
      if (item.phase == "phase2") {
        ++phase2_count;
      }
    }
  }
  uint32_t failed_count = static_cast<uint32_t>(result.candidates.size()) - ok_count;
  std::string stop_tone = (result.search_time_budget_hit || result.model_train_budget_hit ||
                           result.final_calibration_reserve_hit)
                              ? "warn"
                              : "good";

  std::ostringstream html;
  html << "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
       << "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
       << "<title>Vortex v1 Model Selector Report</title>"
       << "<style>"
       << ":root{--page:#f6f7f3;--card:#ffffff;--ink:#111827;--muted:#64748b;--line:#d8ded2;"
       << "--soft:#eef1ea;--blue:#1d4ed8;--amber:#b45309;--green:#166534;--red:#b91c1c;}"
       << "*{box-sizing:border-box;}body{margin:0;background:var(--page);color:var(--ink);"
       << "font:14px/1.45 \"Avenir Next\",\"Segoe UI\",system-ui,sans-serif;padding:24px;"
       << "font-variant-numeric:tabular-nums;}"
       << ".wrap{max-width:1480px;margin:0 auto;display:grid;gap:16px;}"
       << ".hero{background:#111827;color:#f8fafc;border:1px solid #0f172a;"
       << "border-radius:8px;padding:20px 22px;box-shadow:0 18px 40px rgba(15,23,42,0.16);}"
       << ".hero h1{margin:0 0 8px;font-size:24px;letter-spacing:0.1px;}"
       << ".meta{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:6px 16px;color:#cbd5e1;}"
       << ".meta strong{color:#f8fafc;}.section-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:10px;}"
       << ".eyebrow{text-transform:uppercase;letter-spacing:0.08em;font-size:11px;color:var(--muted);font-weight:700;}"
       << ".panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;overflow:auto;}"
       << ".panel.primary{border-top:4px solid var(--blue);}.panel h2{margin:0;font-size:17px;line-height:1.25;}"
       << ".panel h3{margin:16px 0 8px;font-size:14px;color:#334155;}"
       << ".result-grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(280px,0.65fr);gap:16px;align-items:start;}"
       << ".stat-grid,.winner-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;}"
       << ".run-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;}"
       << ".stat{background:#fbfcf9;border:1px solid var(--line);border-radius:6px;padding:10px 12px;}"
       << ".stat.good{border-left:4px solid var(--green);}.stat.warn{border-left:4px solid var(--amber);}"
       << ".stat-label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:0.06em;font-weight:700;}"
       << ".stat-value{font-size:20px;font-weight:750;line-height:1.2;margin-top:4px;}.stat-detail{color:var(--muted);font-size:12px;margin-top:4px;}"
       << ".config-line{white-space:pre-wrap;background:#f8fafc;border:1px solid var(--line);border-radius:6px;padding:10px;margin:12px 0 0;}"
       << ".grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:12px;}"
       << ".guide-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;}"
       << ".guide-grid div{background:#fbfcf9;border:1px solid var(--line);border-radius:6px;padding:10px;}"
       << ".guide-grid strong{display:block;margin-bottom:4px;}.guide-grid span{color:var(--muted);font-size:12px;}"
       << "details.panel summary{cursor:pointer;font-weight:700;list-style:none;}details.panel summary::-webkit-details-marker{display:none;}"
       << "details.panel summary::after{content:'show';float:right;color:var(--muted);font-size:12px;font-weight:600;}details[open].panel summary::after{content:'hide';}"
       << "table{width:100%;border-collapse:collapse;}caption{text-align:left;color:var(--muted);padding:4px 0 8px;}"
       << "th,td{padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top;}"
       << "th{text-align:left;color:#334155;font-weight:700;position:sticky;top:0;background:#f8fafc;}"
       << "td{color:#1f2937;}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;}"
       << ".empty-card{padding:18px;border:1px dashed #b6c0b0;border-radius:6px;color:var(--muted);background:#fbfcf9;}"
       << ".pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:3px 8px;font-size:12px;font-weight:700;}"
       << ".pill.neutral{background:#f8fafc;color:#334155;}.pill.good{background:#ecfdf5;color:var(--green);border-color:#bbf7d0;}"
       << ".pill.warn{background:#fffbeb;color:var(--amber);border-color:#fde68a;}"
       << "a{color:var(--blue);}.muted{color:var(--muted);}.note{font-size:12px;color:var(--muted);}"
       << "@media(max-width:900px){body{padding:12px;}.result-grid{grid-template-columns:1fr;}.grid2{grid-template-columns:1fr;}}"
       << "@media print{body{background:#fff;padding:0;}.panel,.hero{box-shadow:none;break-inside:avoid;}details.panel{display:block;}details.panel>*{display:block;}}"
       << "</style></head><body><div class=\"wrap\">";

  html << "<section class=\"hero\"><h1>Vortex v1 Model Selection Report</h1><div class=\"meta\">"
       << "<div><strong>Dataset:</strong> " << escape_html(options.dataset_path.string()) << "</div>"
       << "<div><strong>Generated:</strong> " << escape_html(format_local_timestamp()) << "</div>"
       << "<div><strong>JSON:</strong> " << escape_html(json_output_path.string()) << "</div>"
       << "<div><strong>Selector profile:</strong> requested="
       << escape_html(options.selector_profile) << ", effective="
       << escape_html(result.effective_selector_profile) << "</div>"
       << "<div><strong>Parallelism:</strong> "
       << (options.selector_parallelism == 0 ? std::string("auto") :
           std::to_string(options.selector_parallelism))
       << ", peak " << result.selector_peak_parallelism << " x "
       << result.selector_peak_threads_per_candidate << " threads</div>"
       << "</div></section>";

  html << build_best_config_panel(result);
  html << build_metric_guide();

  html << "<section class=\"panel\"><div class=\"section-head\">"
       << "<div><div class=\"eyebrow\">Run summary</div><h2>Search health and budget state</h2></div>"
       << status_pill(result.stopped_reason.empty() ? "unknown" : result.stopped_reason, stop_tone)
       << "</div><div class=\"run-grid\">"
       << stat_card("Evaluated", fmt_count(result.candidates.size()),
                    fmt_count(ok_count) + " ok, " + fmt_count(failed_count) + " guarded/failed")
       << stat_card("Phase-2 OK", fmt_count(phase2_count), "screening chart/sweep population")
       << stat_card("Recommended", fmt_count(result.recommended.size()),
                    fmt_count(result.pareto_front.size()) + " Pareto configs")
       << stat_card("Search Time", fmt_double(result.elapsed_search_seconds, 2) + " s",
                    result.effective_max_search_seconds > 0
                        ? "budget " + fmt_count(result.effective_max_search_seconds) + " s"
                        : "no explicit budget")
       << stat_card("Model Trains", fmt_count(result.model_trains_started),
                    result.max_model_trains > 0
                        ? "budget " + fmt_count(result.max_model_trains)
                        : "no explicit train cap")
       << stat_card("Memory Budget",
                    result.effective_selector_memory_budget_bytes > 0
                        ? fmt_bytes(result.effective_selector_memory_budget_bytes)
                        : std::string("auto/off"),
                    result.selector_memory_budget_limited ? "parallelism was capped" : "not limiting")
       << stat_card("Train Cache", fmt_count(result.train_cache_hits) + " / " +
                    fmt_count(result.train_cache_misses), "hits / misses")
       << stat_card("Base-Rank Cache", fmt_count(result.eval_base_rank_cache_hits) + " / " +
                    fmt_count(result.eval_base_rank_cache_misses),
                    fmt_bytes(result.eval_base_rank_cache_memory_bytes) + " retained")
       << stat_card("Nearest Cache", fmt_count(result.eval_nearest_cache_hits) + " / " +
                    fmt_count(result.eval_nearest_cache_misses),
                    fmt_bytes(result.eval_nearest_cache_memory_bytes) + " retained")
       << stat_card("Training Lane Wait", fmt_ms(result.training_lane_wait_ms, 1) + " ms",
                    "serialized fresh training pressure")
       << "</div>";
  if (!result.effective_eval_windows.empty() || !result.effective_eval_node_counts.empty()) {
    html << "<p class=\"note\">Evaluation windows: "
         << escape_html(format_list(result.effective_eval_windows, 8))
         << ". Overlay node counts: "
         << escape_html(format_list(result.effective_eval_node_counts, 8)) << ".</p>";
  }
  if (options.recall_target.has_value()) {
    html << "<p class=\"note\">Recall target " << escape_html(fmt_double(*options.recall_target, 4))
         << ": " << result.recall_target_met_recommended << " / "
         << result.recommended.size() << " recommended configs met the target.</p>";
  }
  html << "</section>";

  html << "<section class=\"panel\"><div class=\"section-head\">"
       << "<div><div class=\"eyebrow\">Recommendations</div><h2>Recommendation Roles</h2></div>"
       << "</div>"
       << build_recommendation_roles_table(result) << "</section>";

  html << "<section class=\"panel\"><div class=\"section-head\">"
       << "<div><div class=\"eyebrow\">Selected configs</div><h2>Top Recommended Configs</h2></div>"
       << "</div>"
       << build_metrics_table(recommended_ptrs, 32) << "</section>";

  html << "<section class=\"panel\"><div class=\"section-head\">"
       << "<div><div class=\"eyebrow\">Recall window curve</div><h2>Recall Curves By Recommendation Role</h2></div>"
       << "</div>"
       << build_recall_curve_table(result) << "</section>";

  html << "<section class=\"grid2\">"
       << "<div class=\"panel\"><div class=\"section-head\"><div><div class=\"eyebrow\">Figure 1</div><h2>Recall vs Latency (Phase-2 Screening)</h2></div></div>"
       << build_scatter_svg("Recall vs Latency (Screening)", "Latency Avg (ms)", "Recall@K in Window",
                            recall_vs_latency, std::make_pair(0.0, 1.0))
       << "</div>"
       << "<div class=\"panel\"><div class=\"section-head\"><div><div class=\"eyebrow\">Figure 2</div><h2>Recall vs Model Size (Phase-2 Screening)</h2></div></div>"
       << build_scatter_svg("Recall vs Model Size (Screening)", "Model Size (KiB)", "Recall@K in Window",
                            recall_vs_size, std::make_pair(0.0, 1.0))
       << "</div>"
       << "</section>";

  html << "<section class=\"grid2\">"
       << "<div class=\"panel\"><div class=\"section-head\"><div><div class=\"eyebrow\">Figure 3</div><h2>Score vs target_skeleton (Phase-2 Screening)</h2></div></div>"
       << build_scatter_svg("Objective Score vs target_skeleton (Screening)", "target_skeleton",
                            "Objective Score", score_vs_skeleton)
       << "</div>"
       << "<div class=\"panel\"><div class=\"section-head\"><div><div class=\"eyebrow\">Skeleton sweep</div><h2>target_skeleton Screening Summary</h2></div></div>"
       << build_skeleton_summary_table(phase2_ok)
       << "</div>"
       << "</section>";

  html << "<section class=\"panel\"><div class=\"section-head\">"
       << "<div><div class=\"eyebrow\">Skeleton benchmark</div><h2>Skeleton Sweep Screening Benchmarks</h2></div>"
       << "</div>"
       << build_skeleton_benchmark_table(
              phase2_ok,
              result.dataset_count,
              options.target_skeleton_percentages)
       << "<p class=\"note\">These phase-2 screening rows use selector search limits "
       << "(base="
       << (options.eval_base_limit == 0 ? std::string("full") : fmt_count(options.eval_base_limit))
       << ", query="
       << (options.eval_query_limit == 0 ? std::string("full") : fmt_count(options.eval_query_limit))
       << "). Selected-best and recommendation-role rows above are the calibrated "
          "artifact candidates; benchmark-pack report.html contains generated-code "
          "validation.</p>"
       << "</section>";

  html << "<details class=\"panel\"><summary>Pareto Front (Top 24)</summary>"
       << "<p class=\"note\">Secondary view for the full quality/cost tradeoff.</p>"
       << build_metrics_table(pareto_ptrs, 24)
       << "</details>";

  html << "<details class=\"panel\"><summary>Selector Context Telemetry</summary>"
       << "<p class=\"note\">Budget, beam, and per-skeleton search details. Kept collapsed to keep the report focused on selected configs.</p>"
       << build_context_stats_table(result.context_stats) << "</details>";

  html << "<details class=\"panel\"><summary>Base-Rank Cache Miss Attribution</summary>"
       << "<p class=\"note\">Diagnostics for dominant-cost analysis. Aggregates identify expensive phase/skeleton/K families before full config detail.</p>"
       << "<h3>Aggregated by phase / skeleton / K</h3>"
       << build_base_rank_aggregate_table(result.eval_base_rank_cache_attribution, 12)
       << "<h3>Top full config rows</h3>"
       << build_base_rank_attribution_table(result.eval_base_rank_cache_attribution, 16)
       << "</details>";

  html << "<div class=\"note\">Objective scores are phase-local: compare scores only within the "
          "same phase. Score charts and skeleton benchmarks above use phase-2 screening "
          "candidates; final artifact correctness is established by final-calibration roles "
          "and generated-code evaluation.</div>";
  html << "</div></body></html>";
  return html.str();
}
void print_ranked_item(std::size_t rank, const vortex::SelectorCandidateMetrics& item) {
  std::cout << "  [" << rank << "] score=" << fmt_double(item.objective_score, 6)
            << " quality=" << fmt_double(item.locality_quality_score, 6)
            << " match=" << fmt_double(item.overlay_match_score, 6)
            << " recall=" << fmt_double(item.recall_at_k_in_window, 6)
            << " auc=" << fmt_double(item.recall_auc_log_window, 6)
            << " node_locality=" << fmt_double(item.node_locality_score, 6)
            << " match_tail_p05=" << fmt_double(item.query_overlay_match_p05, 6)
            << " tail_p05=" << fmt_double(item.query_recall_p05, 6)
            << " rank_norm=" << fmt_double(item.mean_rank_distance_norm, 6)
            << " rank_p95=" << fmt_double(item.rank_distance_norm_p95, 6)
            << " lat_avg_ms=" << fmt_double(item.latency_avg_ms, 6)
            << " p95_ms=" << fmt_double(item.latency_p95_ms, 6)
            << " size=" << item.model_size_bytes << " (" << fmt_bytes(item.model_size_bytes) << ")"
            << " train_ms=" << fmt_double(item.train_time_ms, 3) << "\n";
  if (item.min_window_for_recall_target > 0) {
    std::cout << "      min_window_for_recall_target="
              << item.min_window_for_recall_target << "\n";
  }
  std::cout << "      " << format_config(item.config) << "\n";
}

void print_role_item(const char* role,
                     const char* reason,
                     const std::optional<vortex::SelectorCandidateMetrics>& item) {
  if (!item.has_value()) {
    return;
  }
  std::cout << "  " << role << " (" << reason << ")\n";
  print_ranked_item(1, *item);
}


} // namespace

std::filesystem::path default_html_report_path(const std::filesystem::path& json_output_path) {
  if (json_output_path.extension() == ".json") {
    return json_output_path.parent_path() / (json_output_path.stem().string() + ".html");
  }
  return json_output_path.string() + ".html";
}

void write_html_report(const std::filesystem::path& report_path,
                       const vortex::SelectorResult& result,
                       const vortex::SelectorOptions& options,
                       const std::filesystem::path& json_output_path) {
  if (!report_path.parent_path().empty()) {
    std::filesystem::create_directories(report_path.parent_path());
  }
  std::ofstream out(report_path);
  if (!out) {
    throw std::runtime_error("Failed to open html report output path: " + report_path.string());
  }
  out << build_html_report(result, options, json_output_path);
}

void print_summary(const vortex::SelectorResult& result,
                   const vortex::SelectorOptions& options) {
  std::cout << "vortex_model_selector results\n";
  std::cout << "  evaluated_candidates: " << result.candidates.size() << "\n";
  std::cout << "  pareto_front_count: " << result.pareto_front.size() << "\n";
  std::cout << "  recommended_count: " << result.recommended.size() << "\n";
  if (result.dataset_count > 0) {
    std::cout << "  dataset_count: " << result.dataset_count << "\n";
  }
  std::cout << "  selector_profile: requested=" << options.selector_profile
            << ", effective=" << result.effective_selector_profile << "\n";
  if (result.target_skeleton_contexts_total > 0) {
    std::cout << "  target_skeleton_contexts: selected_unique="
              << result.target_skeleton_contexts_selected
              << ", unique=" << result.target_skeleton_contexts_unique
              << ", requested=" << result.target_skeleton_contexts_requested
              << ", deduplicated=" << result.target_skeleton_contexts_deduplicated
              << "\n";
  }
  if (result.max_model_trains > 0) {
    std::cout << "  model_train_budget: started=" << result.model_trains_started << "/"
              << result.max_model_trains
              << ", budget_hit=" << (result.model_train_budget_hit ? "true" : "false") << "\n";
  }
  std::cout << "  train_cache: hits=" << result.train_cache_hits
            << ", misses=" << result.train_cache_misses
            << ", cache_wait_ms=" << fmt_double(result.train_cache_wait_ms, 3)
            << ", training_lane_wait_ms=" << fmt_double(result.training_lane_wait_ms, 3)
            << "\n";
  std::cout << "  training_base_cache: hits=" << result.training_base_cache_hits
            << ", misses=" << result.training_base_cache_misses
            << ", build_ms=" << fmt_double(result.training_base_cache_build_ms, 3)
            << ", memory_bytes=" << result.training_base_cache_memory_bytes << "\n";
  std::cout << "  training_order_cache: hits=" << result.training_order_cache_hits
            << ", misses=" << result.training_order_cache_misses
            << ", build_ms=" << fmt_double(result.training_order_cache_build_ms, 3)
            << ", memory_bytes=" << result.training_order_cache_memory_bytes << "\n";
  std::cout << "  training_cdf_cache: hits=" << result.training_cdf_cache_hits
            << ", misses=" << result.training_cdf_cache_misses
            << ", failures=" << result.training_cdf_cache_failures
            << ", wait_ms=" << fmt_double(result.training_cdf_cache_wait_ms, 3)
            << ", build_ms=" << fmt_double(result.training_cdf_cache_build_ms, 3)
            << ", memory_bytes=" << result.training_cdf_cache_memory_bytes << "\n";
  if (result.candidate_error_count > 0) {
    std::cout << "  candidate_errors: total=" << result.candidate_error_count
              << ", launch_guard=" << result.candidate_error_launch_guard_count
              << ", training_base=" << result.candidate_error_training_base_count
              << ", centroid_order=" << result.candidate_error_centroid_order_count
              << ", cdf_fit=" << result.candidate_error_cdf_fit_count
              << ", eval=" << result.candidate_error_eval_count
              << ", unknown=" << result.candidate_error_unknown_count << "\n";
  }
  std::cout << "  train_stage_ms_sum: total=" << fmt_double(result.train_time_ms_sum, 3)
            << ", gather=" << fmt_double(result.train_gather_skeleton_ms_sum, 3)
            << ", cluster_assign=" << fmt_double(result.train_cluster_assign_ms_sum, 3)
            << ", assign=" << fmt_double(result.train_assign_centroids_ms_sum, 3)
            << ", order=" << fmt_double(result.train_centroid_order_ms_sum, 3)
            << ", range=" << fmt_double(result.train_range_alloc_ms_sum, 3)
            << ", cdf=" << fmt_double(result.train_cdf_fit_ms_sum, 3)
            << ", assembly=" << fmt_double(result.train_model_assembly_ms_sum, 3)
            << "\n";
  std::cout << "  scheduler_cache_states: ready=" << result.scheduler_ready_cached_candidates
            << ", in_flight=" << result.scheduler_in_flight_candidates
            << ", missing=" << result.scheduler_missing_candidates
            << ", single_admission_batches="
            << result.scheduler_single_admission_batches << "\n";
  if (result.effective_max_search_seconds > 0 || options.max_search_seconds > 0) {
    std::cout << "  search_time_budget: requested=" << options.max_search_seconds
              << "s, effective=" << result.effective_max_search_seconds
              << "s, elapsed=" << fmt_double(result.elapsed_search_seconds, 3)
              << "s, budget_hit=" << (result.search_time_budget_hit ? "true" : "false") << "\n";
  } else {
    std::cout << "  search_time_budget: disabled, elapsed="
              << fmt_double(result.elapsed_search_seconds, 3) << "s\n";
  }
  if (result.final_calibration_reserve_seconds > 0.0) {
    std::cout << "  final_calibration_reserve: "
              << fmt_double(result.final_calibration_reserve_seconds, 3)
              << "s, hit=" << (result.final_calibration_reserve_hit ? "true" : "false")
              << "\n";
  }
  std::cout << "  stopped_reason: "
            << (result.stopped_reason.empty() ? "unknown" : result.stopped_reason)
            << "\n";
  std::cout << "  plateau_rounds: " << result.plateau_rounds
            << ", best_quality=" << fmt_double(result.best_quality, 6)
            << ", best_margin=" << fmt_double(result.best_margin, 6) << "\n";
  if (result.target_skeleton_contexts_total > 0) {
    std::cout << "  context_progress: evaluated="
              << result.target_skeleton_contexts_evaluated << "/"
              << result.target_skeleton_contexts_selected
              << ", skipped_low_potential="
              << result.contexts_skipped_low_potential << "\n";
  }
  std::cout << "  effective_weights: recall=" << fmt_double(result.effective_weights.recall, 3)
            << ", rank=" << fmt_double(result.effective_weights.rank_distance, 3)
            << ", latency=" << fmt_double(result.effective_weights.latency, 3)
            << ", model_size=" << fmt_double(result.effective_weights.model_size, 3)
            << ", train_time=" << fmt_double(result.effective_weights.train_time, 3) << "\n";
  if (!result.effective_eval_windows.empty()) {
    std::cout << "  eval_windows: " << format_list(result.effective_eval_windows, 16) << "\n";
  }
  if (!result.effective_eval_node_counts.empty()) {
    std::cout << "  eval_node_counts: "
              << format_list(result.effective_eval_node_counts, 16) << "\n";
  }
  if (options.recall_target.has_value()) {
    std::cout << "  recall_target: " << fmt_double(*options.recall_target, 4) << "\n";
    std::cout << "  recall_target_met_by_best: "
              << (result.recall_target_met ? "true" : "false") << "\n";
    std::cout << "  recall_target_met_recommended: " << result.recall_target_met_recommended
              << "/" << result.recommended.size() << "\n";
  }
  std::cout << "  skeleton_capacity_scaling: "
            << (options.skeleton_capacity_scaling ? "enabled" : "disabled") << "\n";
  std::cout << "  selector_capacity_policy: scale_aware_recall_first_config_families\n";
  std::cout << "  selector_objective: primary=locality_quality_score"
            << ", deployment=overlay_match_score"
            << ", compatibility=node_locality_score"
            << ", recall_signal=recall_auc_log_window\n";
  std::cout << "  advanced_search: "
            << (options.advanced_search ? "enabled" : "disabled")
            << " (beam_width=" << options.beam_width
            << ", beam_rounds=" << options.beam_rounds
            << ", hill_climb_steps=" << options.hill_climb_steps
            << ", beam_neighbor_limit=" << options.beam_neighbor_limit
            << ")\n";
  std::cout << "  optimize_for_recall: "
            << (options.optimize_for_recall ? "enabled" : "disabled")
            << " (max_recall_refine_rounds=" << options.max_recall_refine_rounds << ")\n";
  std::cout << "  adaptive_stop: max_stagnant_contexts="
            << options.max_stagnant_contexts
            << ", min_quality_improvement="
            << fmt_double(options.min_quality_improvement, 6)
            << ", min_recall_improvement="
            << fmt_double(options.min_recall_improvement, 6) << "\n";
  if (options.advanced_search) {
    std::cout << "  strategy_refine: rounds=" << result.strategy_refine_rounds
              << ", evaluated=" << result.strategy_refine_evaluated
              << ", frontier=" << result.strategy_frontier_candidates
              << ", compacted=" << result.strategy_compacted_candidates
              << ", attribution_guided_compaction="
              << (options.attribution_guided_compaction ? "enabled" : "disabled")
              << "\n";
  }
  if (options.optimize_for_recall) {
    std::cout << "  recall_refine: rounds=" << result.recall_refine_rounds
              << ", evaluated=" << result.recall_refine_evaluated << "\n";
  }
  if (options.final_calibration_count > 0) {
    std::cout << "  final_calibration: requested=" << options.final_calibration_count
              << ", seeded=" << result.final_calibration_seed_count
              << ", full_limit=" << result.effective_final_calibration_full_count
              << ", screen_seed_limit="
              << result.effective_final_calibration_screen_candidate_count
              << ", screen_evaluated="
              << result.final_calibration_screen_evaluated
              << ", screen_promoted="
              << result.final_calibration_screen_promoted
              << " (roles=" << result.final_calibration_role_promoted
              << ", recall_challengers="
              << result.final_calibration_recall_promoted << ")"
              << ", evaluated=" << result.final_calibration_evaluated
              << ", reused=" << result.final_calibration_reused
              << ", progressive_probes="
              << result.final_calibration_progressive_probes.size()
              << ", screen_query_limit="
              << result.effective_final_calibration_screen_query_limit
              << ", screen_status="
              << result.final_calibration_screen_status
              << ", query_limit="
              << (options.final_calibration_query_limit > 0
                      ? options.final_calibration_query_limit
                      : options.eval_query_limit)
              << "\n";
    if (!result.final_calibration_progressive_probes.empty()) {
      const auto& probe = result.final_calibration_progressive_probes.back();
      std::cout << "    calibration_stability: " << probe.name
                << ", probe_base=" << probe.probe_base_count
                << ", final_base=" << probe.final_base_count
                << ", peak_match="
                << (probe.peak_recall_match ? "yes" : "no")
                << ", matched_roles="
                << probe.matched_role_count << "/4"
                << ", all_roles_match="
                << (probe.peak_recall_match && probe.knee_match &&
                            probe.fast_match && probe.small_match
                        ? "yes"
                        : "no")
                << "\n";
    }
  }
  if (result.eval_nearest_cache_hits > 0 ||
      result.eval_nearest_cache_misses > 0) {
    std::cout << "  eval_nearest_cache: hits="
              << result.eval_nearest_cache_hits
              << ", misses=" << result.eval_nearest_cache_misses
              << ", inflight_bypasses="
              << result.eval_nearest_cache_inflight_bypasses
              << ", prewarm_requests="
              << result.eval_nearest_cache_prewarm_requests
              << ", prewarm_ms="
              << fmt_double(result.eval_nearest_cache_prewarm_ms, 3)
              << ", prewarm_threads="
              << result.eval_nearest_cache_prewarm_threads
              << ", wait_ms=" << fmt_double(result.eval_nearest_cache_wait_ms, 3)
              << " (sampled="
              << fmt_double(result.eval_nearest_cache_sampled_wait_ms, 3)
              << ", calibration="
              << fmt_double(result.eval_nearest_cache_calibration_wait_ms, 3)
              << ")"
              << ", build_ms=" << fmt_double(result.eval_nearest_cache_build_ms, 3)
              << " (sampled="
              << fmt_double(result.eval_nearest_cache_sampled_build_ms, 3)
              << ", calibration="
              << fmt_double(result.eval_nearest_cache_calibration_build_ms, 3)
              << ")"
              << ", entries=" << result.eval_nearest_cache_entries
              << ", memory=" << result.eval_nearest_cache_memory_bytes
              << " (" << fmt_bytes(result.eval_nearest_cache_memory_bytes) << ")"
              << ", sampled_budget="
              << fmt_bytes(result.eval_nearest_cache_sampled_budget_bytes)
              << ", sampled_memory="
              << fmt_bytes(result.eval_nearest_cache_sampled_memory_bytes)
              << ", calibration_memory="
              << fmt_bytes(result.eval_nearest_cache_calibration_memory_bytes)
              << ", evictions=" << result.eval_nearest_cache_evictions
              << ", evicted="
              << fmt_bytes(result.eval_nearest_cache_evicted_bytes)
              << "\n";
  }
  if (result.eval_base_rank_cache_hits > 0 ||
      result.eval_base_rank_cache_misses > 0) {
    std::cout << "  eval_base_rank_cache: hits="
              << result.eval_base_rank_cache_hits
              << ", misses=" << result.eval_base_rank_cache_misses
              << ", wait_ms=" << fmt_double(result.eval_base_rank_cache_wait_ms, 3)
              << ", build_ms=" << fmt_double(result.eval_base_rank_cache_build_ms, 3)
              << ", nearest_ms="
              << fmt_double(result.eval_base_rank_cache_nearest_ms, 3)
              << ", hash_ms=" << fmt_double(result.eval_base_rank_cache_hash_ms, 3)
              << ", sort_ms=" << fmt_double(result.eval_base_rank_cache_sort_ms, 3)
              << ", rank_index_ms="
              << fmt_double(result.eval_base_rank_cache_rank_index_ms, 3)
              << ", overhead_ms="
              << fmt_double(result.eval_base_rank_cache_overhead_ms, 3)
              << ", entries=" << result.eval_base_rank_cache_entries
              << ", memory=" << result.eval_base_rank_cache_memory_bytes
              << " (" << fmt_bytes(result.eval_base_rank_cache_memory_bytes) << ")"
              << ", sampled_budget="
              << fmt_bytes(result.eval_base_rank_cache_sampled_budget_bytes)
              << ", sampled_memory="
              << fmt_bytes(result.eval_base_rank_cache_sampled_memory_bytes)
              << ", calibration_memory="
              << fmt_bytes(result.eval_base_rank_cache_calibration_memory_bytes)
              << ", evictions=" << result.eval_base_rank_cache_evictions
              << ", evicted="
              << fmt_bytes(result.eval_base_rank_cache_evicted_bytes)
              << "\n";
    auto aggregates = aggregate_base_rank_attribution(result.eval_base_rank_cache_attribution);
    if (!aggregates.empty()) {
      const auto& top = aggregates.front();
      std::cout << "    top_hash_family: phase=" << top.phase
                << ", target_skeleton=" << top.target_skeleton
                << ", skeleton_nodes=" << top.skeleton_node_count
                << ", K=" << top.K
                << ", hits=" << top.hits
                << ", misses=" << top.misses
                << ", hash_ms=" << fmt_double(top.hash_ms, 3)
                << ", retained=" << fmt_bytes(top.retained_memory_bytes)
                << "\n";
    }
  }
  if (result.post_exploration_evaluated > 0 || options.post_exploration_exploitation) {
    std::cout << "  post_exploration: "
              << (options.post_exploration_exploitation ? "on" : "off")
              << ", requested="
              << (options.post_exploration_candidates > 0
                      ? std::to_string(options.post_exploration_candidates)
                      : std::string("auto"))
              << ", evaluated=" << result.post_exploration_evaluated << "\n";
  }
  if (result.effective_selector_memory_budget_bytes > 0) {
    std::cout << "  memory_budget="
              << fmt_bytes(result.effective_selector_memory_budget_bytes)
              << ", peak_parallelism=" << result.selector_peak_parallelism
              << ", peak_threads_per_candidate="
              << result.selector_peak_threads_per_candidate
              << ", memory_limited="
              << (result.selector_memory_budget_limited ? "true" : "false") << "\n";
  }
  if (result.auto_dataset_weighting_applied) {
    std::cout << "  auto_dataset_weighting_applied: true\n";
  }
  if (!result.best.has_value()) {
    std::cout << "  best: none\n";
    return;
  }

  const auto& best = *result.best;
  std::cout << "Best\n";
  print_ranked_item(1, best);

  std::cout << "Recommendation Roles\n";
  print_role_item("peak_recall", "highest locality quality / recall curve",
                  result.recommended_peak_recall);
  print_role_item("knee", "best quality-cost tradeoff", result.recommended_knee);
  print_role_item("fast", "lowest latency near peak quality", result.recommended_fast);
  print_role_item("small", "smallest model near peak quality", result.recommended_small);

  if (!result.recommended.empty()) {
    std::cout << "Recommended\n";
    for (std::size_t i = 0; i < result.recommended.size(); ++i) {
      print_ranked_item(i + 1, result.recommended[i]);
    }
  }

  if (!result.pareto_front.empty()) {
    std::cout << "Pareto Front\n";
    std::size_t show = std::min<std::size_t>(result.pareto_front.size(), 8);
    for (std::size_t i = 0; i < show; ++i) {
      print_ranked_item(i + 1, result.pareto_front[i]);
    }
    if (result.pareto_front.size() > show) {
      std::cout << "  ... (" << (result.pareto_front.size() - show)
                << " more pareto configs)\n";
    }
  }
}

} // namespace vortex::tools::model_selector_report
