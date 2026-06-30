#include "vortex_v1/model_selector.h"

#include "selector_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vortex {

using namespace selector_internal;
using Clock = SelectorClock;

std::vector<SelectorCandidateConfig> build_phase1_selector_candidates(const SelectorOptions& raw) {
  if (raw.nsw_path.empty()) {
    throw std::runtime_error("build_phase1_selector_candidates requires nsw_path");
  }
  NswCsr csr = NswCsr::read(raw.nsw_path);
  SelectorOptions options = normalize_selector_options(raw, csr.node_ids.size());
  return build_phase1_candidates_internal(options, csr.node_ids.size());
}

namespace {

std::vector<uint64_t> ranking_eval_windows(
    const std::vector<SelectorCandidateMetrics>& metrics,
    const std::vector<uint64_t>& fallback) {
  std::vector<uint64_t> windows;
  for (const auto& item : metrics) {
    if (!item.ok) {
      continue;
    }
    windows.insert(windows.end(), item.recall_windows.begin(), item.recall_windows.end());
  }
  if (windows.empty()) {
    windows = fallback;
  }
  dedup_sort(&windows);
  return windows;
}

bool any_candidate_meets_recall_target(const std::vector<SelectorCandidateMetrics>& metrics,
                                       double recall_target) {
  for (const auto& item : metrics) {
    if (item.ok && meets_recall_target(item, recall_target)) {
      return true;
    }
  }
  return false;
}

bool should_use_progressive_phase1_k_guard(
    const SelectorOptions& options,
    const std::vector<SelectorCandidateConfig>& candidates) {
  if (candidates.empty() || options.dataset_count_hint < 500000) {
    return false;
  }
  std::set<uint32_t> unique_k;
  for (const auto& candidate : candidates) {
    unique_k.insert(candidate.K);
  }
  return unique_k.size() > 2;
}

SelectorCandidateMetrics make_progressive_k_guard_metric(
    const SelectorCandidateConfig& config,
    const NswSelectorContext& context,
    const PhaseLimits& phase) {
  SelectorCandidateMetrics metric;
  metric.config = config;
  metric.phase = phase.name;
  metric.nsw_path = context.display_nsw_path;
  metric.skeleton_node_count = static_cast<uint32_t>(context.csr.node_ids.size());
  metric.skeleton_layer = context.csr.layer;
  metric.ok = false;
  metric.error =
      "selector progressive-K guard skipped high-K phase1 candidate after lower-K probes showed no objective gain";
  metric.error_stage = "launch_guard";
  return metric;
}

std::vector<SelectorCandidateMetrics> evaluate_phase1_candidates_progressive_k(
    const std::vector<SelectorCandidateConfig>& candidates,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    const EvalDataset& eval,
    const PhaseLimits& phase,
    TrainCache* cache,
    SearchController* search) {
  if (!should_use_progressive_phase1_k_guard(options, candidates)) {
    return evaluate_candidates(candidates, options, context, eval, phase, cache, search);
  }

  std::vector<SelectorCandidateMetrics> metrics(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    metrics[i].config = candidates[i];
    metrics[i].phase = phase.name;
    metrics[i].nsw_path = context.display_nsw_path;
    metrics[i].skeleton_node_count = static_cast<uint32_t>(context.csr.node_ids.size());
    metrics[i].skeleton_layer = context.csr.layer;
  }

  std::set<uint32_t> unique_k;
  for (const auto& candidate : candidates) {
    unique_k.insert(candidate.K);
  }

  double best_seen_guard_score = -1.0;
  double previous_k_best_guard_score = -1.0;
  uint32_t evaluated_k_count = 0;
  double drop_tolerance =
      options.optimize_for_recall
          ? std::max(0.030, options.min_recall_improvement * 12.0)
          : std::max(0.020, options.min_quality_improvement * 8.0);

  for (uint32_t k : unique_k) {
    if (evaluated_k_count >= 2 &&
        previous_k_best_guard_score >= 0.0 &&
        previous_k_best_guard_score + drop_tolerance < best_seen_guard_score) {
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].K >= k && metrics[i].error.empty() && !metrics[i].ok) {
          metrics[i] = make_progressive_k_guard_metric(candidates[i], context, phase);
        }
      }
      break;
    }

    std::vector<std::size_t> group_indices;
    std::vector<SelectorCandidateConfig> group;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (candidates[i].K == k) {
        group_indices.push_back(i);
        group.push_back(candidates[i]);
      }
    }
    if (group.empty()) {
      continue;
    }

    std::vector<SelectorCandidateMetrics> group_metrics =
        evaluate_candidates(group, options, context, eval, phase, cache, search);
    for (std::size_t i = 0; i < group_indices.size() && i < group_metrics.size(); ++i) {
      metrics[group_indices[i]] = std::move(group_metrics[i]);
    }

    double current_k_best_guard_score = -1.0;
    for (std::size_t index : group_indices) {
      if (metrics[index].ok) {
        double guard_score = options.optimize_for_recall
            ? metrics[index].recall_at_k_in_window
            : selector_quality_score(metrics[index]);
        current_k_best_guard_score = std::max(current_k_best_guard_score,
                                              guard_score);
      }
    }
    if (current_k_best_guard_score >= 0.0) {
      previous_k_best_guard_score = current_k_best_guard_score;
      best_seen_guard_score = std::max(best_seen_guard_score,
                                       current_k_best_guard_score);
      evaluated_k_count += 1;
    }

    if (search != nullptr && !search->can_launch_more_work()) {
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (metrics[i].error.empty() && !metrics[i].ok) {
          metrics[i].ok = false;
          metrics[i].error = "selector search time budget reached";
          metrics[i].error_stage = "launch_guard";
        }
      }
      break;
    }
  }

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (metrics[i].error.empty() && !metrics[i].ok) {
      metrics[i] = make_progressive_k_guard_metric(candidates[i], context, phase);
    }
  }
  return metrics;
}

double estimate_final_calibration_reserve_seconds(const SelectorOptions& options,
                                                  std::size_t base_count,
                                                  std::size_t query_count,
                                                  uint32_t max_search_seconds) {
  if (max_search_seconds == 0 || options.final_calibration_count == 0) {
    return 0.0;
  }
  uint32_t full_count = resolve_final_calibration_full_count(options, base_count);
  if (full_count == 0) {
    return 0.0;
  }

  uint32_t bounded_query_count = static_cast<uint32_t>(
      std::min<std::size_t>(query_count, std::numeric_limits<uint32_t>::max()));
  uint32_t final_query_limit = options.final_calibration_query_limit > 0
      ? options.final_calibration_query_limit
      : options.eval_query_limit;
  if (final_query_limit == 0) {
    final_query_limit = bounded_query_count;
  }
  if (bounded_query_count > 0) {
    final_query_limit = std::min(final_query_limit, bounded_query_count);
  }

  uint32_t screen_query_limit = options.final_calibration_count > full_count
      ? resolve_final_calibration_screen_query_limit(options, query_count)
      : 0;
  double base_scale = std::clamp(static_cast<double>(base_count) / 1'000'000.0,
                                 0.05,
                                 10.0);

  double reserve = 8.0;
  reserve += base_scale * (6.0 * static_cast<double>(full_count));
  reserve += std::min(60.0, 0.02 * static_cast<double>(final_query_limit));
  if (screen_query_limit > 0) {
    reserve += base_scale * 4.0;
    reserve += std::min(30.0, 0.01 * static_cast<double>(screen_query_limit));
  }

  double min_reserve = base_count >= 500'000 ? 45.0 : 10.0;
  double max_reserve =
      std::max(min_reserve, std::min(300.0, 0.30 * static_cast<double>(max_search_seconds)));
  return std::clamp(reserve, min_reserve, max_reserve);
}

uint32_t adapt_context_train_budget_for_deadline(uint32_t context_budget,
                                                 TrainCache* cache,
                                                 const SearchController& search) {
  if (context_budget == 0 || cache == nullptr || !search.deadline.has_value()) {
    return context_budget;
  }
  double observed_train_seconds = 0.0;
  {
    std::lock_guard<std::mutex> lock(cache->mutex);
    observed_train_seconds = cache->max_observed_train_seconds;
  }
  if (observed_train_seconds <= 0.0) {
    return context_budget;
  }

  auto now = Clock::now();
  if (now >= *search.deadline) {
    return 1;
  }
  double remaining_seconds = static_cast<double>(
      std::chrono::duration_cast<std::chrono::milliseconds>(*search.deadline - now).count()) /
      1000.0;
  double usable_seconds = remaining_seconds - search.final_calibration_reserve_seconds;
  double guarded_train_seconds = observed_train_seconds * 1.25;
  if (usable_seconds <= guarded_train_seconds) {
    return std::min<uint32_t>(context_budget, 1);
  }
  uint32_t train_slots = static_cast<uint32_t>(usable_seconds / guarded_train_seconds);
  return std::max<uint32_t>(1, std::min(context_budget, train_slots));
}

bool eval_datasets_equivalent(const EvalDataset& lhs, const EvalDataset& rhs) {
  return lhs.base.dim == rhs.base.dim &&
         lhs.base.count == rhs.base.count &&
         lhs.base.values == rhs.base.values &&
         lhs.query.dim == rhs.query.dim &&
         lhs.query.count == rhs.query.count &&
         lhs.query.values == rhs.query.values &&
         lhs.truth_labels == rhs.truth_labels &&
         lhs.self_base_indices == rhs.self_base_indices &&
         lhs.truth_k == rhs.truth_k &&
         lhs.window == rhs.window &&
         lhs.windows == rhs.windows &&
         lhs.node_counts == rhs.node_counts &&
         lhs.exclude_self == rhs.exclude_self;
}

SelectorCandidateMetrics make_reused_final_calibration_metric(
    const SelectorCandidateMetrics& seed) {
  SelectorCandidateMetrics out = seed;
  out.phase = "final_calibration";
  out.eval_total_ms = 0.0;
  out.eval_hash_base_ms = 0.0;
  out.eval_sort_base_ms = 0.0;
  out.eval_rank_index_ms = 0.0;
  out.eval_query_ms = 0.0;
  out.eval_latency_ms = 0.0;
  out.eval_hash_base_threads = 1;
  out.eval_rank_index_threads = 1;
  return out;
}

std::string candidate_identity(const SelectorCandidateMetrics& item) {
  return item.nsw_path + "|" + std::to_string(item.config.target_skeleton) + "|" +
         selector_config_key(item.config);
}

bool attribution_matches_metric(const SelectorBaseRankCacheAttribution& row,
                                const SelectorCandidateMetrics& metric) {
  if (!row.nsw_path.empty() && !metric.nsw_path.empty() &&
      row.nsw_path != metric.nsw_path) {
    return false;
  }
  const auto& config = metric.config;
  return row.target_skeleton == config.target_skeleton &&
         row.K == config.K &&
         row.centroid_knn == config.centroid_knn &&
         row.cdf_model_spec == config.cdf_model_spec &&
         row.cdf_branching_factor == config.cdf_branching_factor &&
         row.enable_2opt == config.enable_2opt &&
         row.two_opt_iterations == config.two_opt_iterations &&
         row.enable_graph_centroid_order == config.enable_graph_centroid_order &&
         (row.skeleton_node_count == 0 ||
          metric.skeleton_node_count == 0 ||
          row.skeleton_node_count == metric.skeleton_node_count) &&
         (row.skeleton_layer < 0 ||
          metric.skeleton_layer < 0 ||
          row.skeleton_layer == metric.skeleton_layer);
}

void add_base_rank_attribution_role(
    std::vector<SelectorBaseRankCacheAttribution>* rows,
    const std::optional<SelectorCandidateMetrics>& role_metric,
    const std::string& role_name) {
  if (rows == nullptr || !role_metric.has_value()) {
    return;
  }
  for (auto& row : *rows) {
    if (attribution_matches_metric(row, *role_metric)) {
      row.roles.push_back(role_name);
      dedup_strings(&row.roles);
    }
  }
}

void annotate_base_rank_attribution_roles(SelectorResult* result) {
  if (result == nullptr) {
    return;
  }
  add_base_rank_attribution_role(&result->eval_base_rank_cache_attribution,
                                 result->recommended_peak_recall,
                                 "peak_recall");
  add_base_rank_attribution_role(&result->eval_base_rank_cache_attribution,
                                 result->recommended_knee,
                                 "knee");
  add_base_rank_attribution_role(&result->eval_base_rank_cache_attribution,
                                 result->recommended_fast,
                                 "fast");
  add_base_rank_attribution_role(&result->eval_base_rank_cache_attribution,
                                 result->recommended_small,
                                 "small");
}

bool same_recommendation_role(const std::optional<SelectorCandidateMetrics>& lhs,
                              const std::optional<SelectorCandidateMetrics>& rhs) {
  if (!lhs.has_value() || !rhs.has_value()) {
    return !lhs.has_value() && !rhs.has_value();
  }
  return candidate_identity(*lhs) == candidate_identity(*rhs);
}

SelectorCalibrationProbe make_final_calibration_probe(
    const std::string& name,
    const std::vector<SelectorCandidateMetrics>& probe_metrics,
    const EvalDataset& probe_eval,
    const std::vector<SelectorCandidateMetrics>& final_metrics,
    const EvalDataset& final_eval,
    const SelectorOptions& options) {
  SelectorRecommendationRoles probe_roles =
      choose_recommendation_roles(probe_metrics,
                                  options.size_budget_bytes,
                                  options.recall_target);
  SelectorRecommendationRoles final_roles =
      choose_recommendation_roles(final_metrics,
                                  options.size_budget_bytes,
                                  options.recall_target);

  SelectorCalibrationProbe probe;
  probe.name = name;
  probe.probe_base_count = static_cast<uint64_t>(probe_eval.base.count);
  probe.probe_query_count = static_cast<uint64_t>(probe_eval.query.count);
  probe.final_base_count = static_cast<uint64_t>(final_eval.base.count);
  probe.final_query_count = static_cast<uint64_t>(final_eval.query.count);
  probe.evaluated = count_ok_metrics(probe_metrics);
  probe.probe_peak_recall =
      probe_roles.peak_recall.has_value()
          ? probe_roles.peak_recall->recall_at_k_in_window
          : 0.0;
  probe.final_peak_recall =
      final_roles.peak_recall.has_value()
          ? final_roles.peak_recall->recall_at_k_in_window
          : 0.0;
  probe.peak_recall_match =
      same_recommendation_role(probe_roles.peak_recall, final_roles.peak_recall);
  probe.knee_match = same_recommendation_role(probe_roles.knee, final_roles.knee);
  probe.fast_match = same_recommendation_role(probe_roles.fast, final_roles.fast);
  probe.small_match = same_recommendation_role(probe_roles.small, final_roles.small);
  probe.matched_role_count =
      static_cast<uint32_t>(probe.peak_recall_match) +
      static_cast<uint32_t>(probe.knee_match) +
      static_cast<uint32_t>(probe.fast_match) +
      static_cast<uint32_t>(probe.small_match);
  return probe;
}

bool contains_candidate_identity(const std::vector<SelectorCandidateMetrics>& metrics,
                                 const SelectorCandidateMetrics& item) {
  std::string item_identity = candidate_identity(item);
  for (const auto& existing : metrics) {
    if (candidate_identity(existing) == item_identity) {
      return true;
    }
  }
  return false;
}

std::vector<SelectorCandidateMetrics> metrics_for_seed_identities(
    const std::vector<SelectorCandidateMetrics>& metrics,
    const std::vector<SelectorCandidateMetrics>& seeds) {
  std::vector<SelectorCandidateMetrics> out;
  out.reserve(std::min(metrics.size(), seeds.size()));
  for (const auto& seed : seeds) {
    std::string seed_identity = candidate_identity(seed);
    auto found = std::find_if(metrics.begin(),
                              metrics.end(),
                              [&](const SelectorCandidateMetrics& metric) {
                                return candidate_identity(metric) == seed_identity;
                              });
    if (found != metrics.end()) {
      out.push_back(*found);
    }
  }
  return out;
}

std::vector<SelectorCandidateMetrics> fill_promoted_calibration_seeds(
    std::vector<SelectorCandidateMetrics> promoted,
    const std::vector<SelectorCandidateMetrics>& fallback,
    uint32_t limit) {
  if (limit == 0) {
    return {};
  }
  for (const auto& seed : fallback) {
    if (promoted.size() >= limit) {
      break;
    }
    if (!contains_candidate_identity(promoted, seed)) {
      promoted.push_back(seed);
    }
  }
  if (promoted.size() > limit) {
    promoted.resize(limit);
  }
  return promoted;
}

struct CalibrationGroup {
  const NswSelectorContext* context = nullptr;
  std::vector<std::size_t> seed_indices;
  std::vector<SelectorCandidateConfig> configs;
};

std::vector<SelectorCandidateMetrics> evaluate_calibration_seeds(
    const std::vector<SelectorCandidateMetrics>& seeds,
    const std::vector<NswSelectorContext>& contexts,
    const SelectorOptions& options,
    const EvalDataset& eval,
    const PhaseLimits& limits,
    TrainCache* train_cache,
    SearchController* search_controller) {
  std::vector<CalibrationGroup> groups;
  groups.reserve(contexts.size());
  for (std::size_t seed_index = 0; seed_index < seeds.size(); ++seed_index) {
    const auto& seed = seeds[seed_index];
    const NswSelectorContext* context = find_context_for_metric(contexts, seed);
    if (context == nullptr) {
      continue;
    }
    auto group_it = std::find_if(groups.begin(),
                                 groups.end(),
                                 [&](const CalibrationGroup& group) {
                                   return group.context == context;
                                 });
    if (group_it == groups.end()) {
      CalibrationGroup group;
      group.context = context;
      groups.push_back(std::move(group));
      group_it = groups.end() - 1;
    }
    group_it->seed_indices.push_back(seed_index);
    group_it->configs.push_back(seed.config);
  }

  std::vector<std::optional<SelectorCandidateMetrics>> by_seed(seeds.size());
  for (const auto& group : groups) {
    if (!search_controller->can_launch_more_work()) {
      break;
    }
    SelectorOptions group_options = options;
    group_options.nsw_path = group.context->nsw_path;
    std::vector<SelectorCandidateMetrics> group_metrics =
        evaluate_candidates(group.configs,
                            group_options,
                            *group.context,
                            eval,
                            limits,
                            train_cache,
                            search_controller);
    std::size_t count = std::min(group_metrics.size(), group.seed_indices.size());
    for (std::size_t i = 0; i < count; ++i) {
      std::size_t seed_index = group.seed_indices[i];
      group_metrics[i].config.target_skeleton =
          seeds[seed_index].config.target_skeleton;
      by_seed[seed_index] = std::move(group_metrics[i]);
    }
  }

  std::vector<SelectorCandidateMetrics> out;
  out.reserve(seeds.size());
  for (auto& item : by_seed) {
    if (item.has_value()) {
      out.push_back(std::move(*item));
    }
  }
  return out;
}

} // namespace

SelectorResult select_vortex_models(const SelectorOptions& raw) {
  std::vector<SelectorPhaseTiming> timings;
  auto total_start = Clock::now();

  auto load_base_start = Clock::now();
  vector_io::VectorStorage<float> full_base = load_vectors_or_throw(raw.dataset_path);
  append_timing(&timings, "load_base_vectors", load_base_start);

  std::optional<vector_io::VectorStorage<float>> full_query;
  if (raw.query_path.has_value()) {
    if (*raw.query_path != raw.dataset_path) {
      auto load_query_start = Clock::now();
      full_query = load_vectors_or_throw(*raw.query_path);
      append_timing(&timings, "load_query_vectors", load_query_start);
    }
  }

  if (full_query.has_value() && full_query->dim != full_base.dim) {
    throw std::runtime_error("Query dim does not match dataset dim");
  }

  bool auto_weighting_applied = false;
  SelectorOptions effective_raw = raw;
  effective_raw.dataset_count_hint = static_cast<uint64_t>(full_base.count);
  effective_raw.weights = resolve_effective_weights(
      effective_raw, full_base.count, &auto_weighting_applied);
  std::string effective_profile =
      resolve_effective_selector_profile(effective_raw, full_base.count);
  effective_raw.selector_profile = effective_profile;
  uint32_t effective_max_search_seconds = resolve_search_time_budget_seconds(
      effective_raw, effective_profile, full_base.count);
  uint32_t effective_max_stagnant_contexts = 0;
  SearchController search_controller;
  if (effective_max_search_seconds > 0) {
    search_controller.deadline =
        search_controller.start_time + std::chrono::seconds(effective_max_search_seconds);
  }

  ScopedTempFiles temp_files;
  auto context_start = Clock::now();
  std::vector<NswSelectorContext> contexts = build_nsw_contexts(effective_raw, full_base, &temp_files);
  append_timing(&timings, "build_nsw_contexts", context_start);
  uint32_t requested_contexts = requested_context_count(contexts);
  effective_max_stagnant_contexts = resolve_stagnant_context_limit(
      effective_raw, effective_profile, contexts.size());
  std::size_t max_skeleton_count = 0;
  for (const auto& context : contexts) {
    max_skeleton_count = std::max<std::size_t>(max_skeleton_count, context.csr.node_ids.size());
  }
  SelectorOptions eval_options =
      normalize_selector_options(effective_raw, contexts.front().csr.node_ids.size());
  eval_options.nsw_path = contexts.front().nsw_path;
  search_controller.final_calibration_reserve_seconds =
      estimate_final_calibration_reserve_seconds(
          eval_options,
          full_base.count,
          full_query.has_value() ? full_query->count : full_base.count,
          effective_max_search_seconds);

  auto coarse_eval_start = Clock::now();
  EvalDataset coarse_eval = build_eval_dataset(full_base,
                                               full_query,
                                               eval_options.eval_k,
                                               eval_options.eval_window,
                                               eval_options.eval_window_values,
                                               eval_options.eval_node_count_values,
                                               eval_options.coarse_base_limit,
                                               eval_options.coarse_query_limit,
                                               eval_options.seed + 1001);
  append_timing(&timings, "build_coarse_eval", coarse_eval_start);

  auto full_eval_start = Clock::now();
  EvalDataset full_eval = build_eval_dataset(full_base,
                                            full_query,
                                            eval_options.eval_k,
                                            eval_options.eval_window,
                                            eval_options.eval_window_values,
                                            eval_options.eval_node_count_values,
                                               eval_options.eval_base_limit,
                                            eval_options.eval_query_limit,
                                            eval_options.seed + 2003);
  append_timing(&timings, "build_full_eval", full_eval_start);

  PhaseLimits phase1_limits;
  phase1_limits.name = "phase1";
  phase1_limits.base_limit = eval_options.coarse_base_limit;
  phase1_limits.query_limit = eval_options.coarse_query_limit;
  phase1_limits.latency_iterations = eval_options.coarse_latency_iterations;
  phase1_limits.latency_warmup = eval_options.coarse_latency_warmup;

  PhaseLimits phase2_limits;
  phase2_limits.name = "phase2";
  phase2_limits.base_limit = eval_options.eval_base_limit;
  phase2_limits.query_limit = eval_options.eval_query_limit;
  phase2_limits.latency_iterations = eval_options.latency_iterations;
  phase2_limits.latency_warmup = eval_options.latency_warmup;

  TrainCache train_cache;
  train_cache.dataset = &full_base;
  train_cache.shared_dataset_bytes =
      static_cast<uint64_t>(full_base.values.size()) * static_cast<uint64_t>(sizeof(float));
  train_cache.memory_budget_bytes = resolve_selector_memory_budget_bytes(effective_raw);
  train_cache.max_model_trains = resolve_model_train_budget(
      effective_raw, effective_profile, full_base.count, contexts.size());
  uint32_t post_exploration_candidate_count =
      resolve_post_exploration_candidate_count(eval_options, train_cache.max_model_trains);
  auto context_plan_start = Clock::now();
  ContextSelectionPlan context_plan = select_context_indices_for_full_search(
      contexts,
      effective_raw,
      coarse_eval,
      phase1_limits,
      full_base.count,
      max_skeleton_count,
      effective_profile,
      &train_cache,
      &search_controller);
  append_timing(&timings, "select_contexts", context_plan_start);
  std::vector<SelectorCandidateMetrics> all_phase1_metrics;
  std::vector<SelectorCandidateMetrics> all_phase2_metrics;
  uint32_t total_refine_evaluated = 0;
  uint32_t total_refine_rounds = 0;
  uint32_t total_frontier_candidates = 0;
  uint32_t total_compacted_candidates = 0;
  uint32_t recall_refine_evaluated = 0;
  uint32_t recall_refine_rounds = 0;
  uint32_t evaluated_contexts = 0;
  uint32_t contexts_skipped_low_potential = 0;
  uint32_t stagnant_contexts = 0;
  double best_recall_so_far = -1.0;
  double best_quality_so_far = -1.0;
  bool recall_target_met_so_far = false;
  std::string stopped_reason = "completed_all_contexts";
  std::vector<SelectorContextStats> context_stats;
  context_stats.reserve(context_plan.indices.size());
  auto context_search_start = Clock::now();
  for (std::size_t context_offset = 0; context_offset < context_plan.indices.size();
       ++context_offset) {
    std::size_t context_index = context_plan.indices[context_offset];
    if (!search_controller.can_launch_more_work()) {
      stopped_reason = "time_budget";
      break;
    }
    auto per_context_start = Clock::now();
    evaluated_contexts += 1;
    const auto& context = contexts[context_index];
    SelectorOptions context_options =
        normalize_selector_options(effective_raw, context.csr.node_ids.size());
    context_options.nsw_path = context.nsw_path;
    SelectorOptions options = apply_context_capacity_scaling(context_options,
                                                             context.csr.node_ids.size(),
                                                             max_skeleton_count);
    options = normalize_selector_options(options, context.csr.node_ids.size());
    options.nsw_path = context.nsw_path;

    uint32_t context_budget =
        fair_context_train_budget(&train_cache,
                                  context_plan.indices.size() - context_offset,
                                  options.phase1_keep,
                                  post_exploration_candidate_count);
    context_budget =
        adapt_context_train_budget_for_deadline(context_budget,
                                                &train_cache,
                                                search_controller);
    SelectorContextStats stats;
    stats.target_skeleton = context.target_skeleton;
    stats.requested_target_skeletons = context.requested_target_skeletons;
    stats.skeleton_node_count = static_cast<uint32_t>(context.csr.node_ids.size());
    stats.skeleton_layer = context.csr.layer;
    stats.effective_K_values = options.K_values;
    for (uint32_t requested_k : context_options.K_values) {
      if (std::find(options.K_values.begin(), options.K_values.end(), requested_k) ==
          options.K_values.end()) {
        stats.pruned_K_values.push_back(requested_k);
      }
    }
    stats.effective_centroid_knn_values = options.centroid_knn_values;
    stats.effective_cdf_branching_values = options.cdf_branching_values;
    if (!options.K_values.empty()) {
      uint32_t min_k = options.K_values.front();
      uint32_t max_k = options.K_values.back();
      if (max_k > 0) {
        stats.min_vectors_per_centroid =
            static_cast<double>(full_base.count) / static_cast<double>(max_k);
        stats.min_skeleton_nodes_per_centroid =
            static_cast<double>(context.csr.node_ids.size()) / static_cast<double>(max_k);
      }
      if (min_k > 0) {
        stats.max_vectors_per_centroid =
            static_cast<double>(full_base.count) / static_cast<double>(min_k);
        stats.max_skeleton_nodes_per_centroid =
            static_cast<double>(context.csr.node_ids.size()) / static_cast<double>(min_k);
      }
    }
    stats.context_model_train_budget = context_budget;
    stats.model_trains_started_before = read_model_trains_started(&train_cache);
    begin_context_train_budget(&train_cache, context_budget);

    bool stats_recorded = false;
    auto finish_context_stats = [&]() {
      if (stats_recorded) {
        return;
      }
      stats.model_trains_started_after = read_model_trains_started(&train_cache);
      stats.context_model_train_budget_hit = read_context_budget_hit(&train_cache);
      stats.elapsed_ms = elapsed_ms(per_context_start);
      context_stats.push_back(stats);
      end_context_train_budget(&train_cache);
      stats_recorded = true;
    };

    std::vector<SelectorCandidateConfig> phase1_candidates =
        build_phase1_candidates_internal(options, context.csr.node_ids.size());
    for (auto& config : phase1_candidates) {
      config.target_skeleton = context.target_skeleton;
    }
    phase1_candidates =
        fit_candidates_to_remaining_train_budget(phase1_candidates,
                                                 options,
                                                 context,
                                                 &train_cache);
    stats.phase1_candidates = static_cast<uint32_t>(phase1_candidates.size());
    std::vector<SelectorCandidateMetrics> phase1_metrics =
        evaluate_phase1_candidates_progressive_k(phase1_candidates,
                                                options,
                                                context,
                                                coarse_eval,
                                                phase1_limits,
                                                &train_cache,
                                                &search_controller);
    stats.phase1_ok = count_ok_metrics(phase1_metrics);
    stats.budget_failures += count_budget_failures(phase1_metrics);
    stats.best_phase1_recall = best_recall_in_metrics(phase1_metrics);
    stats.best_phase1_quality = best_quality_in_metrics(phase1_metrics);
    if (!has_valid_metrics(phase1_metrics)) {
      stats.promotion_decision = "no_valid_phase1";
      all_phase1_metrics.insert(all_phase1_metrics.end(), phase1_metrics.begin(), phase1_metrics.end());
      finish_context_stats();
      if (search_controller.time_budget_hit.load(std::memory_order_relaxed)) {
        stopped_reason = "time_budget";
        break;
      }
      continue;
    }
    assign_objective_scores(&phase1_metrics, options.weights);
    all_phase1_metrics.insert(all_phase1_metrics.end(), phase1_metrics.begin(), phase1_metrics.end());

    std::vector<SelectorCandidateMetrics> phase1_ranked =
        top_ok_candidates(phase1_metrics, options.phase1_keep, std::nullopt, options.recall_target);
    if (phase1_ranked.empty()) {
      stats.promotion_decision = "no_phase1_promotion";
      finish_context_stats();
      continue;
    }
    double context_phase1_best_recall = phase1_ranked.front().recall_at_k_in_window;
    double context_phase1_best_quality = best_quality_in_metrics(phase1_ranked);
    if (recall_target_met_so_far &&
        best_quality_so_far >= 0.0 &&
        context_phase1_best_quality >= 0.0 &&
        (best_quality_so_far - context_phase1_best_quality) >
            std::max(0.08, options.min_quality_improvement * 8.0)) {
      contexts_skipped_low_potential += 1;
      stats.promotion_decision = "skipped_low_potential";
      finish_context_stats();
      continue;
    }

    uint32_t phase2_limit = adaptive_phase2_candidate_limit(options,
                                                            context.csr.node_ids.size(),
                                                            max_skeleton_count,
                                                            context_phase1_best_recall,
                                                            best_recall_so_far,
                                                            recall_target_met_so_far);
    stats.phase2_candidate_limit = phase2_limit;
    SelectorOptions phase2_search_options = tune_phase2_search_options(options, phase2_limit);
    if (phase2_search_options.optimize_for_recall) {
      phase2_search_options.max_phase2_candidates =
          std::max<uint32_t>(phase2_search_options.max_phase2_candidates,
                             phase2_search_options.phase1_keep * 3);
      phase2_search_options.beam_width =
          std::min<uint32_t>(phase2_search_options.max_phase2_candidates,
                             std::max<uint32_t>(phase2_search_options.beam_width,
                                                phase2_search_options.phase1_keep));
      phase2_search_options.beam_rounds =
          std::max<uint32_t>(phase2_search_options.beam_rounds, 3);
      phase2_search_options.hill_climb_steps =
          std::max<uint32_t>(phase2_search_options.hill_climb_steps, 3);
      phase2_search_options.beam_neighbor_limit =
          std::max<uint32_t>(phase2_search_options.beam_neighbor_limit, 128);
    }

    Phase2SearchResult phase2_search =
        search_phase2_candidates_hybrid(phase1_metrics,
                                        phase1_ranked,
                                        phase2_search_options,
                                        context,
                                        coarse_eval,
                                        phase1_limits,
                                        context.csr.node_ids.size(),
                                        &train_cache,
                                        &search_controller);
    stats.promotion_decision = "promoted_phase2";
    stats.phase2_refine_evaluated = phase2_search.refine_evaluated;
    stats.phase2_refine_rounds = phase2_search.refine_rounds;
    stats.phase2_frontier_candidates = phase2_search.frontier_candidates;
    stats.phase2_compacted_candidates = phase2_search.compacted_candidates;
    stats.phase2_plateau_rounds = phase2_search.plateau_rounds;
    stats.phase2_stopped_reason = phase2_search.stopped_reason;
    total_refine_evaluated += phase2_search.refine_evaluated;
    total_refine_rounds += phase2_search.refine_rounds;
    total_frontier_candidates += phase2_search.frontier_candidates;
    total_compacted_candidates += phase2_search.compacted_candidates;
    if (phase2_search.candidates.empty()) {
      stats.promotion_decision = "phase2_no_candidates";
      finish_context_stats();
      if (search_controller.time_budget_hit.load(std::memory_order_relaxed)) {
        stopped_reason = "time_budget";
        break;
      }
      continue;
    }
    phase2_search.candidates =
        fit_candidates_to_remaining_train_budget(phase2_search.candidates,
                                                 options,
                                                 context,
                                                 &train_cache);
    if (phase2_search.candidates.empty()) {
      stats.phase2_candidates = 0;
      stats.promotion_decision = "phase2_budget_exhausted";
      finish_context_stats();
      if (search_controller.time_budget_hit.load(std::memory_order_relaxed)) {
        stopped_reason = "time_budget";
        break;
      }
      continue;
    }

    std::vector<SelectorCandidateMetrics> phase2_metrics =
        evaluate_candidates(phase2_search.candidates,
                            phase2_search_options,
                            context,
                            full_eval,
                            phase2_limits,
                            &train_cache,
                            &search_controller);
    stats.phase2_candidates = static_cast<uint32_t>(phase2_metrics.size());
    if (phase2_search_options.optimize_for_recall &&
        has_valid_metrics(phase2_metrics) &&
        search_controller.can_launch_more_work()) {
      uint32_t recall_rounds_this_context = 0;
      std::vector<SelectorCandidateMetrics> recall_refined =
          refine_phase2_for_recall(phase2_metrics,
                                   phase2_search_options,
                                   context,
                                   full_eval,
                                   phase2_limits,
                                   context.csr.node_ids.size(),
                                   &train_cache,
                                   &search_controller,
                                   &recall_rounds_this_context);
      if (!recall_refined.empty()) {
        recall_refine_evaluated += static_cast<uint32_t>(recall_refined.size());
        phase2_metrics.insert(phase2_metrics.end(), recall_refined.begin(), recall_refined.end());
      }
      recall_refine_rounds += recall_rounds_this_context;
    }
    stats.phase2_candidates = static_cast<uint32_t>(phase2_metrics.size());
    stats.phase2_ok = count_ok_metrics(phase2_metrics);
    stats.budget_failures += count_budget_failures(phase2_metrics);
    stats.best_phase2_recall = best_recall_in_metrics(phase2_metrics);
    stats.best_phase2_quality = best_quality_in_metrics(phase2_metrics);
    if (stats.best_phase1_quality >= 0.0 && stats.best_phase2_quality >= 0.0) {
      stats.phase2_quality_gain = stats.best_phase2_quality - stats.best_phase1_quality;
    }
    all_phase2_metrics.insert(all_phase2_metrics.end(), phase2_metrics.begin(), phase2_metrics.end());

    double context_best_recall = std::max(context_phase1_best_recall, best_recall_in_metrics(phase2_metrics));
    if (context_best_recall >= 0.0 &&
        (best_recall_so_far < 0.0 ||
         context_best_recall > best_recall_so_far + options.min_recall_improvement)) {
      best_recall_so_far = context_best_recall;
    }
    double context_best_quality =
        std::max(context_phase1_best_quality, best_quality_in_metrics(phase2_metrics));
    if (context_best_quality >= 0.0 &&
        (best_quality_so_far < 0.0 ||
         context_best_quality > best_quality_so_far + options.min_quality_improvement)) {
      best_quality_so_far = context_best_quality;
      stagnant_contexts = 0;
    } else {
      stagnant_contexts += 1;
    }
    if (options.recall_target.has_value()) {
      double target = *options.recall_target;
      if (any_candidate_meets_recall_target(phase1_metrics, target) ||
          any_candidate_meets_recall_target(phase2_metrics, target)) {
        recall_target_met_so_far = true;
      }
    }
    if (effective_max_stagnant_contexts > 0 &&
        stagnant_contexts >= effective_max_stagnant_contexts &&
        (!options.recall_target.has_value() || recall_target_met_so_far)) {
      stopped_reason = options.recall_target.has_value()
          ? "recall_target_quality_plateau"
          : "quality_plateau";
      if (stats.promotion_decision == "promoted_phase2") {
        stats.promotion_decision = "promoted_phase2_then_plateau_stop";
      }
      finish_context_stats();
      break;
    }
    if (search_controller.time_budget_hit.load(std::memory_order_relaxed)) {
      stopped_reason = "time_budget";
      finish_context_stats();
      break;
    }
    finish_context_stats();
  }
  append_timing(&timings, "run_context_search", context_search_start);
  std::vector<SelectorCandidateMetrics> scored_phase1_metrics = all_phase1_metrics;
  if (has_valid_metrics(scored_phase1_metrics)) {
    assign_objective_scores(&scored_phase1_metrics, eval_options.weights);
  }
  std::vector<SelectorCandidateMetrics> scored_phase2_metrics = all_phase2_metrics;
  if (has_valid_metrics(scored_phase2_metrics)) {
    assign_objective_scores(&scored_phase2_metrics, eval_options.weights);
  }

  std::vector<SelectorCandidateMetrics> exploitation_pool = scored_phase2_metrics;
  if (exploitation_pool.empty() || !has_valid_metrics(exploitation_pool)) {
    exploitation_pool = scored_phase1_metrics;
  }
  std::vector<SelectorCandidateMetrics> post_exploration_metrics;
  if (post_exploration_candidate_count > 0) {
    auto exploitation_start = Clock::now();
    post_exploration_metrics = run_post_exploration(exploitation_pool,
                                                    contexts,
                                                    eval_options,
                                                    max_skeleton_count,
                                                    full_eval,
                                                    phase2_limits,
                                                    post_exploration_candidate_count,
                                                    &train_cache,
                                                    &search_controller);
    append_timing(&timings, "run_post_exploration", exploitation_start);
  }

  std::vector<SelectorCandidateMetrics> final_calibration_metrics;
  std::vector<SelectorCandidateMetrics> final_calibration_screen_metrics;
  std::optional<EvalDataset> final_calibration_screen_eval;
  std::vector<SelectorCalibrationProbe> final_calibration_progressive_probes;
  uint32_t final_calibration_seed_count = 0;
  uint32_t final_calibration_screen_promoted = 0;
  uint32_t final_calibration_role_promoted = 0;
  uint32_t final_calibration_recall_promoted = 0;
  uint32_t effective_final_calibration_screen_candidate_count = 0;
  uint32_t effective_final_calibration_screen_query_limit = 0;
  uint32_t final_calibration_reused = 0;
  std::string final_calibration_screen_status =
      eval_options.final_calibration_count > 0 ? "not_needed" : "not_requested";
  if (eval_options.final_calibration_count > 0) {
    std::vector<SelectorCandidateMetrics> calibration_pool = scored_phase2_metrics;
    if (calibration_pool.empty() || !has_valid_metrics(calibration_pool)) {
      calibration_pool = scored_phase1_metrics;
    }
    if (!post_exploration_metrics.empty()) {
      calibration_pool.insert(calibration_pool.end(),
                              post_exploration_metrics.begin(),
                              post_exploration_metrics.end());
    }
    uint32_t effective_full_calibration_count =
        resolve_final_calibration_full_count(eval_options, full_base.count);
    effective_final_calibration_screen_candidate_count =
        resolve_final_calibration_screen_candidate_count(eval_options, full_base.count);
    uint32_t initial_calibration_seed_count = eval_options.final_calibration_count;
    if (effective_full_calibration_count > 0 &&
        effective_final_calibration_screen_candidate_count > 0 &&
        effective_final_calibration_screen_candidate_count < initial_calibration_seed_count) {
      initial_calibration_seed_count = effective_final_calibration_screen_candidate_count;
    }

    std::vector<SelectorCandidateMetrics> calibration_seeds =
        choose_final_calibration_seeds(calibration_pool,
                                       initial_calibration_seed_count,
                                       eval_options.recall_target);
    final_calibration_seed_count = static_cast<uint32_t>(
        std::min<std::size_t>(calibration_seeds.size(),
                              std::numeric_limits<uint32_t>::max()));
    std::vector<SelectorCandidateMetrics> sampled_calibration_probe_metrics =
        calibration_seeds;
    if (effective_full_calibration_count > 0 &&
        calibration_seeds.size() <= effective_full_calibration_count) {
      final_calibration_screen_status = "not_needed_seed_count_le_full_limit";
    }
    if (effective_full_calibration_count > 0 &&
        calibration_seeds.size() > effective_full_calibration_count) {
      effective_final_calibration_screen_query_limit =
          resolve_final_calibration_screen_query_limit(
              eval_options,
              full_query.has_value() ? full_query->count : full_base.count);
      if (effective_final_calibration_screen_query_limit > 0 &&
          search_controller.can_launch_more_work()) {
        final_calibration_screen_status = "used";
        auto screen_eval_start = Clock::now();
        final_calibration_screen_eval =
            build_eval_dataset(full_base,
                               full_query,
                               eval_options.eval_k,
                               eval_options.eval_window,
                               eval_options.eval_window_values,
                               eval_options.eval_node_count_values,
                                               0,
                               effective_final_calibration_screen_query_limit,
                               eval_options.seed + 2601,
                               true);
        const EvalDataset& screen_eval = *final_calibration_screen_eval;
        append_timing(&timings, "build_final_calibration_screen_eval", screen_eval_start);

        PhaseLimits screen_limits;
        screen_limits.name = "final_calibration_screen";
        screen_limits.base_limit = 0;
        screen_limits.query_limit = effective_final_calibration_screen_query_limit;
        screen_limits.latency_iterations =
            std::min(eval_options.latency_iterations,
                     std::max<uint32_t>(1, eval_options.coarse_latency_iterations));
        screen_limits.latency_warmup =
            std::min(eval_options.latency_warmup,
                     std::max<uint32_t>(1, eval_options.coarse_latency_warmup));

        auto screen_start = Clock::now();
        final_calibration_screen_metrics =
            evaluate_calibration_seeds(calibration_seeds,
                                       contexts,
                                       eval_options,
                                       screen_eval,
                                       screen_limits,
                                       &train_cache,
                                       &search_controller);
        if (has_valid_metrics(final_calibration_screen_metrics)) {
          assign_objective_scores(&final_calibration_screen_metrics,
                                  eval_options.weights);
          if (has_valid_metrics(sampled_calibration_probe_metrics)) {
            final_calibration_progressive_probes.push_back(
                make_final_calibration_probe("sampled_to_final_calibration_screen",
                                             sampled_calibration_probe_metrics,
                                             full_eval,
                                             final_calibration_screen_metrics,
                                             screen_eval,
                                             eval_options));
          }
          std::vector<SelectorCandidateMetrics> promoted =
              choose_role_aware_final_calibration_seeds(
                  final_calibration_screen_metrics,
                  calibration_seeds,
                  effective_full_calibration_count,
                  eval_options.size_budget_bytes,
                  eval_options.recall_target,
                  &final_calibration_role_promoted,
                  &final_calibration_recall_promoted);
          calibration_seeds =
              fill_promoted_calibration_seeds(std::move(promoted),
                                              calibration_seeds,
                                              effective_full_calibration_count);
          final_calibration_screen_promoted =
              static_cast<uint32_t>(std::min<std::size_t>(
                  calibration_seeds.size(),
                  std::numeric_limits<uint32_t>::max()));
        } else {
          final_calibration_screen_status = "no_valid_screen_metrics";
          calibration_seeds =
              choose_final_calibration_seeds(calibration_seeds,
                                             effective_full_calibration_count,
                                             eval_options.recall_target);
        }
        append_timing(&timings, "run_final_calibration_screen", screen_start);
      } else {
        final_calibration_screen_status =
            effective_final_calibration_screen_query_limit == 0
                ? "disabled"
                : "time_budget_guard";
        calibration_seeds =
            choose_final_calibration_seeds(calibration_seeds,
                                           effective_full_calibration_count,
                                           eval_options.recall_target);
      }
    }
    if (!calibration_seeds.empty() && search_controller.can_launch_more_work()) {
      auto calibration_eval_start = Clock::now();
      uint32_t calibration_query_limit = eval_options.final_calibration_query_limit > 0
          ? eval_options.final_calibration_query_limit
          : eval_options.eval_query_limit;
      EvalDataset calibration_eval = build_eval_dataset(full_base,
                                                        full_query,
                                                        eval_options.eval_k,
                                                        eval_options.eval_window,
                                                        eval_options.eval_window_values,
                                                        eval_options.eval_node_count_values,
                                               0,
                                                        calibration_query_limit,
                                                        eval_options.seed + 3001,
                                                        true);
      append_timing(&timings, "build_final_calibration_eval", calibration_eval_start);

      PhaseLimits final_limits;
      final_limits.name = "final_calibration";
      final_limits.base_limit = 0;
      final_limits.query_limit = calibration_query_limit;
      final_limits.latency_iterations = eval_options.latency_iterations;
      final_limits.latency_warmup = eval_options.latency_warmup;

      auto calibration_start = Clock::now();
      final_calibration_metrics.reserve(calibration_seeds.size());
      if (eval_datasets_equivalent(full_eval, calibration_eval)) {
        for (const auto& seed : calibration_seeds) {
          if (!seed.ok) {
            continue;
          }
          final_calibration_metrics.push_back(
              make_reused_final_calibration_metric(seed));
        }
        final_calibration_reused =
            static_cast<uint32_t>(std::min<std::size_t>(
                final_calibration_metrics.size(),
                std::numeric_limits<uint32_t>::max()));
      }
      if (final_calibration_reused == 0) {
        final_calibration_metrics =
            evaluate_calibration_seeds(calibration_seeds,
                                       contexts,
                                       eval_options,
                                       calibration_eval,
                                       final_limits,
                                       &train_cache,
                                       &search_controller);
      }
      if (has_valid_metrics(final_calibration_metrics)) {
        assign_objective_scores(&final_calibration_metrics, eval_options.weights);
        if (has_valid_metrics(final_calibration_screen_metrics) &&
            final_calibration_screen_eval.has_value()) {
          std::vector<SelectorCandidateMetrics> promoted_screen_metrics =
              metrics_for_seed_identities(final_calibration_screen_metrics,
                                          calibration_seeds);
          if (has_valid_metrics(promoted_screen_metrics)) {
            final_calibration_progressive_probes.push_back(
                make_final_calibration_probe("screen_to_final_calibration",
                                             promoted_screen_metrics,
                                             *final_calibration_screen_eval,
                                             final_calibration_metrics,
                                             calibration_eval,
                                             eval_options));
          }
        }
        if (has_valid_metrics(sampled_calibration_probe_metrics)) {
          final_calibration_progressive_probes.push_back(
              make_final_calibration_probe("sampled_to_final_calibration",
                                           sampled_calibration_probe_metrics,
                                           full_eval,
                                           final_calibration_metrics,
                                           calibration_eval,
                                           eval_options));
        }
      }
      append_timing(&timings, "run_final_calibration", calibration_start);
    }
  }

  std::vector<SelectorCandidateMetrics> ranking_metrics = final_calibration_metrics;
  if (ranking_metrics.empty() || !has_valid_metrics(ranking_metrics)) {
    ranking_metrics = scored_phase2_metrics;
    ranking_metrics.insert(ranking_metrics.end(),
                           post_exploration_metrics.begin(),
                           post_exploration_metrics.end());
  }
  if (ranking_metrics.empty() || !has_valid_metrics(ranking_metrics)) {
    ranking_metrics = scored_phase1_metrics;
  }
  if (ranking_metrics.empty() || !has_valid_metrics(ranking_metrics)) {
    throw std::runtime_error("Selector failed: no valid candidates");
  }

  std::vector<SelectorCandidateMetrics> ranked_for_margin =
      top_ok_candidates(ranking_metrics,
                        ranking_metrics.size(),
                        std::nullopt,
                        eval_options.recall_target);

  SelectorResult result;
  result.candidates = scored_phase1_metrics;
  result.candidates.insert(result.candidates.end(),
                           scored_phase2_metrics.begin(),
                           scored_phase2_metrics.end());
  result.candidates.insert(result.candidates.end(),
                           post_exploration_metrics.begin(),
                           post_exploration_metrics.end());
  result.candidates.insert(result.candidates.end(),
                           final_calibration_screen_metrics.begin(),
                           final_calibration_screen_metrics.end());
  result.candidates.insert(result.candidates.end(),
                           final_calibration_metrics.begin(),
                           final_calibration_metrics.end());
  for (const auto& item : result.candidates) {
    if (item.ok || item.error.empty()) {
      continue;
    }
    result.candidate_error_count += 1;
    if (item.error_stage == "launch_guard") {
      result.candidate_error_launch_guard_count += 1;
    } else if (item.error_stage == "training_base") {
      result.candidate_error_training_base_count += 1;
    } else if (item.error_stage == "centroid_order") {
      result.candidate_error_centroid_order_count += 1;
    } else if (item.error_stage == "cdf_fit") {
      result.candidate_error_cdf_fit_count += 1;
    } else if (item.error_stage == "evaluation") {
      result.candidate_error_eval_count += 1;
    } else {
      result.candidate_error_unknown_count += 1;
    }
  }
  result.pareto_front = pareto_front(ranking_metrics);
  result.recommended =
      top_ok_candidates(ranking_metrics,
                        eval_options.recommend_count,
                        eval_options.size_budget_bytes,
                        eval_options.recall_target);
  if (result.recommended.empty()) {
    result.recommended =
        top_ok_candidates(ranking_metrics,
                          eval_options.recommend_count,
                          std::nullopt,
                          eval_options.recall_target);
  }
  if (!result.recommended.empty()) {
    result.best = result.recommended.front();
  }
  SelectorRecommendationRoles roles =
      choose_recommendation_roles(ranking_metrics,
                                  eval_options.size_budget_bytes,
                                  eval_options.recall_target);
  result.recommended_peak_recall = std::move(roles.peak_recall);
  result.recommended_knee = std::move(roles.knee);
  result.recommended_fast = std::move(roles.fast);
  result.recommended_small = std::move(roles.small);
  if (eval_options.recall_target.has_value()) {
    double recall_target = *eval_options.recall_target;
    for (const auto& item : result.recommended) {
      if (meets_recall_target(item, recall_target)) {
        result.recall_target_met_recommended += 1;
      }
    }
    if (result.best.has_value()) {
      result.recall_target_met = meets_recall_target(*result.best, recall_target);
    }
  }
  result.effective_weights = eval_options.weights;
  result.dataset_count = static_cast<uint64_t>(full_base.count);
  result.effective_eval_windows = ranking_eval_windows(ranking_metrics, full_eval.windows);
  result.effective_eval_node_counts = full_eval.node_counts;
  result.auto_dataset_weighting_applied = auto_weighting_applied;
  result.effective_selector_profile = effective_profile;
  result.target_skeleton_contexts_total = static_cast<uint32_t>(contexts.size());
  result.target_skeleton_contexts_requested = requested_contexts;
  result.target_skeleton_contexts_unique = static_cast<uint32_t>(contexts.size());
  result.target_skeleton_contexts_deduplicated =
      requested_contexts > result.target_skeleton_contexts_unique
          ? requested_contexts - result.target_skeleton_contexts_unique
          : 0;
  result.target_skeleton_contexts_selected =
      static_cast<uint32_t>(context_plan.indices.size());
  result.target_skeleton_contexts_evaluated = evaluated_contexts;
  result.contexts_skipped_low_potential = contexts_skipped_low_potential;
  result.max_model_trains = train_cache.max_model_trains;
  {
    std::lock_guard<std::mutex> lock(train_cache.mutex);
    result.model_trains_started = train_cache.model_trains_started;
    result.train_cache_hits = train_cache.cache_hits;
    result.train_cache_misses = train_cache.cache_misses;
    result.train_cache_wait_ms = train_cache.cache_wait_ms;
    result.training_lane_wait_ms = train_cache.training_lane_wait_ms;
    result.training_base_cache_hits = train_cache.training_base_cache_hits;
    result.training_base_cache_misses = train_cache.training_base_cache_misses;
    result.training_order_cache_hits = train_cache.training_order_cache_hits;
    result.training_order_cache_misses = train_cache.training_order_cache_misses;
    result.training_cdf_cache_hits = train_cache.training_cdf_cache_hits;
    result.training_cdf_cache_misses = train_cache.training_cdf_cache_misses;
    result.training_cdf_cache_failures = train_cache.training_cdf_cache_failures;
    result.training_base_cache_wait_ms = train_cache.training_base_cache_wait_ms;
    result.training_order_cache_wait_ms = train_cache.training_order_cache_wait_ms;
    result.training_cdf_cache_wait_ms = train_cache.training_cdf_cache_wait_ms;
    result.training_base_cache_build_ms = train_cache.training_base_cache_build_ms;
    result.training_order_cache_build_ms = train_cache.training_order_cache_build_ms;
    result.training_cdf_cache_build_ms = train_cache.training_cdf_cache_build_ms;
    result.training_base_cache_memory_bytes =
        train_cache.training_base_cache_memory_bytes;
    result.training_order_cache_memory_bytes =
        train_cache.training_order_cache_memory_bytes;
    result.training_cdf_cache_memory_bytes =
        train_cache.training_cdf_cache_memory_bytes;
    result.train_time_ms_sum = train_cache.train_time_ms_sum;
    result.train_gather_skeleton_ms_sum = train_cache.train_gather_skeleton_ms_sum;
    result.train_cluster_assign_ms_sum = train_cache.train_cluster_assign_ms_sum;
    result.train_assign_centroids_ms_sum = train_cache.train_assign_centroids_ms_sum;
    result.train_centroid_order_ms_sum = train_cache.train_centroid_order_ms_sum;
    result.train_range_alloc_ms_sum = train_cache.train_range_alloc_ms_sum;
    result.train_cdf_fit_ms_sum = train_cache.train_cdf_fit_ms_sum;
    result.train_model_assembly_ms_sum = train_cache.train_model_assembly_ms_sum;
    result.scheduler_ready_cached_candidates =
        train_cache.scheduler_ready_cached_candidates;
    result.scheduler_in_flight_candidates =
        train_cache.scheduler_in_flight_candidates;
    result.scheduler_missing_candidates =
        train_cache.scheduler_missing_candidates;
    result.scheduler_single_admission_batches =
        train_cache.scheduler_single_admission_batches;
    result.scheduler_reuse_prioritized_candidates =
        train_cache.scheduler_reuse_prioritized_candidates;
    result.scheduler_fresh_base_candidates =
        train_cache.scheduler_fresh_base_candidates;
    result.eval_nearest_cache_hits = train_cache.eval_nearest_cache_hits;
    result.eval_nearest_cache_misses = train_cache.eval_nearest_cache_misses;
    result.eval_nearest_cache_inflight_bypasses =
        train_cache.eval_nearest_cache_inflight_bypasses;
    result.eval_nearest_cache_prewarm_requests =
        train_cache.eval_nearest_cache_prewarm_requests;
    result.eval_nearest_cache_prewarm_ready_models =
        train_cache.eval_nearest_cache_prewarm_ready_models;
    result.eval_nearest_cache_prewarm_skipped =
        train_cache.eval_nearest_cache_prewarm_skipped;
    result.eval_nearest_cache_prewarm_failures =
        train_cache.eval_nearest_cache_prewarm_failures;
    result.eval_nearest_cache_prewarm_threads =
        train_cache.eval_nearest_cache_prewarm_threads;
    result.eval_nearest_cache_prewarm_ms =
        train_cache.eval_nearest_cache_prewarm_ms;
    result.eval_nearest_cache_wait_ms = train_cache.eval_nearest_cache_wait_ms;
    result.eval_nearest_cache_sampled_wait_ms =
        train_cache.eval_nearest_cache_sampled_wait_ms;
    result.eval_nearest_cache_calibration_wait_ms =
        train_cache.eval_nearest_cache_calibration_wait_ms;
    result.eval_nearest_cache_build_ms = train_cache.eval_nearest_cache_build_ms;
    result.eval_nearest_cache_sampled_build_ms =
        train_cache.eval_nearest_cache_sampled_build_ms;
    result.eval_nearest_cache_calibration_build_ms =
        train_cache.eval_nearest_cache_calibration_build_ms;
    result.eval_nearest_cache_distance_terms =
        train_cache.eval_nearest_cache_distance_terms;
    result.eval_nearest_cache_skipped_distance_terms =
        train_cache.eval_nearest_cache_skipped_distance_terms;
    result.eval_nearest_cache_early_abandon_builds =
        train_cache.eval_nearest_cache_early_abandon_builds;
    result.eval_nearest_cache_entries = train_cache.eval_nearest_cache_entries;
    result.eval_nearest_cache_memory_bytes =
        train_cache.eval_nearest_cache_memory_bytes;
    result.eval_nearest_cache_sampled_budget_bytes =
        train_cache.eval_nearest_cache_sampled_budget_bytes;
    result.eval_nearest_cache_sampled_memory_bytes =
        train_cache.eval_nearest_cache_sampled_memory_bytes;
    result.eval_nearest_cache_calibration_memory_bytes =
        train_cache.eval_nearest_cache_calibration_memory_bytes;
    result.eval_nearest_cache_evictions = train_cache.eval_nearest_cache_evictions;
    result.eval_nearest_cache_evicted_bytes =
        train_cache.eval_nearest_cache_evicted_bytes;
    result.eval_base_rank_cache_hits = train_cache.eval_base_rank_cache_hits;
    result.eval_base_rank_cache_misses = train_cache.eval_base_rank_cache_misses;
    result.eval_base_rank_cache_wait_ms = train_cache.eval_base_rank_cache_wait_ms;
    result.eval_base_rank_cache_build_ms = train_cache.eval_base_rank_cache_build_ms;
    result.eval_base_rank_cache_nearest_ms =
        train_cache.eval_base_rank_cache_nearest_ms;
    result.eval_base_rank_cache_hash_ms =
        train_cache.eval_base_rank_cache_hash_ms;
    result.eval_base_rank_cache_sort_ms =
        train_cache.eval_base_rank_cache_sort_ms;
    result.eval_base_rank_cache_rank_index_ms =
        train_cache.eval_base_rank_cache_rank_index_ms;
    result.eval_base_rank_cache_overhead_ms =
        train_cache.eval_base_rank_cache_overhead_ms;
    result.eval_base_rank_cache_entries = train_cache.eval_base_rank_cache_entries;
    result.eval_base_rank_cache_memory_bytes =
        train_cache.eval_base_rank_cache_memory_bytes;
    result.eval_base_rank_cache_sampled_budget_bytes =
        train_cache.eval_base_rank_cache_sampled_budget_bytes;
    result.eval_base_rank_cache_sampled_memory_bytes =
        train_cache.eval_base_rank_cache_sampled_memory_bytes;
    result.eval_base_rank_cache_calibration_memory_bytes =
        train_cache.eval_base_rank_cache_calibration_memory_bytes;
    result.eval_base_rank_cache_evictions = train_cache.eval_base_rank_cache_evictions;
    result.eval_base_rank_cache_evicted_bytes =
        train_cache.eval_base_rank_cache_evicted_bytes;
    result.model_train_budget_hit =
        train_cache.model_train_budget_hit ||
        (train_cache.max_model_trains > 0 &&
         train_cache.model_trains_started >= train_cache.max_model_trains);
  }
  result.eval_base_rank_cache_attribution =
      snapshot_eval_base_rank_cache_attribution(&train_cache);
  annotate_base_rank_attribution_roles(&result);
  result.effective_max_search_seconds = effective_max_search_seconds;
  result.elapsed_search_seconds = search_controller.elapsed_seconds();
  result.search_time_budget_hit =
      search_controller.time_budget_hit.load(std::memory_order_relaxed);
  result.remaining_time_guard_hit =
      search_controller.remaining_time_guard_hit.load(std::memory_order_relaxed);
  result.final_calibration_reserve_seconds =
      search_controller.final_calibration_reserve_seconds;
  result.final_calibration_reserve_hit =
      search_controller.final_calibration_reserve_hit.load(std::memory_order_relaxed);
  if (result.search_time_budget_hit) {
    stopped_reason = "time_budget";
  } else if (result.final_calibration_reserve_hit) {
    stopped_reason = "final_calibration_reserve";
  } else if (result.remaining_time_guard_hit) {
    stopped_reason = "remaining_time_guard";
  } else if (result.model_train_budget_hit &&
             evaluated_contexts < static_cast<uint32_t>(context_plan.indices.size())) {
    stopped_reason = "model_train_budget";
  }
  result.stopped_reason = stopped_reason;
  result.plateau_rounds = stagnant_contexts;
  result.best_quality = best_quality_in_metrics(ranking_metrics);
  result.best_margin = quality_margin_in_ranked(ranked_for_margin);
  result.strategy_refine_evaluated = total_refine_evaluated;
  result.strategy_refine_rounds = total_refine_rounds;
  result.strategy_frontier_candidates = total_frontier_candidates;
  result.strategy_compacted_candidates = total_compacted_candidates;
  result.recall_refine_evaluated = recall_refine_evaluated;
  result.recall_refine_rounds = recall_refine_rounds;
  result.post_exploration_evaluated =
      static_cast<uint32_t>(post_exploration_metrics.size());
  result.post_exploration_rounds = post_exploration_metrics.empty() ? 0 : 1;
  result.final_calibration_evaluated =
      static_cast<uint32_t>(final_calibration_metrics.size());
  result.final_calibration_seed_count = final_calibration_seed_count;
  result.final_calibration_screen_evaluated =
      static_cast<uint32_t>(final_calibration_screen_metrics.size());
  result.final_calibration_screen_promoted = final_calibration_screen_promoted;
  result.final_calibration_role_promoted = final_calibration_role_promoted;
  result.final_calibration_recall_promoted = final_calibration_recall_promoted;
  result.final_calibration_screen_status = std::move(final_calibration_screen_status);
  result.final_calibration_reused = final_calibration_reused;
  result.final_calibration_progressive_probes =
      std::move(final_calibration_progressive_probes);
  result.effective_final_calibration_full_count =
      resolve_final_calibration_full_count(eval_options, full_base.count);
  result.effective_final_calibration_screen_candidate_count =
      effective_final_calibration_screen_candidate_count;
  result.effective_final_calibration_screen_query_limit =
      effective_final_calibration_screen_query_limit;
  result.effective_selector_memory_budget_bytes = train_cache.memory_budget_bytes;
  result.selector_peak_parallelism = train_cache.peak_parallelism;
  result.selector_peak_threads_per_candidate = train_cache.peak_threads_per_candidate;
  result.selector_memory_budget_limited = train_cache.memory_budget_limited;
  append_timing(&timings, "total_select", total_start);
  result.timings = std::move(timings);
  result.context_stats = std::move(context_stats);
  return result;
}


} // namespace vortex
