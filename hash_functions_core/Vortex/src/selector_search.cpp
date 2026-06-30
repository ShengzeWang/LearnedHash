#include "selector_internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vortex::selector_internal {

uint32_t weak_high_k_family_limit(const SelectorOptions& options) {
  uint32_t half_phase1_keep = std::max<uint32_t>(2, options.phase1_keep / 2);
  return std::max<uint32_t>(2, std::min<uint32_t>(options.beam_width, half_phase1_keep));
}

ContextSelectionPlan select_context_indices_for_full_search(
    const std::vector<NswSelectorContext>& contexts,
    const SelectorOptions& effective_raw,
    const EvalDataset& coarse_eval,
    const PhaseLimits& phase1_limits,
    std::size_t dataset_count,
    std::size_t max_skeleton_count,
    const std::string& effective_profile,
    TrainCache* train_cache,
    SearchController* search) {
  ContextSelectionPlan plan;
  std::vector<std::size_t> all = all_indices(contexts.size());
  uint32_t keep_count_u32 = resolve_context_limit(
      effective_raw, effective_profile, dataset_count, contexts.size());
  std::size_t keep_count = static_cast<std::size_t>(keep_count_u32);
  if (keep_count == 0 || keep_count >= contexts.size()) {
    plan.indices = std::move(all);
    return plan;
  }

  struct ProbeRow {
    std::size_t index = 0;
    bool ok = false;
    double score = -std::numeric_limits<double>::infinity();
    uint64_t target_skeleton = 0;
    std::size_t skeleton_count = 0;
  };
  std::size_t probe_count =
      std::min<std::size_t>(contexts.size(), std::max<std::size_t>(keep_count * 2, keep_count + 2));
  std::vector<std::size_t> probe_indices = evenly_spaced_indices(contexts.size(), probe_count);
  if (!probe_indices.empty()) {
    probe_indices.front() = 0;
    probe_indices.back() = contexts.size() - 1;
  }
  std::vector<ProbeRow> rows;
  rows.reserve(probe_indices.size());
  for (std::size_t idx : probe_indices) {
    if (search != nullptr && !search->can_launch_more_work()) {
      break;
    }
    const auto& context = contexts[idx];
    SelectorOptions probe_options =
        normalize_selector_options(effective_raw, context.csr.node_ids.size());
    probe_options.nsw_path = context.nsw_path;
    probe_options = apply_context_capacity_scaling(
        probe_options, context.csr.node_ids.size(), max_skeleton_count);
    probe_options = normalize_selector_options(probe_options, context.csr.node_ids.size());
    probe_options.nsw_path = context.nsw_path;
    SelectorCandidateConfig probe_config = build_context_probe_candidate(
        probe_options, context.csr.node_ids.size(), context.target_skeleton);
    std::vector<SelectorCandidateMetrics> probe_metrics =
        evaluate_candidates({probe_config},
                            probe_options,
                            context,
                            coarse_eval,
                            phase1_limits,
                            train_cache,
                            search);
    assign_objective_scores(&probe_metrics, probe_options.weights);
    ProbeRow row;
    row.index = idx;
    row.target_skeleton = context.target_skeleton;
    row.skeleton_count = context.csr.node_ids.size();
    if (!probe_metrics.empty() && probe_metrics.front().ok) {
      row.ok = true;
      row.score = probe_metrics.front().objective_score;
      plan.probe_contexts_evaluated += 1;
    }
    rows.push_back(row);
  }

  std::stable_sort(rows.begin(),
                   rows.end(),
                   [](const ProbeRow& lhs, const ProbeRow& rhs) {
                     if (lhs.ok != rhs.ok) {
                       return lhs.ok;
                     }
                     if (lhs.ok && rhs.ok &&
                         std::abs(lhs.score - rhs.score) > kNormEpsilon) {
                       return lhs.score > rhs.score;
                     }
                     if (lhs.skeleton_count != rhs.skeleton_count) {
                       return lhs.skeleton_count > rhs.skeleton_count;
                     }
                     return lhs.index < rhs.index;
                   });

  std::set<std::size_t> selected;
  if (keep_count == 1) {
    if (!rows.empty()) {
      selected.insert(rows.front().index);
    } else {
      selected.insert(0);
    }
  } else {
    // Keep boundary contexts so users still see low/high skeleton behavior.
    selected.insert(0);
    selected.insert(contexts.size() - 1);
  }
  for (const auto& row : rows) {
    if (selected.size() >= keep_count) {
      break;
    }
    selected.insert(row.index);
  }

  if (selected.size() < keep_count) {
    std::vector<std::size_t> spaced = evenly_spaced_indices(contexts.size(), keep_count);
    for (std::size_t idx : spaced) {
      if (selected.size() >= keep_count) {
        break;
      }
      selected.insert(idx);
    }
  }

  std::unordered_map<std::size_t, double> score_by_index;
  score_by_index.reserve(rows.size());
  for (const auto& row : rows) {
    score_by_index[row.index] =
        row.ok ? row.score : -std::numeric_limits<double>::infinity();
  }
  std::vector<std::size_t> out(selected.begin(), selected.end());
  std::stable_sort(
      out.begin(),
      out.end(),
      [&](std::size_t lhs_idx, std::size_t rhs_idx) {
        double lhs_score = -std::numeric_limits<double>::infinity();
        double rhs_score = -std::numeric_limits<double>::infinity();
        auto lhs_it = score_by_index.find(lhs_idx);
        auto rhs_it = score_by_index.find(rhs_idx);
        if (lhs_it != score_by_index.end()) {
          lhs_score = lhs_it->second;
        }
        if (rhs_it != score_by_index.end()) {
          rhs_score = rhs_it->second;
        }
        if (std::abs(lhs_score - rhs_score) > kNormEpsilon) {
          return lhs_score > rhs_score;
        }
        const auto& lhs = contexts[lhs_idx];
        const auto& rhs = contexts[rhs_idx];
        if (lhs.csr.node_ids.size() != rhs.csr.node_ids.size()) {
          return lhs.csr.node_ids.size() > rhs.csr.node_ids.size();
        }
        return lhs_idx < rhs_idx;
      });
  plan.indices = std::move(out);
  return plan;
}

Phase2SearchResult search_phase2_candidates_hybrid(
    const std::vector<SelectorCandidateMetrics>& phase1_metrics,
    const std::vector<SelectorCandidateMetrics>& phase1_ranked,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    const EvalDataset& coarse_eval,
    const PhaseLimits& coarse_limits,
    std::size_t skeleton_count,
    TrainCache* cache,
    SearchController* search) {
  Phase2SearchResult result;
  std::vector<SelectorCandidateConfig> grid_candidates =
      expand_phase2_candidates_grid(phase1_ranked, options, skeleton_count);
  if (!options.advanced_search) {
    result.candidates = std::move(grid_candidates);
    result.stopped_reason = "advanced_search_disabled";
    return result;
  }

  std::vector<SelectorCandidateMetrics> pool = phase1_metrics;
  assign_objective_scores(&pool, options.weights);
  std::vector<SelectorCandidateMetrics> beam =
      top_ok_candidates(pool, options.beam_width, std::nullopt, options.recall_target);
  if (beam.empty()) {
    result.candidates = std::move(grid_candidates);
    result.stopped_reason = "no_valid_beam";
    return result;
  }

  std::unordered_set<std::string> seen_keys;
  seen_keys.reserve(pool.size() * 2 + 1);
  for (const auto& item : pool) {
    seen_keys.insert(selector_config_key(item.config));
  }

  PhaseLimits beam_limits = coarse_limits;
  beam_limits.name = "phase1_beam";
  double best_beam_quality = best_quality_in_metrics(beam);
  uint32_t plateau_rounds = 0;
  for (uint32_t round = 0; round < options.beam_rounds; ++round) {
    if (search != nullptr && !search->can_launch_more_work()) {
      result.stopped_reason = "time_budget";
      break;
    }
    std::vector<SelectorCandidateConfig> frontier;
    for (const auto& seed : beam) {
      std::vector<SelectorCandidateConfig> neighbors =
          local_neighbors(seed.config,
                          options,
                          skeleton_count,
                          options.hill_climb_steps,
                          options.beam_neighbor_limit);
      for (const auto& candidate : neighbors) {
        std::string key = selector_config_key(candidate);
        if (!seen_keys.insert(key).second) {
          continue;
        }
        frontier.push_back(candidate);
      }
    }
    if (frontier.empty()) {
      result.stopped_reason = "frontier_exhausted";
      break;
    }
    result.frontier_candidates += static_cast<uint32_t>(
        std::min<std::size_t>(frontier.size(), std::numeric_limits<uint32_t>::max()));
    uint32_t compacted = 0;
    frontier = compact_attribution_guided_candidates(frontier,
                                                     phase1_metrics,
                                                     options,
                                                     weak_high_k_family_limit(options),
                                                     &compacted);
    result.compacted_candidates += compacted;
    if (frontier.empty()) {
      result.stopped_reason = "frontier_compacted";
      break;
    }
    frontier = fit_candidates_to_remaining_train_budget(frontier, options, context, cache);
    if (frontier.empty()) {
      result.stopped_reason = "model_train_budget";
      break;
    }

    std::size_t round_limit =
        std::max<std::size_t>(static_cast<std::size_t>(options.max_phase2_candidates) * 4,
                              options.beam_width);
    if (frontier.size() > round_limit) {
      frontier.resize(round_limit);
    }
    std::vector<SelectorCandidateMetrics> frontier_metrics =
        evaluate_candidates(frontier, options, context, coarse_eval, beam_limits, cache, search);
    if (frontier_metrics.empty()) {
      result.stopped_reason = "empty_frontier_eval";
      break;
    }
    result.refine_rounds += 1;
    result.refine_evaluated += static_cast<uint32_t>(frontier_metrics.size());

    pool.insert(pool.end(), frontier_metrics.begin(), frontier_metrics.end());
    assign_objective_scores(&pool, options.weights);
    beam = top_ok_candidates(pool, options.beam_width, std::nullopt, options.recall_target);
    if (beam.empty()) {
      result.stopped_reason = "beam_exhausted";
      break;
    }
    double round_best_quality = best_quality_in_metrics(beam);
    if (round_best_quality > best_beam_quality + options.min_quality_improvement) {
      best_beam_quality = round_best_quality;
      plateau_rounds = 0;
    } else {
      plateau_rounds += 1;
      if (plateau_rounds >= 2) {
        result.stopped_reason = "beam_quality_plateau";
        break;
      }
    }
  }
  result.plateau_rounds = plateau_rounds;
  if (result.stopped_reason.empty()) {
    result.stopped_reason = "beam_round_budget";
  }
  result.best_margin = quality_margin_in_ranked(beam);

  std::vector<SelectorCandidateConfig> final_configs;
  final_configs.reserve(options.max_phase2_candidates * 2);
  std::vector<SelectorCandidateConfig> beam_configs =
      collect_seed_configs(beam, options.beam_width);
  final_configs.insert(final_configs.end(), beam_configs.begin(), beam_configs.end());
  std::vector<SelectorCandidateConfig> phase1_seeds =
      collect_seed_configs(phase1_ranked, options.phase1_keep);
  final_configs.insert(final_configs.end(), phase1_seeds.begin(), phase1_seeds.end());
  for (const auto& seed : beam_configs) {
    std::vector<SelectorCandidateConfig> neighbors =
        local_neighbors(seed, options, skeleton_count, 1, options.beam_neighbor_limit / 2 + 1);
    final_configs.insert(final_configs.end(), neighbors.begin(), neighbors.end());
  }
  final_configs.insert(final_configs.end(), grid_candidates.begin(), grid_candidates.end());
  final_configs = dedup_candidates(final_configs, skeleton_count);

  std::unordered_map<std::string, SelectorCandidateMetrics> metric_by_key;
  metric_by_key.reserve(pool.size());
  for (const auto& item : pool) {
    if (!item.ok) {
      continue;
    }
    std::string key = selector_config_key(item.config);
    auto it = metric_by_key.find(key);
    if (it == metric_by_key.end() || item.objective_score > it->second.objective_score) {
      metric_by_key[key] = item;
    }
  }

  std::stable_sort(final_configs.begin(), final_configs.end(),
                   [&](const SelectorCandidateConfig& lhs, const SelectorCandidateConfig& rhs) {
                     auto lhs_it = metric_by_key.find(selector_config_key(lhs));
                     auto rhs_it = metric_by_key.find(selector_config_key(rhs));
                     bool lhs_has = lhs_it != metric_by_key.end();
                     bool rhs_has = rhs_it != metric_by_key.end();
                     if (lhs_has != rhs_has) {
                       return lhs_has;
                     }
                     if (lhs_has && rhs_has) {
                       const auto& a = lhs_it->second;
                       const auto& b = rhs_it->second;
                       if (options.recall_target.has_value()) {
                         bool a_hit = meets_recall_target(a, *options.recall_target);
                         bool b_hit = meets_recall_target(b, *options.recall_target);
                         if (a_hit != b_hit) {
                           return a_hit;
                         }
                       }
                       if (std::abs(a.objective_score - b.objective_score) > kNormEpsilon) {
                         return a.objective_score > b.objective_score;
                       }
                       if (std::abs(selector_quality_score(a) -
                                    selector_quality_score(b)) > kNormEpsilon) {
                         return selector_quality_score(a) > selector_quality_score(b);
                       }
                       if (std::abs(a.latency_avg_ms - b.latency_avg_ms) > kNormEpsilon) {
                         return a.latency_avg_ms < b.latency_avg_ms;
                       }
                     }
                     return selector_config_key(lhs) < selector_config_key(rhs);
                   });

  uint32_t final_compacted = 0;
  final_configs = compact_attribution_guided_candidates(
      final_configs,
      phase1_metrics,
      options,
      std::max<uint32_t>(weak_high_k_family_limit(options), options.phase1_keep),
      &final_compacted);
  result.compacted_candidates += final_compacted;
  if (final_configs.size() > options.max_phase2_candidates) {
    final_configs.resize(options.max_phase2_candidates);
  }
  if (final_configs.empty()) {
    final_configs = std::move(grid_candidates);
  }
  result.candidates = std::move(final_configs);
  return result;
}

bool has_valid_metrics(const std::vector<SelectorCandidateMetrics>& metrics) {
  for (const auto& item : metrics) {
    if (item.ok) {
      return true;
    }
  }
  return false;
}

double best_recall_in_metrics(const std::vector<SelectorCandidateMetrics>& metrics) {
  double best = -1.0;
  for (const auto& item : metrics) {
    if (!item.ok) {
      continue;
    }
    if (item.recall_at_k_in_window > best) {
      best = item.recall_at_k_in_window;
    }
  }
  return best;
}

double best_quality_in_metrics(const std::vector<SelectorCandidateMetrics>& metrics) {
  double best = -1.0;
  for (const auto& item : metrics) {
    if (!item.ok) {
      continue;
    }
    best = std::max(best, selector_quality_score(item));
  }
  return best;
}

double quality_margin_in_ranked(const std::vector<SelectorCandidateMetrics>& ranked) {
  std::vector<double> qualities;
  qualities.reserve(ranked.size());
  for (const auto& item : ranked) {
    if (item.ok) {
      qualities.push_back(selector_quality_score(item));
    }
  }
  if (qualities.size() < 2) {
    return 0.0;
  }
  std::sort(qualities.begin(), qualities.end(), std::greater<double>());
  return std::max(0.0, qualities[0] - qualities[1]);
}

uint32_t adaptive_phase2_candidate_limit(const SelectorOptions& options,
                                         std::size_t skeleton_count,
                                         std::size_t max_skeleton_count,
                                         double context_phase1_best_recall,
                                         double global_best_recall,
                                         bool recall_target_met_so_far) {
  uint32_t limit = options.max_phase2_candidates;
  double scale = skeleton_capacity_scale(skeleton_count, max_skeleton_count);
  if (scale < 0.50) {
    limit = std::max<uint32_t>(options.phase1_keep, (limit * 2) / 3);
  }
  if (global_best_recall >= 0.0 && context_phase1_best_recall >= 0.0) {
    double gap = global_best_recall - context_phase1_best_recall;
    if (gap > 0.10) {
      limit = std::max<uint32_t>(options.phase1_keep, limit / 2);
    } else if (gap > 0.05) {
      limit = std::max<uint32_t>(options.phase1_keep, (limit * 3) / 4);
    }
  }
  if (options.recall_target.has_value() && !recall_target_met_so_far && scale >= 0.80) {
    limit = options.max_phase2_candidates;
  }
  return std::max<uint32_t>(limit, options.phase1_keep);
}

SelectorOptions tune_phase2_search_options(const SelectorOptions& options,
                                           uint32_t phase2_limit) {
  SelectorOptions tuned = options;
  tuned.max_phase2_candidates = std::max<uint32_t>(phase2_limit, tuned.phase1_keep);
  tuned.beam_width = std::min<uint32_t>(tuned.beam_width, tuned.max_phase2_candidates);
  if (tuned.beam_width == 0) {
    tuned.beam_width = 1;
  }
  if (options.max_phase2_candidates > tuned.max_phase2_candidates) {
    uint32_t scaled_neighbor_limit =
        static_cast<uint32_t>(std::max<uint64_t>(
            8,
            static_cast<uint64_t>(options.beam_neighbor_limit) *
                static_cast<uint64_t>(tuned.max_phase2_candidates) /
                static_cast<uint64_t>(options.max_phase2_candidates)));
    tuned.beam_neighbor_limit = std::min<uint32_t>(tuned.beam_neighbor_limit, scaled_neighbor_limit);
    tuned.beam_rounds = std::min<uint32_t>(
        tuned.beam_rounds,
        tuned.max_phase2_candidates <= tuned.phase1_keep + 1 ? 1 : tuned.beam_rounds);
  }
  return tuned;
}

SelectorCandidateConfig choose_recall_anchor(
    const std::vector<SelectorCandidateMetrics>& metrics,
    const SelectorCandidateConfig& fallback) {
  SelectorCandidateConfig anchor = fallback;
  const SelectorCandidateMetrics* best_metric = nullptr;
  for (const auto& item : metrics) {
    if (!item.ok) {
      continue;
    }
    if (best_metric == nullptr) {
      best_metric = &item;
      continue;
    }
    if (std::abs(item.recall_at_k_in_window - best_metric->recall_at_k_in_window) > kNormEpsilon) {
      if (item.recall_at_k_in_window > best_metric->recall_at_k_in_window) {
        best_metric = &item;
      }
      continue;
    }
    if (std::abs(selector_quality_score(item) -
                 selector_quality_score(*best_metric)) > kNormEpsilon) {
      if (selector_quality_score(item) > selector_quality_score(*best_metric)) {
        best_metric = &item;
      }
      continue;
    }
    if (std::abs(item.objective_score - best_metric->objective_score) > kNormEpsilon) {
      if (item.objective_score > best_metric->objective_score) {
        best_metric = &item;
      }
      continue;
    }
    if (item.latency_avg_ms + kNormEpsilon < best_metric->latency_avg_ms) {
      best_metric = &item;
    }
  }
  if (best_metric != nullptr) {
    anchor = best_metric->config;
  }
  return anchor;
}

SelectorCandidateConfig choose_quality_anchor(
    const std::vector<SelectorCandidateMetrics>& metrics,
    const SelectorCandidateConfig& fallback) {
  SelectorCandidateConfig anchor = fallback;
  const SelectorCandidateMetrics* best_metric = nullptr;
  for (const auto& item : metrics) {
    if (!item.ok) {
      continue;
    }
    if (best_metric == nullptr) {
      best_metric = &item;
      continue;
    }
    if (std::abs(selector_quality_score(item) -
                 selector_quality_score(*best_metric)) > kNormEpsilon) {
      if (selector_quality_score(item) > selector_quality_score(*best_metric)) {
        best_metric = &item;
      }
      continue;
    }
    if (std::abs(item.recall_at_k_in_window -
                 best_metric->recall_at_k_in_window) > kNormEpsilon) {
      if (item.recall_at_k_in_window > best_metric->recall_at_k_in_window) {
        best_metric = &item;
      }
      continue;
    }
    if (std::abs(item.objective_score - best_metric->objective_score) > kNormEpsilon) {
      if (item.objective_score > best_metric->objective_score) {
        best_metric = &item;
      }
      continue;
    }
    if (item.latency_avg_ms + kNormEpsilon < best_metric->latency_avg_ms) {
      best_metric = &item;
    }
  }
  if (best_metric != nullptr) {
    anchor = best_metric->config;
  }
  return anchor;
}

std::vector<SelectorCandidateMetrics> refine_phase2_for_recall(
    const std::vector<SelectorCandidateMetrics>& seed_metrics,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    const EvalDataset& full_eval,
    const PhaseLimits& phase2_limits,
    std::size_t skeleton_count,
    TrainCache* cache,
    SearchController* search,
    uint32_t* rounds_out) {
  std::vector<SelectorCandidateMetrics> extra;
  if (!options.optimize_for_recall || seed_metrics.empty()) {
    return extra;
  }
  double best_recall = best_recall_in_metrics(seed_metrics);
  if (best_recall < 0.0) {
    return extra;
  }
  SelectorCandidateConfig anchor = choose_recall_anchor(seed_metrics, seed_metrics.front().config);
  std::unordered_set<std::string> seen_keys;
  seen_keys.reserve(seed_metrics.size() * 2 + 16);
  for (const auto& item : seed_metrics) {
    seen_keys.insert(selector_config_key(item.config));
  }

  uint32_t rounds = 0;
  for (uint32_t round = 0; round < options.max_recall_refine_rounds; ++round) {
    if (search != nullptr && !search->can_launch_more_work()) {
      break;
    }
    uint32_t radius = std::min<uint32_t>(4, 2 + round);
    uint32_t neighbor_limit =
        std::max<uint32_t>(options.beam_neighbor_limit * 2, options.max_phase2_candidates * 2);
    std::vector<SelectorCandidateConfig> neighbors =
        local_neighbors(anchor, options, skeleton_count, radius, neighbor_limit);

    std::vector<SelectorCandidateConfig> frontier;
    frontier.reserve(neighbors.size());
    for (const auto& cand : neighbors) {
      std::string key = selector_config_key(cand);
      if (!seen_keys.insert(key).second) {
        continue;
      }
      frontier.push_back(cand);
    }
    if (frontier.empty()) {
      break;
    }
    frontier = fit_candidates_to_remaining_train_budget(frontier, options, context, cache);
    if (frontier.empty()) {
      break;
    }

    std::vector<SelectorCandidateMetrics> metrics =
        evaluate_candidates(frontier,
                            options,
                            context,
                            full_eval,
                            phase2_limits,
                            cache,
                            search);
    assign_objective_scores(&metrics, options.weights);
    rounds += 1;

    double round_best_recall = best_recall_in_metrics(metrics);
    if (round_best_recall > best_recall + options.min_recall_improvement) {
      best_recall = round_best_recall;
      anchor = choose_recall_anchor(metrics, anchor);
    } else {
      extra.insert(extra.end(), metrics.begin(), metrics.end());
      break;
    }

    extra.insert(extra.end(), metrics.begin(), metrics.end());
  }

  if (rounds_out != nullptr) {
    *rounds_out = rounds;
  }
  return extra;
}

} // namespace vortex::selector_internal
