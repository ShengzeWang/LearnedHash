#include "selector_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace vortex::selector_internal {

double elapsed_ms(SelectorClock::time_point start) {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::microseconds>(
                 SelectorClock::now() - start)
                 .count()) /
         1000.0;
}

void append_timing(std::vector<SelectorPhaseTiming>* timings,
                   const std::string& name,
                   SelectorClock::time_point start) {
  timings->push_back(SelectorPhaseTiming{name, elapsed_ms(start)});
}

uint32_t count_ok_metrics(const std::vector<SelectorCandidateMetrics>& metrics) {
  return static_cast<uint32_t>(
      std::count_if(metrics.begin(), metrics.end(), [](const auto& item) {
        return item.ok;
      }));
}

bool is_budget_failure(const SelectorCandidateMetrics& metrics) {
  return !metrics.error.empty() &&
         metrics.error.find("budget reached") != std::string::npos;
}

uint32_t count_budget_failures(const std::vector<SelectorCandidateMetrics>& metrics) {
  return static_cast<uint32_t>(
      std::count_if(metrics.begin(), metrics.end(), is_budget_failure));
}

uint32_t read_model_trains_started(TrainCache* cache) {
  std::lock_guard<std::mutex> lock(cache->mutex);
  return cache->model_trains_started;
}

bool read_context_budget_hit(TrainCache* cache) {
  std::lock_guard<std::mutex> lock(cache->mutex);
  return cache->context_model_train_budget_hit;
}

std::string calibration_seed_key(const SelectorCandidateMetrics& item) {
  return item.nsw_path + "||" + selector_config_key(item.config);
}

uint32_t requested_context_count(const std::vector<NswSelectorContext>& contexts) {
  uint64_t total = 0;
  for (const auto& context : contexts) {
    total += std::max<std::size_t>(1, context.requested_target_skeletons.size());
  }
  return static_cast<uint32_t>(
      std::min<uint64_t>(total, std::numeric_limits<uint32_t>::max()));
}

std::vector<SelectorCandidateConfig> fit_candidates_to_remaining_train_budget(
    const std::vector<SelectorCandidateConfig>& candidates,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    TrainCache* cache) {
  if (cache == nullptr || candidates.empty()) {
    return candidates;
  }

  std::vector<SelectorCandidateConfig> out;
  out.reserve(candidates.size());
  std::lock_guard<std::mutex> lock(cache->mutex);
  uint32_t global_remaining = std::numeric_limits<uint32_t>::max();
  if (cache->max_model_trains > 0) {
    if (cache->model_trains_started >= cache->max_model_trains) {
      global_remaining = 0;
    } else {
      global_remaining = cache->max_model_trains - cache->model_trains_started;
    }
  }
  uint32_t context_remaining = std::numeric_limits<uint32_t>::max();
  if (cache->context_model_train_limit > 0) {
    uint32_t context_started = 0;
    if (cache->model_trains_started >= cache->context_model_train_start) {
      context_started = cache->model_trains_started - cache->context_model_train_start;
    }
    context_remaining = context_started >= cache->context_model_train_limit
        ? 0
        : cache->context_model_train_limit - context_started;
  }
  uint32_t new_train_slots = std::min(global_remaining, context_remaining);
  bool pruned_for_budget = false;
  for (const auto& candidate : candidates) {
    std::string key = selector_training_cache_key(context, candidate, options);
    if (cache->future_by_key.find(key) != cache->future_by_key.end()) {
      out.push_back(candidate);
      continue;
    }
    if (new_train_slots == 0) {
      pruned_for_budget = true;
      continue;
    }
    out.push_back(candidate);
    if (new_train_slots != std::numeric_limits<uint32_t>::max()) {
      new_train_slots -= 1;
    }
  }
  if (pruned_for_budget) {
    if (global_remaining != std::numeric_limits<uint32_t>::max() &&
        global_remaining <= context_remaining) {
      cache->model_train_budget_hit = true;
    }
    if (context_remaining != std::numeric_limits<uint32_t>::max() &&
        context_remaining <= global_remaining) {
      cache->context_model_train_budget_hit = true;
    }
  }
  return out;
}

uint32_t remaining_model_train_budget(TrainCache* cache) {
  std::lock_guard<std::mutex> lock(cache->mutex);
  if (cache->max_model_trains == 0) {
    return std::numeric_limits<uint32_t>::max();
  }
  if (cache->model_trains_started >= cache->max_model_trains) {
    return 0;
  }
  return cache->max_model_trains - cache->model_trains_started;
}

uint32_t fair_context_train_budget(TrainCache* cache,
                                   std::size_t remaining_contexts,
                                   uint32_t phase1_keep,
                                   uint32_t post_exploration_reserve) {
  if (cache == nullptr || cache->max_model_trains == 0) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(cache->mutex);
  if (cache->model_trains_started >= cache->max_model_trains) {
    return 0;
  }
  uint32_t remaining = cache->max_model_trains - cache->model_trains_started;
  uint32_t remaining_contexts_u32 = static_cast<uint32_t>(
      std::min<std::size_t>(remaining_contexts, std::numeric_limits<uint32_t>::max()));
  if (post_exploration_reserve > 0 &&
      remaining > post_exploration_reserve + remaining_contexts_u32) {
    remaining -= post_exploration_reserve;
  }
  if (remaining_contexts <= 1) {
    return remaining;
  }
  uint32_t equal_share =
      std::max<uint32_t>(1, (remaining + remaining_contexts_u32 - 1) / remaining_contexts_u32);
  uint32_t minimum_useful =
      std::min<uint32_t>(remaining, std::max<uint32_t>(8, phase1_keep * 3));
  uint32_t budget = equal_share;
  if (remaining / remaining_contexts_u32 >= minimum_useful) {
    budget = minimum_useful;
  }
  uint32_t reserve_for_later = remaining_contexts_u32 > 1 ? remaining_contexts_u32 - 1 : 0;
  uint32_t max_for_context =
      remaining > reserve_for_later ? remaining - reserve_for_later : 1;
  return std::min<uint32_t>(budget, max_for_context);
}

const NswSelectorContext* find_context_for_metric(
    const std::vector<NswSelectorContext>& contexts,
    const SelectorCandidateMetrics& metric) {
  for (const auto& context : contexts) {
    if (!metric.nsw_path.empty() &&
        (metric.nsw_path == context.display_nsw_path ||
         metric.nsw_path == context.nsw_path.string())) {
      return &context;
    }
  }
  if (metric.config.target_skeleton > 0) {
    for (const auto& context : contexts) {
      if (context.target_skeleton == metric.config.target_skeleton) {
        return &context;
      }
    }
  }
  return contexts.empty() ? nullptr : &contexts.front();
}

std::vector<SelectorCandidateMetrics> choose_final_calibration_seeds(
    const std::vector<SelectorCandidateMetrics>& pool,
    uint32_t limit,
    const std::optional<double>& recall_target) {
  if (limit == 0) {
    return {};
  }
  std::vector<SelectorCandidateMetrics> ranked =
      top_ok_candidates(pool, pool.size(), std::nullopt, recall_target);
  std::vector<SelectorCandidateMetrics> out;
  out.reserve(std::min<std::size_t>(ranked.size(), limit));
  std::unordered_set<std::string> seen_configs;
  seen_configs.reserve(limit * 2 + 1);
  std::unordered_set<uint64_t> seen_targets;
  seen_targets.reserve(limit * 2 + 1);

  auto append_if_new = [&](const SelectorCandidateMetrics& item) {
    if (out.size() >= limit) {
      return;
    }
    std::string key = calibration_seed_key(item);
    if (!seen_configs.insert(key).second) {
      return;
    }
    out.push_back(item);
  };

  for (const auto& item : ranked) {
    uint64_t target = item.config.target_skeleton;
    if (target == 0 || !seen_targets.insert(target).second) {
      continue;
    }
    append_if_new(item);
    if (out.size() >= limit) {
      return out;
    }
  }
  for (const auto& item : ranked) {
    append_if_new(item);
    if (out.size() >= limit) {
      break;
    }
  }
  return out;
}

std::vector<SelectorCandidateMetrics> choose_role_aware_final_calibration_seeds(
    const std::vector<SelectorCandidateMetrics>& screen_metrics,
    const std::vector<SelectorCandidateMetrics>& fallback_seeds,
    uint32_t limit,
    const std::optional<uint64_t>& size_budget_bytes,
    const std::optional<double>& recall_target,
    uint32_t* role_promoted,
    uint32_t* recall_promoted) {
  if (role_promoted != nullptr) {
    *role_promoted = 0;
  }
  if (recall_promoted != nullptr) {
    *recall_promoted = 0;
  }
  if (limit == 0) {
    return {};
  }

  std::vector<SelectorCandidateMetrics> promoted;
  promoted.reserve(limit);
  std::unordered_set<std::string> seen;
  seen.reserve(limit * 2 + 1);

  auto append_if_new = [&](const SelectorCandidateMetrics& item,
                           uint32_t* counter) {
    if (!item.ok || promoted.size() >= limit) {
      return false;
    }
    if (!seen.insert(calibration_seed_key(item)).second) {
      return false;
    }
    promoted.push_back(item);
    if (counter != nullptr) {
      *counter += 1;
    }
    return true;
  };

  SelectorRecommendationRoles roles =
      choose_recommendation_roles(screen_metrics, size_budget_bytes, recall_target);
  const std::optional<SelectorCandidateMetrics>* ordered_roles[] = {
      &roles.peak_recall,
      &roles.knee,
      &roles.fast,
      &roles.small,
  };
  for (const auto* role : ordered_roles) {
    if (role->has_value()) {
      append_if_new(**role, role_promoted);
    }
  }

  std::vector<SelectorCandidateMetrics> recall_ranked;
  auto collect_recall_candidates = [&](const std::optional<uint64_t>& budget) {
    recall_ranked.clear();
    for (const auto& item : screen_metrics) {
      if (!item.ok) {
        continue;
      }
      if (budget.has_value() && item.model_size_bytes > *budget) {
        continue;
      }
      recall_ranked.push_back(item);
    }
  };
  collect_recall_candidates(size_budget_bytes);
  if (recall_ranked.empty() && size_budget_bytes.has_value()) {
    collect_recall_candidates(std::nullopt);
  }

  std::stable_sort(
      recall_ranked.begin(),
      recall_ranked.end(),
      [&](const SelectorCandidateMetrics& a, const SelectorCandidateMetrics& b) {
        if (recall_target.has_value()) {
          bool a_hit = meets_recall_target(a, *recall_target);
          bool b_hit = meets_recall_target(b, *recall_target);
          if (a_hit != b_hit) {
            return a_hit;
          }
          if (a_hit && b_hit &&
              a.min_window_for_recall_target != b.min_window_for_recall_target) {
            if (a.min_window_for_recall_target == 0) {
              return false;
            }
            if (b.min_window_for_recall_target == 0) {
              return true;
            }
            return a.min_window_for_recall_target < b.min_window_for_recall_target;
          }
        }
        if (std::abs(a.recall_at_k_in_window - b.recall_at_k_in_window) >
            kNormEpsilon) {
          return a.recall_at_k_in_window > b.recall_at_k_in_window;
        }
        if (std::abs(a.recall_auc_log_window - b.recall_auc_log_window) >
            kNormEpsilon) {
          return a.recall_auc_log_window > b.recall_auc_log_window;
        }
        if (std::abs(a.query_recall_p05 - b.query_recall_p05) > kNormEpsilon) {
          return a.query_recall_p05 > b.query_recall_p05;
        }
        if (std::abs(a.mean_rank_distance_norm - b.mean_rank_distance_norm) >
            kNormEpsilon) {
          return a.mean_rank_distance_norm < b.mean_rank_distance_norm;
        }
        if (std::abs(selector_quality_score(a) - selector_quality_score(b)) >
            kNormEpsilon) {
          return selector_quality_score(a) > selector_quality_score(b);
        }
        if (std::abs(a.latency_avg_ms - b.latency_avg_ms) > kNormEpsilon) {
          return a.latency_avg_ms < b.latency_avg_ms;
        }
        return a.model_size_bytes < b.model_size_bytes;
      });
  for (const auto& item : recall_ranked) {
    append_if_new(item, recall_promoted);
    if (promoted.size() >= limit) {
      return promoted;
    }
  }

  for (const auto& item : fallback_seeds) {
    append_if_new(item, nullptr);
    if (promoted.size() >= limit) {
      break;
    }
  }
  return promoted;
}

std::vector<SelectorCandidateMetrics> run_post_exploration(
    const std::vector<SelectorCandidateMetrics>& scored_pool,
    const std::vector<NswSelectorContext>& contexts,
    const SelectorOptions& eval_options,
    std::size_t max_skeleton_count,
    const EvalDataset& full_eval,
    const PhaseLimits& phase2_limits,
    uint32_t candidate_limit,
    TrainCache* train_cache,
    SearchController* search_controller) {
  if (candidate_limit == 0 || !has_valid_metrics(scored_pool)) {
    return {};
  }
  uint32_t remaining_budget = remaining_model_train_budget(train_cache);
  if (remaining_budget == 0) {
    return {};
  }
  if (remaining_budget != std::numeric_limits<uint32_t>::max()) {
    candidate_limit = std::min<uint32_t>(candidate_limit, remaining_budget);
  }

  std::vector<SelectorCandidateMetrics> ranked =
      top_ok_candidates(scored_pool, scored_pool.size(), std::nullopt, eval_options.recall_target);
  std::vector<const NswSelectorContext*> ordered_contexts;
  ordered_contexts.reserve(contexts.size());
  std::unordered_set<std::string> seen_contexts;
  seen_contexts.reserve(contexts.size() * 2 + 1);
  for (const auto& item : ranked) {
    const NswSelectorContext* context = find_context_for_metric(contexts, item);
    if (context == nullptr) {
      continue;
    }
    std::string key = context->nsw_path.string();
    if (seen_contexts.insert(key).second) {
      ordered_contexts.push_back(context);
    }
  }

  std::vector<SelectorCandidateMetrics> out;
  out.reserve(candidate_limit);
  PhaseLimits exploit_limits = phase2_limits;
  exploit_limits.name = "post_exploration";

  for (const NswSelectorContext* context : ordered_contexts) {
    if (out.size() >= candidate_limit ||
        (search_controller != nullptr && !search_controller->can_launch_more_work())) {
      break;
    }
    std::vector<SelectorCandidateMetrics> context_pool;
    context_pool.reserve(ranked.size());
    for (const auto& item : ranked) {
      const NswSelectorContext* item_context = find_context_for_metric(contexts, item);
      if (item_context == context) {
        context_pool.push_back(item);
      }
    }
    if (context_pool.empty()) {
      continue;
    }

    SelectorOptions context_options =
        normalize_selector_options(eval_options, context->csr.node_ids.size());
    context_options.nsw_path = context->nsw_path;
    context_options = apply_context_capacity_scaling(context_options,
                                                     context->csr.node_ids.size(),
                                                     max_skeleton_count);
    context_options = normalize_selector_options(context_options, context->csr.node_ids.size());
    context_options.nsw_path = context->nsw_path;

    uint32_t remaining_slots =
        static_cast<uint32_t>(candidate_limit - static_cast<uint32_t>(out.size()));
    std::vector<SelectorCandidateConfig> candidates =
        build_post_exploration_candidates(context_pool,
                                          context_options,
                                          context->csr.node_ids.size(),
                                          remaining_slots);
    if (candidates.empty()) {
      continue;
    }
    std::vector<SelectorCandidateMetrics> metrics =
        evaluate_candidates(candidates,
                            context_options,
                            *context,
                            full_eval,
                            exploit_limits,
                            train_cache,
                            search_controller);
    assign_objective_scores(&metrics, eval_options.weights);
    out.insert(out.end(), metrics.begin(), metrics.end());
  }
  return out;
}


} // namespace vortex::selector_internal
