#include "selector_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace vortex::selector_internal {

double selector_quality_score(const SelectorCandidateMetrics& item) {
  if (item.locality_quality_score > 0.0) {
    return item.locality_quality_score;
  }
  return item.recall_at_k_in_window;
}

double normalize(double value, double min_value, double max_value) {
  if (max_value - min_value < kNormEpsilon) {
    return 0.5;
  }
  return (value - min_value) / (max_value - min_value);
}

void assign_objective_scores(std::vector<SelectorCandidateMetrics>* metrics,
                             const SelectorWeights& weights) {
  std::vector<std::size_t> ok_indices;
  for (std::size_t i = 0; i < metrics->size(); ++i) {
    if ((*metrics)[i].ok) {
      ok_indices.push_back(i);
    } else {
      (*metrics)[i].objective_score = -1.0;
    }
  }
  if (ok_indices.empty()) {
    return;
  }

  double min_quality = std::numeric_limits<double>::infinity();
  double max_quality = -std::numeric_limits<double>::infinity();
  double min_rank = std::numeric_limits<double>::infinity();
  double max_rank = -std::numeric_limits<double>::infinity();
  double min_latency = std::numeric_limits<double>::infinity();
  double max_latency = -std::numeric_limits<double>::infinity();
  double min_size = std::numeric_limits<double>::infinity();
  double max_size = -std::numeric_limits<double>::infinity();
  double min_train = std::numeric_limits<double>::infinity();
  double max_train = -std::numeric_limits<double>::infinity();

  for (std::size_t idx : ok_indices) {
    const auto& item = (*metrics)[idx];
    double quality = selector_quality_score(item);
    min_quality = std::min(min_quality, quality);
    max_quality = std::max(max_quality, quality);
    min_rank = std::min(min_rank, item.mean_rank_distance_norm);
    max_rank = std::max(max_rank, item.mean_rank_distance_norm);
    min_latency = std::min(min_latency, item.latency_avg_ms);
    max_latency = std::max(max_latency, item.latency_avg_ms);
    min_size = std::min(min_size, static_cast<double>(item.model_size_bytes));
    max_size = std::max(max_size, static_cast<double>(item.model_size_bytes));
    min_train = std::min(min_train, item.train_time_ms);
    max_train = std::max(max_train, item.train_time_ms);
  }

  double weight_sum = weights.recall + weights.rank_distance + weights.latency +
                      weights.model_size + weights.train_time;
  if (weight_sum <= 0.0) {
    for (std::size_t idx : ok_indices) {
      (*metrics)[idx].objective_score = 0.0;
    }
    return;
  }

  for (std::size_t idx : ok_indices) {
    auto& item = (*metrics)[idx];
    double recall_score = normalize(selector_quality_score(item), min_quality, max_quality);
    double rank_score = 1.0 - normalize(item.mean_rank_distance_norm, min_rank, max_rank);
    double latency_score = 1.0 - normalize(item.latency_avg_ms, min_latency, max_latency);
    double size_score =
        1.0 - normalize(static_cast<double>(item.model_size_bytes), min_size, max_size);
    double train_score = 1.0 - normalize(item.train_time_ms, min_train, max_train);

    item.objective_score =
        (weights.recall * recall_score +
         weights.rank_distance * rank_score +
         weights.latency * latency_score +
         weights.model_size * size_score +
         weights.train_time * train_score) /
        weight_sum;
  }
}

bool dominates(const SelectorCandidateMetrics& a, const SelectorCandidateMetrics& b) {
  if (!a.ok || !b.ok) {
    return false;
  }
  bool at_least_as_good =
      selector_quality_score(a) >= selector_quality_score(b) - kNormEpsilon &&
      a.mean_rank_distance_norm <= b.mean_rank_distance_norm + kNormEpsilon &&
      a.latency_avg_ms <= b.latency_avg_ms + kNormEpsilon &&
      a.model_size_bytes <= b.model_size_bytes &&
      a.train_time_ms <= b.train_time_ms + kNormEpsilon;
  bool strictly_better =
      selector_quality_score(a) > selector_quality_score(b) + kNormEpsilon ||
      a.mean_rank_distance_norm + kNormEpsilon < b.mean_rank_distance_norm ||
      a.latency_avg_ms + kNormEpsilon < b.latency_avg_ms ||
      a.model_size_bytes < b.model_size_bytes ||
      a.train_time_ms + kNormEpsilon < b.train_time_ms;
  return at_least_as_good && strictly_better;
}

std::vector<SelectorCandidateMetrics> pareto_front(
    const std::vector<SelectorCandidateMetrics>& metrics) {
  std::vector<SelectorCandidateMetrics> front;
  for (std::size_t i = 0; i < metrics.size(); ++i) {
    if (!metrics[i].ok) {
      continue;
    }
    bool dominated_flag = false;
    for (std::size_t j = 0; j < metrics.size(); ++j) {
      if (i == j) {
        continue;
      }
      if (dominates(metrics[j], metrics[i])) {
        dominated_flag = true;
        break;
      }
    }
    if (!dominated_flag) {
      front.push_back(metrics[i]);
    }
  }
  std::sort(front.begin(), front.end(),
            [](const SelectorCandidateMetrics& a, const SelectorCandidateMetrics& b) {
              if (std::abs(selector_quality_score(a) -
                           selector_quality_score(b)) > kNormEpsilon) {
                return selector_quality_score(a) > selector_quality_score(b);
              }
              if (std::abs(a.objective_score - b.objective_score) > kNormEpsilon) {
                return a.objective_score > b.objective_score;
              }
              return a.latency_avg_ms < b.latency_avg_ms;
            });
  return front;
}

SelectorRecommendationRoles choose_recommendation_roles(
    const std::vector<SelectorCandidateMetrics>& metrics,
    const std::optional<uint64_t>& size_budget_bytes,
    const std::optional<double>& recall_target) {
  std::vector<SelectorCandidateMetrics> ranked;
  for (const auto& item : metrics) {
    if (!item.ok) {
      continue;
    }
    if (size_budget_bytes.has_value() && item.model_size_bytes > *size_budget_bytes) {
      continue;
    }
    ranked.push_back(item);
  }
  if (ranked.empty() && size_budget_bytes.has_value()) {
    return choose_recommendation_roles(metrics, std::nullopt, recall_target);
  }
  SelectorRecommendationRoles roles;
  if (ranked.empty()) {
    return roles;
  }

  auto target_first = [&](const SelectorCandidateMetrics& a,
                          const SelectorCandidateMetrics& b) {
    if (!recall_target.has_value()) {
      return false;
    }
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
    return false;
  };

  roles.peak_recall = *std::max_element(
      ranked.begin(), ranked.end(),
      [&](const SelectorCandidateMetrics& a, const SelectorCandidateMetrics& b) {
        if (target_first(a, b)) {
          return false;
        }
        if (target_first(b, a)) {
          return true;
        }
        if (std::abs(a.recall_at_k_in_window - b.recall_at_k_in_window) > kNormEpsilon) {
          return a.recall_at_k_in_window < b.recall_at_k_in_window;
        }
        if (std::abs(a.recall_auc_log_window - b.recall_auc_log_window) > kNormEpsilon) {
          return a.recall_auc_log_window < b.recall_auc_log_window;
        }
        if (std::abs(selector_quality_score(a) -
                     selector_quality_score(b)) > kNormEpsilon) {
          return selector_quality_score(a) < selector_quality_score(b);
        }
        return a.latency_avg_ms > b.latency_avg_ms;
      });

  double min_quality = std::numeric_limits<double>::infinity();
  double max_quality = -std::numeric_limits<double>::infinity();
  double min_latency = std::numeric_limits<double>::infinity();
  double max_latency = -std::numeric_limits<double>::infinity();
  double min_size = std::numeric_limits<double>::infinity();
  double max_size = -std::numeric_limits<double>::infinity();
  for (const auto& item : ranked) {
    min_quality = std::min(min_quality, selector_quality_score(item));
    max_quality = std::max(max_quality, selector_quality_score(item));
    min_latency = std::min(min_latency, item.latency_avg_ms);
    max_latency = std::max(max_latency, item.latency_avg_ms);
    min_size = std::min(min_size, static_cast<double>(item.model_size_bytes));
    max_size = std::max(max_size, static_cast<double>(item.model_size_bytes));
  }

  auto knee_score = [&](const SelectorCandidateMetrics& item) {
    double quality = normalize(selector_quality_score(item), min_quality, max_quality);
    double latency = normalize(item.latency_avg_ms, min_latency, max_latency);
    double size = normalize(static_cast<double>(item.model_size_bytes), min_size, max_size);
    return quality - 0.16 * latency - 0.10 * size;
  };
  roles.knee = *std::max_element(
      ranked.begin(), ranked.end(),
      [&](const SelectorCandidateMetrics& a, const SelectorCandidateMetrics& b) {
        if (target_first(a, b)) {
          return false;
        }
        if (target_first(b, a)) {
          return true;
        }
        double a_score = knee_score(a);
        double b_score = knee_score(b);
        if (std::abs(a_score - b_score) > kNormEpsilon) {
          return a_score < b_score;
        }
        return selector_quality_score(a) < selector_quality_score(b);
      });

  double quality_floor = std::max(0.0, max_quality - 0.02);
  auto useful = [&](const SelectorCandidateMetrics& item) {
    if (recall_target.has_value() && meets_recall_target(item, *recall_target)) {
      return true;
    }
    return selector_quality_score(item) + kNormEpsilon >= quality_floor;
  };
  std::vector<SelectorCandidateMetrics> useful_ranked;
  for (const auto& item : ranked) {
    if (useful(item)) {
      useful_ranked.push_back(item);
    }
  }
  if (useful_ranked.empty()) {
    useful_ranked = ranked;
  }
  roles.fast = *std::min_element(
      useful_ranked.begin(), useful_ranked.end(),
      [](const SelectorCandidateMetrics& a, const SelectorCandidateMetrics& b) {
        if (std::abs(a.latency_avg_ms - b.latency_avg_ms) > kNormEpsilon) {
          return a.latency_avg_ms < b.latency_avg_ms;
        }
        return selector_quality_score(a) > selector_quality_score(b);
      });
  roles.small = *std::min_element(
      useful_ranked.begin(), useful_ranked.end(),
      [](const SelectorCandidateMetrics& a, const SelectorCandidateMetrics& b) {
        if (a.model_size_bytes != b.model_size_bytes) {
          return a.model_size_bytes < b.model_size_bytes;
        }
        return selector_quality_score(a) > selector_quality_score(b);
      });
  return roles;
}

std::vector<SelectorCandidateMetrics> top_ok_candidates(
    const std::vector<SelectorCandidateMetrics>& metrics,
    std::size_t limit,
    const std::optional<uint64_t>& size_budget_bytes,
    const std::optional<double>& recall_target) {
  std::vector<SelectorCandidateMetrics> ranked;
  for (const auto& item : metrics) {
    if (!item.ok) {
      continue;
    }
    if (size_budget_bytes.has_value() && item.model_size_bytes > *size_budget_bytes) {
      continue;
    }
    ranked.push_back(item);
  }
  if (recall_target.has_value()) {
    double target = *recall_target;
    std::stable_sort(ranked.begin(), ranked.end(),
                     [target](const SelectorCandidateMetrics& a,
                              const SelectorCandidateMetrics& b) {
                       bool a_hit = meets_recall_target(a, target);
                       bool b_hit = meets_recall_target(b, target);
                       if (a_hit != b_hit) {
                         return a_hit;
                       }
                       if (a_hit && b_hit) {
                         if (a.min_window_for_recall_target != b.min_window_for_recall_target) {
                           if (a.min_window_for_recall_target == 0) {
                             return false;
                           }
                           if (b.min_window_for_recall_target == 0) {
                             return true;
                           }
                           return a.min_window_for_recall_target < b.min_window_for_recall_target;
                         }
                         if (std::abs(a.objective_score - b.objective_score) > kNormEpsilon) {
                           return a.objective_score > b.objective_score;
                         }
                         if (std::abs(a.latency_avg_ms - b.latency_avg_ms) > kNormEpsilon) {
                           return a.latency_avg_ms < b.latency_avg_ms;
                         }
                         if (a.model_size_bytes != b.model_size_bytes) {
                           return a.model_size_bytes < b.model_size_bytes;
                         }
                         return selector_quality_score(a) > selector_quality_score(b);
                       }

                       if (std::abs(selector_quality_score(a) -
                                    selector_quality_score(b)) > kNormEpsilon) {
                         return selector_quality_score(a) > selector_quality_score(b);
                       }
                       if (std::abs(a.objective_score - b.objective_score) > kNormEpsilon) {
                         return a.objective_score > b.objective_score;
                       }
                       if (std::abs(a.latency_avg_ms - b.latency_avg_ms) > kNormEpsilon) {
                         return a.latency_avg_ms < b.latency_avg_ms;
                       }
                       return a.model_size_bytes < b.model_size_bytes;
                     });
  } else {
    std::sort(ranked.begin(), ranked.end(),
              [](const SelectorCandidateMetrics& a, const SelectorCandidateMetrics& b) {
                if (std::abs(selector_quality_score(a) -
                             selector_quality_score(b)) > kNormEpsilon) {
                  return selector_quality_score(a) > selector_quality_score(b);
                }
                if (std::abs(a.objective_score - b.objective_score) > kNormEpsilon) {
                  return a.objective_score > b.objective_score;
                }
                return a.latency_avg_ms < b.latency_avg_ms;
              });
  }
  if (ranked.size() > limit) {
    ranked.resize(limit);
  }
  return ranked;
}

} // namespace vortex::selector_internal
