#include "selector_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vortex::selector_internal {

std::vector<SelectorCandidateConfig> collect_seed_configs(
    const std::vector<SelectorCandidateMetrics>& ranked,
    std::size_t keep) {
  std::vector<SelectorCandidateConfig> seeds;
  seeds.reserve(std::min<std::size_t>(ranked.size(), keep));
  for (const auto& item : ranked) {
    if (!item.ok) {
      continue;
    }
    seeds.push_back(item.config);
    if (seeds.size() >= keep) {
      break;
    }
  }
  return seeds;
}

std::size_t nearest_index_u32(const std::vector<uint32_t>& values, uint32_t value) {
  if (values.empty()) {
    return 0;
  }
  auto it = std::lower_bound(values.begin(), values.end(), value);
  if (it == values.begin()) {
    return 0;
  }
  if (it == values.end()) {
    return values.size() - 1;
  }
  std::size_t hi = static_cast<std::size_t>(it - values.begin());
  std::size_t lo = hi - 1;
  uint32_t d_lo = values[lo] > value ? (values[lo] - value) : (value - values[lo]);
  uint32_t d_hi = values[hi] > value ? (values[hi] - value) : (value - values[hi]);
  return d_lo <= d_hi ? lo : hi;
}

std::size_t nearest_index_u64(const std::vector<uint64_t>& values, uint64_t value) {
  if (values.empty()) {
    return 0;
  }
  auto it = std::lower_bound(values.begin(), values.end(), value);
  if (it == values.begin()) {
    return 0;
  }
  if (it == values.end()) {
    return values.size() - 1;
  }
  std::size_t hi = static_cast<std::size_t>(it - values.begin());
  std::size_t lo = hi - 1;
  uint64_t d_lo = values[lo] > value ? (values[lo] - value) : (value - values[lo]);
  uint64_t d_hi = values[hi] > value ? (values[hi] - value) : (value - values[hi]);
  return d_lo <= d_hi ? lo : hi;
}

std::size_t index_of_or_zero(const std::vector<std::string>& values, const std::string& value) {
  auto it = std::find(values.begin(), values.end(), value);
  if (it == values.end()) {
    return 0;
  }
  return static_cast<std::size_t>(it - values.begin());
}

template <typename T>
std::vector<T> ring_neighbors(const std::vector<T>& values,
                              std::size_t center,
                              uint32_t radius) {
  std::vector<T> out;
  if (values.empty()) {
    return out;
  }
  if (center >= values.size()) {
    center = values.size() - 1;
  }
  out.push_back(values[center]);
  for (uint32_t step = 1; step <= radius; ++step) {
    if (center >= step) {
      out.push_back(values[center - step]);
    }
    if (center + step < values.size()) {
      out.push_back(values[center + step]);
    }
  }
  return out;
}

uint32_t choose_knn_for_k(const SelectorOptions& options,
                          uint32_t current_knn,
                          uint32_t k_value) {
  uint32_t max_knn = k_value > 1 ? (k_value - 1) : 1;
  uint32_t best = 1;
  uint32_t best_diff = std::numeric_limits<uint32_t>::max();
  bool found = false;
  for (uint32_t knn : options.centroid_knn_values) {
    if (knn == 0 || knn > max_knn) {
      continue;
    }
    uint32_t diff = knn > current_knn ? (knn - current_knn) : (current_knn - knn);
    if (!found || diff < best_diff) {
      best = knn;
      best_diff = diff;
      found = true;
    }
  }
  if (!found) {
    return max_knn;
  }
  return best;
}

SelectorCandidateConfig fixup_config_for_grid(SelectorCandidateConfig config,
                                              const SelectorOptions& options) {
  if (config.K < 2) {
    config.K = 2;
  }
  config.centroid_knn = choose_knn_for_k(options, config.centroid_knn, config.K);
  if (config.enable_2opt) {
    if (config.two_opt_iterations == 0) {
      config.two_opt_iterations =
          options.two_opt_iterations_values.empty() ? 1 : options.two_opt_iterations_values.front();
    }
  } else {
    config.two_opt_iterations = 1;
  }
  if (!options.enable_graph_centroid_order) {
    config.enable_graph_centroid_order = false;
  }
  return config;
}

std::vector<SelectorCandidateConfig> local_neighbors(
    const SelectorCandidateConfig& seed,
    const SelectorOptions& options,
    std::size_t skeleton_count,
    uint32_t radius,
    uint32_t neighbor_limit) {
  std::vector<SelectorCandidateConfig> out;
  uint32_t k_radius = radius;
  if (options.dataset_count_hint >= 500000 && options.K_values.size() > 2) {
    // Large-dataset K increases are expensive because a first-seen K requires a
    // fresh k-means base. Climb the K ladder one rung at a time; if the next K
    // wins, later beam/post-exploration rounds can continue upward.
    k_radius = std::min<uint32_t>(k_radius, 1);
  }
  std::vector<uint32_t> k_values = ring_neighbors(
      options.K_values, nearest_index_u32(options.K_values, seed.K), k_radius);
  std::vector<uint32_t> knn_values = ring_neighbors(
      options.centroid_knn_values,
      nearest_index_u32(options.centroid_knn_values, seed.centroid_knn),
      radius);
  std::vector<uint64_t> branch_values = ring_neighbors(
      options.cdf_branching_values,
      nearest_index_u64(options.cdf_branching_values, seed.cdf_branching_factor),
      radius);
  std::vector<std::string> spec_values = ring_neighbors(
      options.cdf_model_specs,
      index_of_or_zero(options.cdf_model_specs, seed.cdf_model_spec),
      1);
  std::vector<uint32_t> two_opt_values = ring_neighbors(
      options.two_opt_iterations_values,
      nearest_index_u32(options.two_opt_iterations_values, seed.two_opt_iterations),
      radius);

  auto push_candidate = [&](SelectorCandidateConfig cand) {
    cand.target_skeleton = seed.target_skeleton;
    out.push_back(fixup_config_for_grid(cand, options));
    if (options.enable_graph_centroid_order &&
        options.include_disable_graph_centroid_order) {
      cand.enable_graph_centroid_order = !cand.enable_graph_centroid_order;
      out.push_back(fixup_config_for_grid(cand, options));
    }
  };

  for (uint32_t k : k_values) {
    if (k == seed.K) {
      continue;
    }
    auto cand = seed;
    cand.K = k;
    push_candidate(cand);
  }
  for (uint32_t knn : knn_values) {
    if (knn == seed.centroid_knn) {
      continue;
    }
    auto cand = seed;
    cand.centroid_knn = knn;
    push_candidate(cand);
  }
  for (uint64_t branch : branch_values) {
    if (branch == seed.cdf_branching_factor) {
      continue;
    }
    auto cand = seed;
    cand.cdf_branching_factor = branch;
    push_candidate(cand);
  }
  for (const auto& spec : spec_values) {
    if (spec == seed.cdf_model_spec) {
      continue;
    }
    auto cand = seed;
    cand.cdf_model_spec = spec;
    push_candidate(cand);
  }
  if (seed.enable_2opt) {
    for (uint32_t iters : two_opt_values) {
      if (iters == seed.two_opt_iterations || iters == 0) {
        continue;
      }
      auto cand = seed;
      cand.two_opt_iterations = iters;
      cand.enable_2opt = true;
      push_candidate(cand);
    }
    if (options.include_disable_2opt) {
      auto cand = seed;
      cand.enable_2opt = false;
      cand.two_opt_iterations = 1;
      push_candidate(cand);
    }
  } else {
    auto cand = seed;
    cand.enable_2opt = true;
    cand.two_opt_iterations = options.two_opt_iterations_values.empty()
                                  ? 1
                                  : nearest_value(options.two_opt_iterations_values, 8);
    push_candidate(cand);
  }
  for (uint32_t k : k_values) {
    for (uint32_t knn : knn_values) {
      if (k == seed.K && knn == seed.centroid_knn) {
        continue;
      }
      auto cand = seed;
      cand.K = k;
      cand.centroid_knn = knn;
      push_candidate(cand);
    }
  }

  auto deduped = dedup_candidates(out, skeleton_count);
  if (deduped.size() > neighbor_limit) {
    deduped.resize(neighbor_limit);
  }
  return deduped;
}

struct Phase1KFamilyEvidence {
  bool seen = false;
  double best_quality = -std::numeric_limits<double>::infinity();
  double best_recall = -std::numeric_limits<double>::infinity();
};

std::vector<SelectorCandidateConfig> compact_attribution_guided_candidates(
    const std::vector<SelectorCandidateConfig>& candidates,
    const std::vector<SelectorCandidateMetrics>& phase1_metrics,
    const SelectorOptions& options,
    uint32_t weak_high_k_family_limit,
    uint32_t* pruned_out) {
  if (pruned_out != nullptr) {
    *pruned_out = 0;
  }
  if (!options.attribution_guided_compaction ||
      !options.optimize_for_recall ||
      options.dataset_count_hint < 500000 ||
      candidates.empty() ||
      weak_high_k_family_limit == 0) {
    return candidates;
  }

  std::unordered_map<uint32_t, Phase1KFamilyEvidence> evidence_by_k;
  double best_quality = -std::numeric_limits<double>::infinity();
  double best_recall = -std::numeric_limits<double>::infinity();
  uint32_t best_k = 0;
  for (const auto& item : phase1_metrics) {
    if (!item.ok) {
      continue;
    }
    auto& evidence = evidence_by_k[item.config.K];
    evidence.seen = true;
    evidence.best_quality = std::max(evidence.best_quality, selector_quality_score(item));
    evidence.best_recall = std::max(evidence.best_recall, item.recall_at_k_in_window);
    bool better = false;
    if (selector_quality_score(item) > best_quality + kNormEpsilon) {
      better = true;
    } else if (std::abs(selector_quality_score(item) - best_quality) <= kNormEpsilon &&
               item.recall_at_k_in_window > best_recall + kNormEpsilon) {
      better = true;
    } else if (std::abs(selector_quality_score(item) - best_quality) <= kNormEpsilon &&
               std::abs(item.recall_at_k_in_window - best_recall) <= kNormEpsilon &&
               (best_k == 0 || item.config.K < best_k)) {
      better = true;
    }
    if (better) {
      best_quality = selector_quality_score(item);
      best_recall = item.recall_at_k_in_window;
      best_k = item.config.K;
    }
  }
  if (best_k == 0 || evidence_by_k.size() < 2) {
    return candidates;
  }

  double quality_margin = std::max(0.015, options.min_quality_improvement * 8.0);
  double recall_margin = std::max(0.015, options.min_recall_improvement * 8.0);
  auto weak_high_k = [&](uint32_t k) {
    if (k <= best_k) {
      return false;
    }
    auto it = evidence_by_k.find(k);
    if (it == evidence_by_k.end() || !it->second.seen) {
      return true;
    }
    double quality_gap = best_quality - it->second.best_quality;
    double recall_gap = best_recall - it->second.best_recall;
    return quality_gap > quality_margin && recall_gap > recall_margin;
  };

  std::unordered_map<uint32_t, uint32_t> kept_by_k;
  std::vector<SelectorCandidateConfig> out;
  out.reserve(candidates.size());
  uint32_t pruned = 0;
  for (const auto& candidate : candidates) {
    if (!weak_high_k(candidate.K)) {
      out.push_back(candidate);
      continue;
    }
    uint32_t& kept = kept_by_k[candidate.K];
    if (kept < weak_high_k_family_limit) {
      out.push_back(candidate);
      kept += 1;
      continue;
    }
    pruned += 1;
  }
  if (pruned_out != nullptr) {
    *pruned_out = pruned;
  }
  return out;
}

std::vector<SelectorCandidateConfig> expand_phase2_candidates_grid(
    const std::vector<SelectorCandidateMetrics>& phase1_ranked,
    const SelectorOptions& options,
    std::size_t skeleton_count) {
  std::vector<SelectorCandidateConfig> seed_configs =
      collect_seed_configs(phase1_ranked, options.phase1_keep);
  if (seed_configs.empty()) {
    return {};
  }
  uint64_t target_skeleton = seed_configs.front().target_skeleton;

  std::set<uint32_t> k_values;
  std::set<uint32_t> knn_values;
  std::set<uint64_t> branch_values;
  std::set<std::string> spec_values;
  std::set<uint32_t> two_opt_values;
  std::set<bool> two_opt_enabled_values;
  std::set<bool> graph_order_values;
  for (const auto& config : seed_configs) {
    k_values.insert(config.K);
    knn_values.insert(config.centroid_knn);
    branch_values.insert(config.cdf_branching_factor);
    spec_values.insert(config.cdf_model_spec);
    two_opt_values.insert(config.two_opt_iterations);
    two_opt_enabled_values.insert(config.enable_2opt);
    graph_order_values.insert(config.enable_graph_centroid_order);
  }

  std::vector<uint32_t> k_top(k_values.begin(), k_values.end());
  std::vector<uint32_t> knn_top(knn_values.begin(), knn_values.end());
  std::vector<uint64_t> branch_top(branch_values.begin(), branch_values.end());
  std::vector<std::string> spec_top(spec_values.begin(), spec_values.end());
  std::vector<uint32_t> two_opt_top(two_opt_values.begin(), two_opt_values.end());
  std::vector<bool> two_opt_enabled_top(two_opt_enabled_values.begin(),
                                        two_opt_enabled_values.end());
  std::vector<bool> graph_order_top(graph_order_values.begin(),
                                    graph_order_values.end());

  if (k_top.size() < 2 && options.K_values.size() > 1) {
    k_top.push_back(options.K_values[std::min<std::size_t>(1, options.K_values.size() - 1)]);
  }
  if (knn_top.size() < 2 && options.centroid_knn_values.size() > 1) {
    knn_top.push_back(options.centroid_knn_values[std::min<std::size_t>(
        1, options.centroid_knn_values.size() - 1)]);
  }
  if (branch_top.size() < 2 && options.cdf_branching_values.size() > 1) {
    branch_top.push_back(options.cdf_branching_values[std::min<std::size_t>(
        1, options.cdf_branching_values.size() - 1)]);
  }
  if (spec_top.size() < 2 && options.cdf_model_specs.size() > 1) {
    spec_top.push_back(options.cdf_model_specs[std::min<std::size_t>(
        1, options.cdf_model_specs.size() - 1)]);
  }
  if (two_opt_top.empty()) {
    two_opt_top.push_back(1);
  }
  if (two_opt_enabled_top.empty()) {
    two_opt_enabled_top.push_back(true);
  }
  if (graph_order_top.empty()) {
    graph_order_top.push_back(options.enable_graph_centroid_order);
  }
  if (options.enable_graph_centroid_order &&
      options.include_disable_graph_centroid_order &&
      graph_order_top.size() < 2) {
    graph_order_top.push_back(false);
  }
  dedup_sort(&k_top);
  dedup_sort(&knn_top);
  dedup_sort(&branch_top);
  dedup_strings(&spec_top);
  dedup_sort(&two_opt_top);

  std::vector<SelectorCandidateConfig> expanded = seed_configs;
  for (uint32_t K : k_top) {
    for (uint32_t knn : knn_top) {
      for (uint64_t branch : branch_top) {
        for (const auto& spec : spec_top) {
          for (bool enable_graph_order : graph_order_top) {
            for (bool enable_2opt : two_opt_enabled_top) {
              if (enable_2opt) {
                for (uint32_t iters : two_opt_top) {
                  SelectorCandidateConfig config;
                  config.K = K;
                  config.centroid_knn = knn;
                  config.cdf_branching_factor = branch;
                  config.cdf_model_spec = spec;
                  config.enable_2opt = true;
                  config.two_opt_iterations = iters;
                  config.target_skeleton = target_skeleton;
                  config.enable_graph_centroid_order = enable_graph_order;
                  expanded.push_back(config);
                }
              } else {
                SelectorCandidateConfig config;
                config.K = K;
                config.centroid_knn = knn;
                config.cdf_branching_factor = branch;
                config.cdf_model_spec = spec;
                config.enable_2opt = false;
                config.two_opt_iterations = 1;
                config.target_skeleton = target_skeleton;
                config.enable_graph_centroid_order = enable_graph_order;
                expanded.push_back(config);
              }
            }
          }
        }
      }
    }
  }

  expanded = dedup_candidates(expanded, skeleton_count);
  if (expanded.size() > options.max_phase2_candidates) {
    expanded.resize(options.max_phase2_candidates);
  }
  return expanded;
}

std::vector<SelectorCandidateConfig> build_post_exploration_candidates(
    const std::vector<SelectorCandidateMetrics>& ranked,
    const SelectorOptions& options,
    std::size_t skeleton_count,
    uint32_t candidate_limit) {
  if (candidate_limit == 0 || ranked.empty()) {
    return {};
  }

  std::vector<SelectorCandidateMetrics> seeds =
      top_ok_candidates(ranked,
                        std::max<std::size_t>(options.phase1_keep,
                                              options.recommend_count),
                        std::nullopt,
                        options.recall_target);
  if (seeds.empty()) {
    return {};
  }

  std::unordered_set<std::string> seen;
  seen.reserve(ranked.size() * 2 + candidate_limit * 2 + 1);
  for (const auto& item : ranked) {
    seen.insert(selector_config_key(item.config));
  }

  std::vector<SelectorCandidateConfig> out;
  out.reserve(candidate_limit);
  uint32_t radius = options.optimize_for_recall ? 3 : 2;
  uint32_t neighbor_limit =
      std::max<uint32_t>(options.beam_neighbor_limit * 2, candidate_limit * 2);

  for (const auto& seed : seeds) {
    std::vector<SelectorCandidateConfig> neighbors =
        local_neighbors(seed.config, options, skeleton_count, radius, neighbor_limit);
    for (const auto& candidate : neighbors) {
      std::string key = selector_config_key(candidate);
      if (!seen.insert(key).second) {
        continue;
      }
      out.push_back(candidate);
      if (out.size() >= candidate_limit) {
        return dedup_candidates(out, skeleton_count);
      }
    }
  }

  return dedup_candidates(out, skeleton_count);
}

} // namespace vortex::selector_internal
