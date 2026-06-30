#include "selector_internal.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace vortex::selector_internal {

SelectorWeights default_selector_weights() {
  return SelectorWeights{};
}

bool nearly_equal(double lhs, double rhs) {
  return std::abs(lhs - rhs) <= 1e-12;
}

bool has_default_selector_weights(const SelectorWeights& weights) {
  const SelectorWeights defaults = default_selector_weights();
  return nearly_equal(weights.recall, defaults.recall) &&
         nearly_equal(weights.rank_distance, defaults.rank_distance) &&
         nearly_equal(weights.latency, defaults.latency) &&
         nearly_equal(weights.model_size, defaults.model_size) &&
         nearly_equal(weights.train_time, defaults.train_time);
}

SelectorWeights dataset_adaptive_weights(std::size_t dataset_count) {
  SelectorWeights tuned = default_selector_weights();
  if (dataset_count >= 2000000) {
    tuned.recall = 0.78;
    tuned.rank_distance = 0.10;
    tuned.latency = 0.09;
    tuned.model_size = 0.02;
    tuned.train_time = 0.01;
  } else if (dataset_count >= 500000) {
    tuned.recall = 0.75;
    tuned.rank_distance = 0.10;
    tuned.latency = 0.10;
    tuned.model_size = 0.03;
    tuned.train_time = 0.02;
  } else if (dataset_count >= 100000) {
    tuned.recall = 0.68;
    tuned.rank_distance = 0.10;
    tuned.latency = 0.14;
    tuned.model_size = 0.05;
    tuned.train_time = 0.03;
  }
  return tuned;
}

SelectorWeights recall_optimized_weights(std::size_t dataset_count) {
  SelectorWeights tuned;
  tuned.recall = 0.88;
  tuned.rank_distance = 0.08;
  tuned.latency = 0.03;
  tuned.model_size = 0.01;
  tuned.train_time = 0.00;
  if (dataset_count >= 500000) {
    tuned.recall = 0.90;
    tuned.rank_distance = 0.07;
    tuned.latency = 0.02;
    tuned.model_size = 0.01;
    tuned.train_time = 0.00;
  }
  return tuned;
}

SelectorWeights resolve_effective_weights(const SelectorOptions& options,
                                          std::size_t dataset_count,
                                          bool* applied_auto_weighting) {
  if (applied_auto_weighting != nullptr) {
    *applied_auto_weighting = false;
  }
  if (options.optimize_for_recall &&
      has_default_selector_weights(options.weights) &&
      options.auto_dataset_weighting) {
    if (applied_auto_weighting != nullptr) {
      *applied_auto_weighting = true;
    }
    return recall_optimized_weights(dataset_count);
  }
  if (!options.auto_dataset_weighting) {
    return options.weights;
  }
  if (!has_default_selector_weights(options.weights)) {
    return options.weights;
  }
  SelectorWeights tuned = dataset_adaptive_weights(dataset_count);
  if (!has_default_selector_weights(tuned) && applied_auto_weighting != nullptr) {
    *applied_auto_weighting = true;
  }
  return tuned;
}

std::string normalize_selector_profile(std::string value) {
  if (value.empty()) {
    return "auto";
  }
  std::transform(value.begin(),
                 value.end(),
                 value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (value == "auto" || value == "fast" || value == "balanced" || value == "exhaustive") {
    return value;
  }
  throw std::runtime_error("selector_profile must be one of: auto, fast, balanced, exhaustive");
}

std::string resolve_effective_selector_profile(const SelectorOptions& options,
                                               std::size_t dataset_count) {
  std::string requested = normalize_selector_profile(options.selector_profile);
  if (requested != "auto") {
    return requested;
  }
  if (dataset_count >= 3000000) {
    return "fast";
  }
  if (dataset_count >= 500000) {
    return "balanced";
  }
  return "exhaustive";
}

uint32_t auto_context_limit(const std::string& profile, std::size_t dataset_count) {
  if (profile == "exhaustive") {
    return 0;
  }
  if (profile == "fast") {
    return dataset_count >= 1000000 ? 3 : 4;
  }
  if (profile == "balanced") {
    return dataset_count >= 1000000 ? 4 : 6;
  }
  return 0;
}

uint32_t resolve_context_limit(const SelectorOptions& options,
                               const std::string& profile,
                               std::size_t dataset_count,
                               std::size_t context_count) {
  if (context_count == 0) {
    return 0;
  }
  uint32_t limit = options.max_target_skeleton_contexts;
  if (options.optimize_for_recall && limit == 0) {
    return static_cast<uint32_t>(context_count);
  }
  if (limit == 0) {
    limit = auto_context_limit(profile, dataset_count);
  }
  if (limit == 0) {
    return static_cast<uint32_t>(context_count);
  }
  return std::min<uint32_t>(limit, static_cast<uint32_t>(context_count));
}

uint32_t auto_model_train_budget(const std::string& profile,
                                 std::size_t dataset_count,
                                 std::size_t context_count) {
  if (profile == "exhaustive") {
    return 0;
  }
  uint32_t context_scale = std::max<uint32_t>(1, static_cast<uint32_t>(context_count));
  if (profile == "fast") {
    if (dataset_count >= 10000000) {
      return std::min<uint32_t>(160, 96 + context_scale * 8);
    }
    if (dataset_count >= 1000000) {
      return std::min<uint32_t>(48, 24 + context_scale * 2);
    }
    if (dataset_count >= 500000) {
      return std::min<uint32_t>(56, 28 + context_scale * 3);
    }
    return std::min<uint32_t>(64, 32 + context_scale * 4);
  }
  if (profile == "balanced") {
    if (dataset_count >= 10000000) {
      return std::min<uint32_t>(320, 160 + context_scale * 16);
    }
    if (dataset_count >= 1000000) {
      return std::min<uint32_t>(96, 56 + context_scale * 4);
    }
    if (dataset_count >= 500000) {
      return std::min<uint32_t>(112, 64 + context_scale * 5);
    }
    return std::min<uint32_t>(128, 72 + context_scale * 6);
  }
  return 0;
}

uint32_t resolve_model_train_budget(const SelectorOptions& options,
                                    const std::string& profile,
                                    std::size_t dataset_count,
                                    std::size_t context_count) {
  if (options.max_model_trains > 0) {
    return options.max_model_trains;
  }
  uint32_t budget = auto_model_train_budget(profile, dataset_count, context_count);
  if (options.optimize_for_recall && budget > 0) {
    budget = std::min<uint32_t>(budget * 2, dataset_count >= 10000000 ? 1024 : 512);
  }
  return budget;
}

uint32_t auto_search_time_budget_seconds(const std::string& profile,
                                         std::size_t dataset_count) {
  if (profile == "exhaustive") {
    return 0;
  }
  if (profile == "fast") {
    if (dataset_count >= 10000000) {
      return 600;
    }
    if (dataset_count >= 1000000) {
      return 180;
    }
    return 120;
  }
  if (profile == "balanced") {
    if (dataset_count >= 10000000) {
      return 1200;
    }
    if (dataset_count >= 1000000) {
      return 600;
    }
    return 360;
  }
  return 0;
}

uint32_t resolve_search_time_budget_seconds(const SelectorOptions& options,
                                            const std::string& profile,
                                            std::size_t dataset_count) {
  if (options.max_search_seconds > 0) {
    return options.max_search_seconds;
  }
  uint32_t budget = auto_search_time_budget_seconds(profile, dataset_count);
  if (options.optimize_for_recall && budget > 0) {
    budget = std::min<uint32_t>(budget * 2, 3600);
  }
  return budget;
}

uint32_t auto_stagnant_context_limit(const SelectorOptions& options,
                                     const std::string& profile,
                                     std::size_t context_count) {
  (void)options;
  if (profile == "exhaustive") {
    return 0;
  }
  if (context_count <= 1) {
    return 0;
  }
  if (profile == "fast") {
    return std::min<uint32_t>(3, static_cast<uint32_t>(context_count));
  }
  if (profile == "balanced") {
    return std::min<uint32_t>(5, static_cast<uint32_t>(context_count));
  }
  return 0;
}

uint32_t resolve_stagnant_context_limit(const SelectorOptions& options,
                                        const std::string& profile,
                                        std::size_t context_count) {
  if (options.max_stagnant_contexts > 0) {
    return options.max_stagnant_contexts;
  }
  if (options.optimize_for_recall) {
    return 0;
  }
  return auto_stagnant_context_limit(options, profile, context_count);
}

uint32_t resolve_post_exploration_candidate_count(const SelectorOptions& options,
                                                  uint32_t max_model_trains) {
  if (!options.post_exploration_exploitation) {
    return 0;
  }
  bool recall_or_calibration_mode =
      options.optimize_for_recall || options.recall_target.has_value() ||
      options.final_calibration_count > 0;
  if (!recall_or_calibration_mode) {
    return 0;
  }
  if (options.post_exploration_candidates > 0) {
    return options.post_exploration_candidates;
  }

  uint32_t base = std::max<uint32_t>(8, std::max(options.recommend_count * 2,
                                                options.phase1_keep * 2));
  if (max_model_trains > 0) {
    base = std::max<uint32_t>(base, std::max<uint32_t>(4, max_model_trains / 5));
    base = std::min<uint32_t>(base, std::max<uint32_t>(1, max_model_trains / 3));
  }
  return std::min<uint32_t>(base, 48);
}

uint32_t resolve_final_calibration_full_count(const SelectorOptions& options,
                                              std::size_t dataset_count) {
  if (options.final_calibration_count == 0) {
    return 0;
  }
  if (options.final_calibration_full_count > 0) {
    return std::min<uint32_t>(options.final_calibration_full_count,
                              options.final_calibration_count);
  }
  if (dataset_count < 500000) {
    return options.final_calibration_count;
  }
  uint32_t promoted =
      std::max<uint32_t>(options.recommend_count,
                         std::max<uint32_t>(1, options.final_calibration_count / 3));
  return std::min<uint32_t>(options.final_calibration_count, promoted);
}

uint32_t resolve_final_calibration_screen_candidate_count(const SelectorOptions& options,
                                                          std::size_t dataset_count) {
  if (!options.final_calibration_screening ||
      options.final_calibration_count == 0) {
    return 0;
  }
  uint32_t full_count = resolve_final_calibration_full_count(options, dataset_count);
  if (full_count == 0 || options.final_calibration_count <= full_count) {
    return options.final_calibration_count;
  }
  if (options.final_calibration_screen_candidate_count > 0) {
    return std::min<uint32_t>(
        options.final_calibration_count,
        std::max<uint32_t>(full_count, options.final_calibration_screen_candidate_count));
  }
  if (dataset_count < 500000) {
    return options.final_calibration_count;
  }

  uint32_t role_floor = full_count;
  if (full_count <= std::numeric_limits<uint32_t>::max() / 2) {
    role_floor = std::max<uint32_t>(role_floor, full_count * 2);
  } else {
    role_floor = options.final_calibration_count;
  }
  if (options.recommend_count <= std::numeric_limits<uint32_t>::max() / 2) {
    role_floor = std::max<uint32_t>(role_floor, options.recommend_count * 2);
  } else {
    role_floor = options.final_calibration_count;
  }
  return std::min<uint32_t>(options.final_calibration_count, role_floor);
}

uint32_t resolve_final_calibration_screen_query_limit(const SelectorOptions& options,
                                                      std::size_t query_count) {
  if (!options.final_calibration_screening ||
      options.final_calibration_count == 0 ||
      query_count == 0) {
    return 0;
  }
  if (options.final_calibration_screen_query_limit > 0) {
    return std::min<uint32_t>(
        options.final_calibration_screen_query_limit,
        static_cast<uint32_t>(
            std::min<std::size_t>(query_count,
                                  std::numeric_limits<uint32_t>::max())));
  }
  uint32_t final_query_limit = options.final_calibration_query_limit > 0
      ? options.final_calibration_query_limit
      : options.eval_query_limit;
  if (final_query_limit == 0) {
    final_query_limit = static_cast<uint32_t>(
        std::min<std::size_t>(query_count, std::numeric_limits<uint32_t>::max()));
  }
  uint32_t auto_limit = 0;
  if (final_query_limit >= 1000) {
    auto_limit = 256;
  } else if (final_query_limit >= 512) {
    auto_limit = 128;
  } else if (final_query_limit >= 256) {
    auto_limit = 64;
  } else {
    auto_limit =
        std::max<uint32_t>(16, std::max<uint32_t>(1, final_query_limit / 4));
  }
  return std::min<uint32_t>(final_query_limit, auto_limit);
}

uint64_t detect_physical_memory_bytes() {
#if defined(__APPLE__)
  uint64_t memsize = 0;
  std::size_t len = sizeof(memsize);
  if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0) {
    return memsize;
  }
#endif
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGESIZE);
  if (pages > 0 && page_size > 0) {
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
  }
#endif
  return 0;
}

uint64_t resolve_selector_memory_budget_bytes(const SelectorOptions& options) {
  if (options.selector_memory_budget_bytes > 0) {
    return options.selector_memory_budget_bytes;
  }
  uint64_t physical_bytes = detect_physical_memory_bytes();
  if (physical_bytes == 0) {
    return 0;
  }
  return (physical_bytes * 7) / 10;
}


uint32_t round_up_pow2_u32(uint64_t value) {
  if (value <= 2) {
    return 2;
  }
  uint64_t out = 1;
  while (out < value && out < (uint64_t{1} << 31)) {
    out <<= 1;
  }
  if (out > std::numeric_limits<uint32_t>::max()) {
    return std::numeric_limits<uint32_t>::max();
  }
  return static_cast<uint32_t>(out);
}

uint32_t max_default_k_for_scale(std::size_t dataset_count,
                                 std::size_t skeleton_count,
                                 bool optimize_for_recall) {
  if (skeleton_count < 2) {
    return 2;
  }
  // FAISS k-means warns and can become very slow when the training set has too
  // few samples per centroid. Keep the default policy in the stable regime;
  // users can still pass explicit --K-values for controlled stress tests.
  std::size_t min_nodes_per_centroid = optimize_for_recall ? 39 : 64;
  uint64_t capacity = std::max<uint64_t>(
      2, static_cast<uint64_t>(skeleton_count / min_nodes_per_centroid));
  uint64_t hard_cap = optimize_for_recall ? 4096 : 2048;
  if (dataset_count >= 500000) {
    hard_cap = optimize_for_recall ? 32768 : 16384;
  }
  if (dataset_count >= 2000000) {
    hard_cap = optimize_for_recall ? 65536 : 32768;
  }
  if (dataset_count >= 10000000) {
    hard_cap = optimize_for_recall ? 131072 : 65536;
  }
  if (dataset_count >= 100000000) {
    hard_cap = optimize_for_recall ? 262144 : 131072;
  }

  if (skeleton_count < 10000) {
    hard_cap = std::min<uint64_t>(hard_cap, optimize_for_recall ? 2048 : 1024);
  } else if (skeleton_count < 30000) {
    hard_cap = std::min<uint64_t>(hard_cap, optimize_for_recall ? 4096 : 2048);
  } else if (skeleton_count < 100000) {
    hard_cap = std::min<uint64_t>(hard_cap, optimize_for_recall ? 8192 : 4096);
  } else if (skeleton_count < 500000) {
    hard_cap = std::min<uint64_t>(hard_cap, optimize_for_recall ? 16384 : 8192);
  }
  uint64_t max_k = std::min<uint64_t>(capacity, hard_cap);
  max_k = std::min<uint64_t>(max_k, static_cast<uint64_t>(skeleton_count));
  return static_cast<uint32_t>(std::max<uint64_t>(2, max_k));
}

std::vector<uint32_t> default_k_scale_ladder(std::size_t dataset_count,
                                             bool optimize_for_recall) {
  if (dataset_count <= 20000) {
    return optimize_for_recall
        ? std::vector<uint32_t>{32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096}
        : std::vector<uint32_t>{32, 64, 128, 256, 512, 1024, 2048};
  }
  if (dataset_count <= 100000) {
    return optimize_for_recall
        ? std::vector<uint32_t>{64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192}
        : std::vector<uint32_t>{64, 128, 256, 512, 1024, 2048, 4096};
  }
  if (dataset_count <= 2000000) {
    return optimize_for_recall
        ? std::vector<uint32_t>{512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768}
        : std::vector<uint32_t>{256, 512, 1024, 2048, 4096, 8192, 16384};
  }
  if (dataset_count <= 10000000) {
    return optimize_for_recall
        ? std::vector<uint32_t>{1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536}
        : std::vector<uint32_t>{512, 1024, 2048, 4096, 8192, 16384, 32768};
  }
  if (dataset_count <= 50000000) {
    return optimize_for_recall
        ? std::vector<uint32_t>{4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072}
        : std::vector<uint32_t>{2048, 4096, 8192, 16384, 32768, 65536};
  }
  return optimize_for_recall
      ? std::vector<uint32_t>{16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144}
      : std::vector<uint32_t>{8192, 16384, 32768, 65536, 131072};
}

std::vector<uint32_t> default_k_values(std::size_t dataset_count,
                                       std::size_t skeleton_count,
                                       bool optimize_for_recall) {
  std::size_t effective_dataset =
      std::max<std::size_t>(dataset_count, skeleton_count);
  if (effective_dataset == 0) {
    effective_dataset = skeleton_count;
  }
  const std::vector<uint32_t> vectors_per_centroid =
      optimize_for_recall ? std::vector<uint32_t>{4096, 2048, 1024, 512, 256, 128, 64}
                          : std::vector<uint32_t>{8192, 4096, 2048, 1024, 512, 256};

  std::vector<uint32_t> seeds = default_k_scale_ladder(effective_dataset,
                                                       optimize_for_recall);
  const uint32_t scale_floor = seeds.empty() ? 2 : seeds.front();
  for (uint32_t target_vpc : vectors_per_centroid) {
    uint64_t raw = (static_cast<uint64_t>(effective_dataset) + target_vpc - 1) /
                   target_vpc;
    uint32_t rounded = round_up_pow2_u32(raw);
    if (rounded >= scale_floor) {
      seeds.push_back(rounded);
    }
  }

  uint32_t max_k = max_default_k_for_scale(effective_dataset,
                                           skeleton_count,
                                           optimize_for_recall);
  if (max_k < scale_floor) {
    seeds.push_back(max_k);
  }
  std::vector<uint32_t> out;
  for (uint32_t value : seeds) {
    if (value >= 2 && value <= max_k &&
        (value >= scale_floor || max_k < scale_floor)) {
      out.push_back(value);
    }
  }
  if (out.empty()) {
    out.push_back(max_k);
  }
  dedup_sort(&out);
  return out;
}

std::vector<uint32_t> default_knn_values() {
  return {8, 16, 32, 64};
}

std::vector<uint32_t> default_knn_values(std::size_t skeleton_count) {
  std::vector<uint32_t> out = default_knn_values();
  if (skeleton_count >= 4096) {
    out.push_back(128);
    out.push_back(256);
  }
  if (skeleton_count >= 16384) {
    out.push_back(512);
  }
  if (skeleton_count >= 65536) {
    out.push_back(1024);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<uint64_t> default_cdf_branching_values(std::size_t dataset_count,
                                                   uint32_t max_k,
                                                   bool optimize_for_recall) {
  std::vector<uint64_t> out = {8, 16, 32, 64};
  if (optimize_for_recall || dataset_count >= 100000 || max_k >= 1024) {
    out.push_back(128);
  }
  if (optimize_for_recall && (dataset_count >= 500000 || max_k >= 4096)) {
    out.push_back(256);
  }
  if (optimize_for_recall && max_k >= 8192) {
    out.push_back(512);
  }
  dedup_sort(&out);
  return out;
}

std::vector<std::string> default_cdf_model_specs() {
  return {"linear,linear", "cubic,linear", "linear,cubic"};
}

std::vector<std::string> default_cdf_model_specs(bool optimize_for_recall) {
  if (!optimize_for_recall) {
    return default_cdf_model_specs();
  }
  return {"cubic,cubic", "cubic,linear", "linear,cubic", "linear,linear"};
}

std::vector<uint32_t> default_two_opt_iterations() {
  return {4, 8};
}

uint32_t scale_aware_k_anchor(const SelectorOptions& options,
                              std::size_t skeleton_count) {
  if (options.K_values.empty()) {
    return skeleton_count >= 2 ? static_cast<uint32_t>(std::min<std::size_t>(
                                     skeleton_count, 256))
                               : 2;
  }
  std::size_t idx = options.K_values.size() / 2;
  if (options.optimize_for_recall) {
    idx = (options.K_values.size() * 2) / 3;
  } else if (options.dataset_count_hint >= 500000) {
    idx = (options.K_values.size() * 3) / 5;
  }
  if (idx >= options.K_values.size()) {
    idx = options.K_values.size() - 1;
  }
  return options.K_values[idx];
}

uint32_t scale_aware_knn_anchor(const SelectorOptions& options) {
  if (options.centroid_knn_values.empty()) {
    return 32;
  }
  std::size_t idx = options.centroid_knn_values.size() / 2;
  if (options.optimize_for_recall || options.dataset_count_hint >= 500000) {
    idx = (options.centroid_knn_values.size() * 2) / 3;
  }
  if (idx >= options.centroid_knn_values.size()) {
    idx = options.centroid_knn_values.size() - 1;
  }
  return options.centroid_knn_values[idx];
}

double skeleton_capacity_scale(std::size_t skeleton_count,
                               std::size_t max_skeleton_count) {
  if (max_skeleton_count == 0 || skeleton_count >= max_skeleton_count) {
    return 1.0;
  }
  double scale =
      static_cast<double>(skeleton_count) / static_cast<double>(max_skeleton_count);
  if (scale < 0.0) {
    return 0.0;
  }
  if (scale > 1.0) {
    return 1.0;
  }
  return scale;
}

std::vector<uint32_t> context_capacity_values(const std::vector<uint32_t>& values,
                                              std::size_t skeleton_count,
                                              double scale,
                                              std::size_t min_keep,
                                              uint32_t anchor) {
  std::vector<uint32_t> filtered;
  filtered.reserve(values.size());
  for (uint32_t value : values) {
    if (value > 0 && value <= skeleton_count) {
      filtered.push_back(value);
    }
  }
  if (filtered.empty()) {
    if (skeleton_count >= 2) {
      filtered.push_back(static_cast<uint32_t>(skeleton_count));
    } else {
      filtered.push_back(2);
    }
  }
  dedup_sort(&filtered);

  std::size_t keep = static_cast<std::size_t>(
      std::llround(static_cast<double>(filtered.size()) * (0.40 + 0.60 * scale)));
  keep = std::max<std::size_t>(min_keep, keep);
  keep = std::min<std::size_t>(keep, filtered.size());

  std::vector<uint32_t> tuned;
  tuned.reserve(keep + 8);
  std::size_t low_keep = scale < 0.5 ? 4 : 2;
  low_keep = std::min<std::size_t>(low_keep, filtered.size());
  for (std::size_t i = 0; i < low_keep; ++i) {
    tuned.push_back(filtered[i]);
  }
  if (filtered.size() >= 4) {
    tuned.push_back(filtered[filtered.size() / 4]);
    tuned.push_back(filtered[filtered.size() / 2]);
    tuned.push_back(filtered[(filtered.size() * 3) / 4]);
  }
  tuned.insert(tuned.end(), filtered.end() - keep, filtered.end());
  tuned.push_back(filtered.front());
  tuned.push_back(nearest_value(filtered, anchor));
  dedup_sort(&tuned);
  return tuned;
}

SelectorOptions apply_context_capacity_scaling(const SelectorOptions& options,
                                               std::size_t skeleton_count,
                                               std::size_t max_skeleton_count) {
  if (!options.skeleton_capacity_scaling) {
    return options;
  }
  SelectorOptions tuned = options;
  double scale = skeleton_capacity_scale(skeleton_count, max_skeleton_count);
  tuned.K_values =
      context_capacity_values(options.K_values,
                              skeleton_count,
                              scale,
                              2,
                              scale_aware_k_anchor(options, skeleton_count));
  tuned.centroid_knn_values =
      context_capacity_values(options.centroid_knn_values,
                              skeleton_count,
                              scale,
                              2,
                              scale_aware_knn_anchor(options));
  return tuned;
}

bool meets_recall_target(const SelectorCandidateMetrics& metrics, double recall_target) {
  if (metrics.recall_target_met_by_curve ||
      metrics.min_window_for_recall_target > 0) {
    return true;
  }
  return metrics.recall_at_k_in_window + kNormEpsilon >= recall_target;
}

void validate_model_specs(const std::vector<std::string>& model_specs) {
  for (const auto& spec : model_specs) {
    auto parts = split_model_spec(spec);
    if (parts.size() != 2) {
      throw std::runtime_error("Each cdf model spec must have exactly two layers: " + spec);
    }
    (void)parse_cdf_model(parts[0]);
    (void)parse_cdf_model(parts[1]);
  }
}

SelectorOptions normalize_selector_options(const SelectorOptions& raw,
                                           std::size_t skeleton_count) {
  SelectorOptions options = raw;
  options.selector_profile = normalize_selector_profile(options.selector_profile);
  options.target_skeleton_values = normalize_target_skeleton_values(options.target_skeleton_values);
  options.target_skeleton_percentages =
      normalize_target_skeleton_percentages(options.target_skeleton_percentages);
  bool has_target_skeleton_search =
      !options.target_skeleton_values.empty() || !options.target_skeleton_percentages.empty();
  if (has_target_skeleton_search && !options.nsw_index_path.has_value()) {
    throw std::runtime_error("target_skeleton_values/percentages require nsw_index_path");
  }
  if (options.nsw_layer.has_value() && *options.nsw_layer < 0) {
    throw std::runtime_error("nsw_layer must be >= 0 when set");
  }
  if (options.nsw_path.empty() && !has_target_skeleton_search &&
      !uses_auto_target_skeleton_policy(options)) {
    throw std::runtime_error(
        "nsw_path must be set, or nsw_index_path must be set for auto target_skeleton search");
  }
  if (options.hash_bits == 0 || options.hash_bits > 256) {
    throw std::runtime_error("hash_bits must be in [1, 256]");
  }
  if (options.threads == 0) {
    throw std::runtime_error("threads must be > 0");
  }
  if (options.eval_k == 0) {
    throw std::runtime_error("eval_k must be > 0");
  }
  options.eval_window_values.erase(
      std::remove(options.eval_window_values.begin(),
                  options.eval_window_values.end(),
                  uint64_t{0}),
      options.eval_window_values.end());
  dedup_sort(&options.eval_window_values);
  options.eval_node_count_values.erase(
      std::remove_if(options.eval_node_count_values.begin(),
                     options.eval_node_count_values.end(),
                     [](uint32_t value) { return value < 2; }),
      options.eval_node_count_values.end());
  dedup_sort(&options.eval_node_count_values);
  if (options.phase1_keep == 0) {
    throw std::runtime_error("phase1_keep must be > 0");
  }
  if (options.max_phase2_candidates == 0) {
    throw std::runtime_error("max_phase2_candidates must be > 0");
  }
  if (options.max_phase2_candidates < options.phase1_keep) {
    options.max_phase2_candidates = options.phase1_keep;
  }
  if (options.recommend_count == 0) {
    throw std::runtime_error("recommend_count must be > 0");
  }
  if (options.final_calibration_full_count > 0 &&
      options.final_calibration_count == 0) {
    throw std::runtime_error(
        "final_calibration_full_count requires final_calibration_count > 0");
  }
  if (options.final_calibration_full_count > options.final_calibration_count &&
      options.final_calibration_count > 0) {
    options.final_calibration_full_count = options.final_calibration_count;
  }
  if (options.final_calibration_screen_query_limit > 0 &&
      options.final_calibration_count == 0) {
    throw std::runtime_error(
        "final_calibration_screen_query_limit requires final_calibration_count > 0");
  }
  if (options.final_calibration_screen_candidate_count > 0 &&
      options.final_calibration_count == 0) {
    throw std::runtime_error(
        "final_calibration_screen_candidate_count requires final_calibration_count > 0");
  }
  if (options.coarse_latency_iterations == 0 || options.latency_iterations == 0) {
    throw std::runtime_error("latency iterations must be > 0");
  }
  if (options.recall_target.has_value()) {
    if (!std::isfinite(*options.recall_target) ||
        *options.recall_target < 0.0 ||
        *options.recall_target > 1.0) {
      throw std::runtime_error("recall_target must be in [0, 1]");
    }
  }
  if (!std::isfinite(options.min_recall_improvement) ||
      options.min_recall_improvement < 0.0) {
    throw std::runtime_error("min_recall_improvement must be finite and >= 0");
  }
  if (!std::isfinite(options.min_quality_improvement) ||
      options.min_quality_improvement < 0.0) {
    throw std::runtime_error("min_quality_improvement must be finite and >= 0");
  }
  if (options.max_recall_refine_rounds == 0) {
    throw std::runtime_error("max_recall_refine_rounds must be > 0");
  }
  if (options.beam_width == 0) {
    throw std::runtime_error("beam_width must be > 0");
  }
  if (options.beam_rounds == 0) {
    throw std::runtime_error("beam_rounds must be > 0");
  }
  if (options.hill_climb_steps == 0) {
    throw std::runtime_error("hill_climb_steps must be > 0");
  }
  if (options.beam_neighbor_limit == 0) {
    throw std::runtime_error("beam_neighbor_limit must be > 0");
  }

  const std::size_t dataset_count =
      options.dataset_count_hint > 0
          ? static_cast<std::size_t>(options.dataset_count_hint)
          : skeleton_count;
  if (!options.use_full_dataset_for_assignments) {
    options.assignment_sample_limit = 0;
  }
  if (options.auto_assignment_sample_limit && options.assignment_sample_limit == 0) {
    options.assignment_sample_limit =
        default_assignment_sample_limit(options, dataset_count);
  }

  if (options.K_values.empty()) {
    options.K_values =
        default_k_values(dataset_count, skeleton_count, options.optimize_for_recall);
  }
  options.K_values.erase(std::remove_if(options.K_values.begin(),
                                        options.K_values.end(),
                                        [&](uint32_t value) {
                                          return value < 2 || value > skeleton_count;
                                        }),
                         options.K_values.end());
  if (options.K_values.empty()) {
    throw std::runtime_error("No valid K values remain after filtering by skeleton size");
  }
  dedup_sort(&options.K_values);

  if (options.centroid_knn_values.empty()) {
    options.centroid_knn_values = default_knn_values(skeleton_count);
  }
  options.centroid_knn_values.erase(
      std::remove_if(options.centroid_knn_values.begin(),
                     options.centroid_knn_values.end(),
                     [](uint32_t value) { return value == 0; }),
      options.centroid_knn_values.end());
  if (options.centroid_knn_values.empty()) {
    options.centroid_knn_values.push_back(1);
  }
  dedup_sort(&options.centroid_knn_values);

  if (options.cdf_branching_values.empty()) {
    options.cdf_branching_values = default_cdf_branching_values(
        dataset_count, options.K_values.back(), options.optimize_for_recall);
  }
  options.cdf_branching_values.erase(
      std::remove_if(options.cdf_branching_values.begin(),
                     options.cdf_branching_values.end(),
                     [](uint64_t value) { return value < 2; }),
      options.cdf_branching_values.end());
  if (options.cdf_branching_values.empty()) {
    options.cdf_branching_values.push_back(2);
  }
  dedup_sort(&options.cdf_branching_values);

  if (options.cdf_model_specs.empty()) {
    options.cdf_model_specs = default_cdf_model_specs(options.optimize_for_recall);
  }
  for (auto& spec : options.cdf_model_specs) {
    spec = trim_copy(spec);
  }
  options.cdf_model_specs.erase(
      std::remove_if(options.cdf_model_specs.begin(),
                     options.cdf_model_specs.end(),
                     [](const std::string& spec) { return spec.empty(); }),
      options.cdf_model_specs.end());
  dedup_strings(&options.cdf_model_specs);
  if (options.cdf_model_specs.empty()) {
    throw std::runtime_error("cdf_model_specs must not be empty");
  }
  validate_model_specs(options.cdf_model_specs);

  if (options.two_opt_iterations_values.empty()) {
    options.two_opt_iterations_values = default_two_opt_iterations();
  }
  options.two_opt_iterations_values.erase(
      std::remove_if(options.two_opt_iterations_values.begin(),
                     options.two_opt_iterations_values.end(),
                     [](uint32_t value) { return value == 0; }),
      options.two_opt_iterations_values.end());
  if (options.two_opt_iterations_values.empty()) {
    options.two_opt_iterations_values.push_back(1);
  }
  dedup_sort(&options.two_opt_iterations_values);

  const SelectorWeights& weights = options.weights;
  std::array<double, 5> all_weights = {
      weights.recall, weights.rank_distance, weights.latency, weights.model_size, weights.train_time};
  double weight_sum = 0.0;
  for (double w : all_weights) {
    if (w < 0.0) {
      throw std::runtime_error("selector weights must be non-negative");
    }
    weight_sum += w;
  }
  if (weight_sum <= 0.0) {
    throw std::runtime_error("at least one selector weight must be positive");
  }

  options.beam_width =
      std::min<uint32_t>(options.beam_width, options.max_phase2_candidates);
  if (options.beam_width == 0) {
    options.beam_width = 1;
  }

  return options;
}

SelectorCandidateConfig clamp_candidate_config(const SelectorCandidateConfig& raw,
                                               std::size_t skeleton_count) {
  SelectorCandidateConfig config = raw;
  if (config.K < 2) {
    config.K = 2;
  }
  if (config.K > skeleton_count) {
    config.K = static_cast<uint32_t>(skeleton_count);
  }
  if (config.centroid_knn == 0) {
    config.centroid_knn = 1;
  }
  if (config.K > 0) {
    uint32_t max_knn = config.K > 1 ? (config.K - 1) : 1;
    if (config.centroid_knn > max_knn) {
      config.centroid_knn = max_knn;
    }
  }
  if (config.cdf_branching_factor < 2) {
    config.cdf_branching_factor = 2;
  }
  if (config.two_opt_iterations == 0) {
    config.two_opt_iterations = 1;
  }
  if (!config.enable_2opt) {
    config.two_opt_iterations = 1;
  }
  return config;
}

std::vector<SelectorCandidateConfig> dedup_candidates(
    const std::vector<SelectorCandidateConfig>& candidates,
    std::size_t skeleton_count) {
  std::unordered_set<std::string> seen;
  std::vector<SelectorCandidateConfig> out;
  out.reserve(candidates.size());
  for (const auto& raw : candidates) {
    auto config = clamp_candidate_config(raw, skeleton_count);
    std::string key = selector_config_key(config);
    if (seen.insert(key).second) {
      out.push_back(config);
    }
  }
  return out;
}

uint32_t target_knn_for_k(uint32_t k, bool optimize_for_recall) {
  if (k <= 2) {
    return 1;
  }
  uint32_t divisor = optimize_for_recall ? 16 : 32;
  uint32_t target = std::max<uint32_t>(8, k / divisor);
  uint32_t max_knn = k > 1 ? k - 1 : 1;
  uint32_t cap = optimize_for_recall ? 1024 : 512;
  target = std::min<uint32_t>(target, max_knn);
  return std::min<uint32_t>(target, cap);
}

uint64_t target_branch_for_k(uint32_t k, bool optimize_for_recall) {
  if (k >= 8192) {
    return optimize_for_recall ? 512 : 256;
  }
  if (k >= 4096) {
    return optimize_for_recall ? 256 : 128;
  }
  if (k >= 1024) {
    return 128;
  }
  return 32;
}

std::string preferred_model_spec(const std::vector<std::string>& specs,
                                 bool optimize_for_recall) {
  const std::vector<std::string> preferred =
      optimize_for_recall ? std::vector<std::string>{"cubic,cubic",
                                                     "cubic,linear",
                                                     "linear,cubic",
                                                     "linear,linear"}
                          : std::vector<std::string>{"linear,linear",
                                                     "cubic,linear",
                                                     "linear,cubic",
                                                     "cubic,cubic"};
  for (const auto& candidate : preferred) {
    if (std::find(specs.begin(), specs.end(), candidate) != specs.end()) {
      return candidate;
    }
  }
  return specs.front();
}

std::vector<SelectorCandidateConfig> build_phase1_candidates_internal(
    const SelectorOptions& options,
    std::size_t skeleton_count) {
  uint32_t base_k = nearest_value(
      options.K_values, scale_aware_k_anchor(options, skeleton_count));
  uint32_t base_knn = nearest_value(options.centroid_knn_values,
                                    target_knn_for_k(base_k, options.optimize_for_recall));
  uint64_t base_branch = nearest_value(options.cdf_branching_values,
                                       target_branch_for_k(base_k, options.optimize_for_recall));
  std::string base_spec = preferred_model_spec(options.cdf_model_specs,
                                               options.optimize_for_recall);
  uint32_t base_two_opt = nearest_value(options.two_opt_iterations_values, 8);

  SelectorCandidateConfig base;
  base.K = base_k;
  base.centroid_knn = base_knn;
  base.cdf_model_spec = base_spec;
  base.cdf_branching_factor = base_branch;
  base.enable_2opt = true;
  base.two_opt_iterations = base_two_opt;
  base.enable_graph_centroid_order = false;

  std::vector<SelectorCandidateConfig> candidates;
  auto push_phase1_candidate = [&](SelectorCandidateConfig cand) {
    cand.enable_graph_centroid_order = false;
    candidates.push_back(cand);
  };
  push_phase1_candidate(base);

  for (uint32_t K : options.K_values) {
    auto cand = base;
    cand.K = K;
    cand.centroid_knn = nearest_value(options.centroid_knn_values,
                                      target_knn_for_k(K, options.optimize_for_recall));
    cand.cdf_branching_factor = nearest_value(
        options.cdf_branching_values, target_branch_for_k(K, options.optimize_for_recall));
    push_phase1_candidate(cand);
  }
  for (uint32_t knn : options.centroid_knn_values) {
    auto cand = base;
    cand.centroid_knn = knn;
    push_phase1_candidate(cand);
  }
  for (uint64_t branch : options.cdf_branching_values) {
    auto cand = base;
    cand.cdf_branching_factor = branch;
    push_phase1_candidate(cand);
  }
  for (const auto& spec : options.cdf_model_specs) {
    auto cand = base;
    cand.cdf_model_spec = spec;
    push_phase1_candidate(cand);
  }
  for (uint32_t K : options.K_values) {
    for (const auto& spec : options.cdf_model_specs) {
      if (spec == base.cdf_model_spec) {
        continue;
      }
      auto cand = base;
      cand.K = K;
      cand.centroid_knn = nearest_value(options.centroid_knn_values,
                                        target_knn_for_k(K, options.optimize_for_recall));
      cand.cdf_branching_factor = nearest_value(
          options.cdf_branching_values, target_branch_for_k(K, options.optimize_for_recall));
      cand.cdf_model_spec = spec;
      push_phase1_candidate(cand);
    }
  }
  for (uint32_t iters : options.two_opt_iterations_values) {
    auto cand = base;
    cand.two_opt_iterations = iters;
    cand.enable_2opt = true;
    push_phase1_candidate(cand);
  }
  if (options.include_disable_2opt) {
    auto cand = base;
    cand.enable_2opt = false;
    cand.two_opt_iterations = 1;
    push_phase1_candidate(cand);
  }

  return dedup_candidates(candidates, skeleton_count);
}

SelectorCandidateConfig build_context_probe_candidate(const SelectorOptions& options,
                                                      std::size_t skeleton_count,
                                                      uint64_t target_skeleton) {
  SelectorCandidateConfig probe;
  uint32_t probe_k_anchor = scale_aware_k_anchor(options, skeleton_count);
  if (!options.K_values.empty()) {
    std::size_t low_idx = options.K_values.size() / 3;
    if (low_idx >= options.K_values.size()) {
      low_idx = options.K_values.size() - 1;
    }
    probe_k_anchor = options.K_values[low_idx];
  }
  probe.K = nearest_value(options.K_values, std::max<uint32_t>(2, probe_k_anchor));
  uint32_t knn_anchor = target_knn_for_k(probe.K, options.optimize_for_recall);
  probe.centroid_knn = nearest_value(options.centroid_knn_values, knn_anchor);
  if (probe.centroid_knn >= probe.K) {
    probe.centroid_knn = probe.K > 1 ? (probe.K - 1) : 1;
  }
  probe.cdf_model_spec = preferred_model_spec(options.cdf_model_specs,
                                              options.optimize_for_recall);
  probe.cdf_branching_factor = nearest_value(
      options.cdf_branching_values, target_branch_for_k(probe.K, options.optimize_for_recall));
  probe.enable_2opt = false;
  probe.two_opt_iterations = 1;
  probe.enable_graph_centroid_order = false;
  probe.target_skeleton = target_skeleton;
  return clamp_candidate_config(probe, skeleton_count);
}


} // namespace vortex::selector_internal
