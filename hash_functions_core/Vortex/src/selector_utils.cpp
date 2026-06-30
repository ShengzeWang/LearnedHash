#include "selector_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vortex::selector_internal {

std::string selector_config_key(const SelectorCandidateConfig& config) {
  std::string key;
  key.reserve(128);
  key += std::to_string(config.target_skeleton);
  key += "|";
  key += std::to_string(config.K);
  key += "|";
  key += std::to_string(config.centroid_knn);
  key += "|";
  key += config.cdf_model_spec;
  key += "|";
  key += std::to_string(config.cdf_branching_factor);
  key += "|";
  key += config.enable_2opt ? "1" : "0";
  key += "|";
  key += std::to_string(config.two_opt_iterations);
  key += "|";
  key += config.enable_graph_centroid_order ? "1" : "0";
  return key;
}

std::string selector_training_context_key(const NswSelectorContext& context) {
  std::string context_key = context.skeleton_identity;
  if (context_key.empty()) {
    context_key = context.nsw_path.string();
  }
  if (context_key.empty()) {
    context_key = context.display_nsw_path;
  }
  return context_key;
}

std::string selector_training_cache_key(const NswSelectorContext& context,
                                        const SelectorCandidateConfig& config,
                                        const SelectorOptions& options) {
  std::string context_key = selector_training_context_key(context);
  std::string key;
  key.reserve(context_key.size() + 2 + 192);
  key += context_key;
  key += "||";
  key += selector_config_key(config);
  key += "|seed|";
  key += std::to_string(options.seed);
  key += "|assign|";
  key += options.use_full_dataset_for_assignments ? "dataset" : "skeleton";
  key += "|sample|";
  key += std::to_string(options.assignment_sample_limit);
  return key;
}

bool selector_training_base_includes_centroid_graph(
    const SelectorOptions& options,
    const SelectorCandidateConfig& config) {
  return config.enable_graph_centroid_order ||
         (options.enable_graph_centroid_order &&
          options.include_disable_graph_centroid_order);
}

std::string selector_training_base_cache_key(const NswSelectorContext& context,
                                             const SelectorCandidateConfig& config,
                                             uint64_t seed,
                                             bool use_full_dataset_for_assignments,
                                             uint64_t assignment_sample_limit,
                                             bool include_centroid_graph) {
  std::string context_key = selector_training_context_key(context);
  std::string key;
  key.reserve(context_key.size() + 128);
  key += context_key;
  key += "||base|";
  key += std::to_string(config.K);
  key += "|";
  key += std::to_string(seed);
  key += "|assign|";
  key += use_full_dataset_for_assignments ? "dataset" : "skeleton";
  key += "|sample|";
  key += std::to_string(assignment_sample_limit);
  key += "|centroid_graph|";
  key += include_centroid_graph ? "1" : "0";
  return key;
}

std::string selector_training_order_cache_key(const std::string& base_key,
                                              const SelectorCandidateConfig& config) {
  std::string key;
  key.reserve(base_key.size() + 80);
  key += base_key;
  key += "|graph_order|";
  key += config.enable_graph_centroid_order ? "1" : "0";
  key += "|order|";
  key += std::to_string(config.centroid_knn);
  key += "|";
  key += config.enable_2opt ? "1" : "0";
  key += "|";
  key += std::to_string(config.two_opt_iterations);
  return key;
}

std::string selector_training_cdf_cache_key(const std::string& base_key,
                                            const SelectorCandidateConfig& config) {
  std::string key;
  key.reserve(base_key.size() + config.cdf_model_spec.size() + 64);
  key += base_key;
  key += "|cdf|";
  key += config.cdf_model_spec;
  key += "|";
  key += std::to_string(config.cdf_branching_factor);
  return key;
}

std::vector<uint64_t> normalize_target_skeleton_values(std::vector<uint64_t> values) {
  values.erase(std::remove(values.begin(), values.end(), 0), values.end());
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

std::vector<double> normalize_target_skeleton_percentages(std::vector<double> values) {
  for (double value : values) {
    if (!std::isfinite(value) || value <= 0.0 || value > 100.0) {
      throw std::runtime_error("target_skeleton_percentages must be in (0, 100]");
    }
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(),
                           values.end(),
                           [](double lhs, double rhs) {
                             return std::abs(lhs - rhs) <= 1e-9;
                           }),
               values.end());
  return values;
}

bool uses_auto_target_skeleton_policy(const SelectorOptions& options) {
  return options.nsw_path.empty() && options.nsw_index_path.has_value() &&
         options.target_skeleton_values.empty() &&
         options.target_skeleton_percentages.empty();
}

std::vector<uint64_t> default_target_skeleton_values(const SelectorOptions& options,
                                                     std::size_t dataset_count) {
  if (dataset_count == 0) {
    throw std::runtime_error("Dataset is empty");
  }
  const uint64_t count = static_cast<uint64_t>(dataset_count);
  std::vector<uint64_t> values;
  values.reserve(8);
  auto add = [&](uint64_t value) {
    if (value == 0) {
      return;
    }
    values.push_back(std::min<uint64_t>(value, count));
  };
  auto add_fraction = [&](uint64_t numerator, uint64_t denominator) {
    if (denominator == 0) {
      return;
    }
    uint64_t value = (count * numerator + denominator - 1) / denominator;
    add(std::max<uint64_t>(1, value));
  };

  if (count <= 20000) {
    add_fraction(1, 5);
    add_fraction(1, 2);
    add(count);
  } else if (count <= 100000) {
    add(10000);
    add(30000);
    add_fraction(3, 5);
    add(count);
  } else if (count <= 500000) {
    add(10000);
    add(30000);
    add(100000);
    add(300000);
    add(count);
  } else if (count <= 2000000) {
    add(30000);
    add(100000);
    add(300000);
    add(1000000);
    add(count);
  } else if (count <= 10000000) {
    add(100000);
    add(300000);
    add(1000000);
    add(3000000);
    add(options.optimize_for_recall ? 5000000 : 3000000);
    if (options.optimize_for_recall && count <= 5000000) {
      add(count);
    }
  } else if (count <= 50000000) {
    add(300000);
    add(1000000);
    add(3000000);
    add(options.optimize_for_recall ? 10000000 : 5000000);
  } else {
    add(1000000);
    add(3000000);
    add(10000000);
    if (options.optimize_for_recall) {
      add(30000000);
    }
  }

  return normalize_target_skeleton_values(std::move(values));
}

uint64_t default_assignment_sample_limit(const SelectorOptions& options,
                                         std::size_t dataset_count) {
  if (!options.use_full_dataset_for_assignments || dataset_count <= 2000000) {
    return 0;
  }
  if (dataset_count <= 10000000) {
    return options.optimize_for_recall ? 3000000 : 2000000;
  }
  if (dataset_count <= 50000000) {
    return options.optimize_for_recall ? 5000000 : 3000000;
  }
  return options.optimize_for_recall ? 10000000 : 5000000;
}

std::vector<uint64_t> resolve_target_skeleton_values(const SelectorOptions& options,
                                                     std::size_t dataset_count) {
  if (dataset_count == 0) {
    throw std::runtime_error("Dataset is empty");
  }
  if (uses_auto_target_skeleton_policy(options)) {
    return default_target_skeleton_values(options, dataset_count);
  }
  const uint64_t max_nodes = static_cast<uint64_t>(dataset_count);
  std::vector<uint64_t> resolved;
  resolved.reserve(options.target_skeleton_values.size() + options.target_skeleton_percentages.size());

  for (uint64_t value : normalize_target_skeleton_values(options.target_skeleton_values)) {
    resolved.push_back(std::min<uint64_t>(value, max_nodes));
  }

  for (double percent : normalize_target_skeleton_percentages(options.target_skeleton_percentages)) {
    double scaled = (static_cast<double>(dataset_count) * percent) / 100.0;
    uint64_t target = static_cast<uint64_t>(std::llround(scaled));
    if (target == 0) {
      target = 1;
    }
    resolved.push_back(std::min<uint64_t>(target, max_nodes));
  }

  return normalize_target_skeleton_values(std::move(resolved));
}

uint64_t estimate_model_memory_bytes(const VortexModel& model) {
  uint64_t total = 0;
  total += static_cast<uint64_t>(model.centroids.size()) * sizeof(float);
  total += static_cast<uint64_t>(model.order.size()) * sizeof(uint32_t);
  total += static_cast<uint64_t>(model.mass.size()) * sizeof(uint64_t);
  total += static_cast<uint64_t>(model.range_start.size()) *
           sizeof(rm_model::CentroidRouter::Range256);
  total += static_cast<uint64_t>(model.range_size.size()) *
           sizeof(rm_model::CentroidRouter::Range256);
  total += static_cast<uint64_t>(model.range_full.size()) * sizeof(uint8_t);
  total += static_cast<uint64_t>(model.min_dist.size()) * sizeof(double);
  total += static_cast<uint64_t>(model.max_dist.size()) * sizeof(double);
  total += static_cast<uint64_t>(model.cdf_rows.size()) * sizeof(uint64_t);
  total += static_cast<uint64_t>(model.cdf_top_params.size()) * sizeof(double);
  total += static_cast<uint64_t>(model.cdf_leaf_params.size()) * sizeof(double);
  total += static_cast<uint64_t>(model.router.active_centroids().size()) * sizeof(uint32_t);
  return total;
}

std::filesystem::path unique_temp_path(const std::string& stem, const std::string& ext) {
  auto temp_dir = std::filesystem::temp_directory_path();
  std::random_device rd;
  for (int attempt = 0; attempt < 64; ++attempt) {
    uint64_t suffix = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    std::filesystem::path candidate = temp_dir / (stem + "_" + std::to_string(suffix) + ext);
    if (!std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  return temp_dir / (stem + "_fallback" + ext);
}

uint64_t serialized_model_size_bytes(const VortexModel& model) {
  std::filesystem::path path = unique_temp_path("vortex_selector_model", ".bin");
  uint64_t size = 0;
  try {
    model.write(path);
    size = static_cast<uint64_t>(std::filesystem::file_size(path));
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw;
  }
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
  return size;
}

std::vector<std::size_t> all_indices(std::size_t count) {
  std::vector<std::size_t> out(count);
  std::iota(out.begin(), out.end(), 0);
  return out;
}

std::vector<std::size_t> sample_indices(std::size_t count,
                                        std::size_t limit,
                                        uint64_t seed) {
  if (count == 0 || limit == 0 || limit >= count) {
    return all_indices(count);
  }

  std::mt19937_64 rng(seed);
  std::unordered_set<std::size_t> chosen;
  chosen.reserve(limit * 2);

  for (std::size_t i = count - limit; i < count; ++i) {
    std::uniform_int_distribution<std::size_t> dist(0, i);
    std::size_t pick = dist(rng);
    if (!chosen.insert(pick).second) {
      chosen.insert(i);
    }
  }

  std::vector<std::size_t> out(chosen.begin(), chosen.end());
  std::sort(out.begin(), out.end());
  return out;
}

vector_io::VectorStorage<float> subset_vectors(const vector_io::VectorStorage<float>& src,
                                               const std::vector<std::size_t>& indices) {
  vector_io::VectorStorage<float> out(src.dim, indices.size());
  for (std::size_t row = 0; row < indices.size(); ++row) {
    std::size_t src_idx = indices[row];
    if (src_idx >= src.count) {
      throw std::runtime_error("Subset index out of range");
    }
    const float* src_ptr = src.values.data() + src_idx * src.dim;
    float* dst_ptr = out.values.data() + row * out.dim;
    std::copy(src_ptr, src_ptr + src.dim, dst_ptr);
  }
  return out;
}

uint32_t nearest_value(const std::vector<uint32_t>& values, uint32_t target) {
  if (values.empty()) {
    throw std::runtime_error("nearest_value requires non-empty values");
  }
  uint32_t best = values.front();
  uint32_t best_diff = best > target ? (best - target) : (target - best);
  for (uint32_t value : values) {
    uint32_t diff = value > target ? (value - target) : (target - value);
    if (diff < best_diff) {
      best = value;
      best_diff = diff;
    }
  }
  return best;
}

uint64_t nearest_value(const std::vector<uint64_t>& values, uint64_t target) {
  if (values.empty()) {
    throw std::runtime_error("nearest_value requires non-empty values");
  }
  uint64_t best = values.front();
  uint64_t best_diff = best > target ? (best - target) : (target - best);
  for (uint64_t value : values) {
    uint64_t diff = value > target ? (value - target) : (target - value);
    if (diff < best_diff) {
      best = value;
      best_diff = diff;
    }
  }
  return best;
}

void dedup_sort(std::vector<uint32_t>* values) {
  std::sort(values->begin(), values->end());
  values->erase(std::unique(values->begin(), values->end()), values->end());
}

void dedup_sort(std::vector<uint64_t>* values) {
  std::sort(values->begin(), values->end());
  values->erase(std::unique(values->begin(), values->end()), values->end());
}

void dedup_strings(std::vector<std::string>* values) {
  std::sort(values->begin(), values->end());
  values->erase(std::unique(values->begin(), values->end()), values->end());
}

} // namespace vortex::selector_internal
