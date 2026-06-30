#include "selector_eval_internal.h"

#include <faiss/IndexFlat.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace vortex::selector_internal {

std::vector<std::size_t> prefix_indices(std::size_t count, std::size_t limit) {
  std::size_t take = limit > 0 ? std::min<std::size_t>(limit, count) : count;
  std::vector<std::size_t> out(take);
  std::iota(out.begin(), out.end(), 0);
  return out;
}

uint64_t clamp_eval_window(uint64_t value, std::size_t base_count) {
  if (base_count <= 1) {
    return 0;
  }
  uint64_t max_window = static_cast<uint64_t>(base_count - 1);
  if (value == 0) {
    return std::min<uint64_t>(1000, max_window);
  }
  return std::min<uint64_t>(value, max_window);
}

uint64_t eval_base_identity(const vector_io::VectorStorage<float>& full_base,
                            const std::vector<std::size_t>& base_indices) {
  constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
  uint64_t fingerprint = kFnvOffset;
  auto base_address = reinterpret_cast<std::uintptr_t>(full_base.values.data());
  fingerprint = fnv1a_update_value(fingerprint, base_address);
  fingerprint = fnv1a_update_value(fingerprint, full_base.count);
  fingerprint = fnv1a_update_value(fingerprint, full_base.dim);
  std::size_t index_count = base_indices.size();
  fingerprint = fnv1a_update_value(fingerprint, index_count);
  if (!base_indices.empty()) {
    fingerprint = fnv1a_update(fingerprint,
                               base_indices.data(),
                               base_indices.size() * sizeof(std::size_t));
  }
  return fingerprint == 0 ? 1 : fingerprint;
}

std::vector<uint64_t> default_eval_windows(uint64_t window, std::size_t base_count) {
  uint64_t base = clamp_eval_window(window, base_count);
  if (base == 0) {
    return {0};
  }
  std::vector<uint64_t> out = {
      std::max<uint64_t>(1, base / 4),
      std::max<uint64_t>(1, base / 2),
      base,
      base > std::numeric_limits<uint64_t>::max() / 2
          ? std::numeric_limits<uint64_t>::max()
          : base * 2,
      base > std::numeric_limits<uint64_t>::max() / 5
          ? std::numeric_limits<uint64_t>::max()
          : base * 5,
  };
  for (uint64_t& value : out) {
    value = clamp_eval_window(value, base_count);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<uint64_t> normalize_eval_windows(const std::vector<uint64_t>& requested,
                                             uint64_t window,
                                             std::size_t base_count) {
  std::vector<uint64_t> out = requested.empty()
      ? default_eval_windows(window, base_count)
      : requested;
  out.push_back(window);
  for (uint64_t& value : out) {
    value = clamp_eval_window(value, base_count);
  }
  out.erase(std::remove(out.begin(), out.end(), uint64_t{0}), out.end());
  if (out.empty() && base_count > 1) {
    out.push_back(clamp_eval_window(window, base_count));
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::vector<uint32_t> default_eval_node_counts(std::size_t base_count) {
  const std::vector<uint32_t> defaults = {32, 64, 128, 256, 512, 1024};
  std::vector<uint32_t> out;
  out.reserve(defaults.size());
  for (uint32_t value : defaults) {
    if (value >= 2 && value <= base_count) {
      out.push_back(value);
    }
  }
  if (out.empty() && base_count >= 2) {
    out.push_back(static_cast<uint32_t>(
        std::min<std::size_t>(base_count,
                              static_cast<std::size_t>(
                                  std::numeric_limits<uint32_t>::max()))));
  }
  return out;
}

std::vector<uint32_t> normalize_eval_node_counts(const std::vector<uint32_t>& requested,
                                                 std::size_t base_count) {
  std::vector<uint32_t> out =
      requested.empty() ? default_eval_node_counts(base_count) : requested;
  out.erase(std::remove_if(out.begin(),
                           out.end(),
                           [&](uint32_t value) {
                             return value < 2 || value > base_count;
                           }),
            out.end());
  if (out.empty() && base_count >= 2) {
    out.push_back(static_cast<uint32_t>(
        std::min<std::size_t>(base_count,
                              static_cast<std::size_t>(
                                  std::numeric_limits<uint32_t>::max()))));
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

EvalDataset build_eval_dataset(const vector_io::VectorStorage<float>& full_base,
                               const std::optional<vector_io::VectorStorage<float>>& full_query,
                               uint32_t eval_k,
                               uint64_t requested_window,
                               const std::vector<uint64_t>& requested_windows,
                               const std::vector<uint32_t>& requested_node_counts,
                               uint32_t base_limit,
                               uint32_t query_limit,
                               uint64_t seed,
                               bool prefix_query) {
  EvalDataset eval;
  std::size_t base_take = base_limit > 0 ? static_cast<std::size_t>(base_limit) : full_base.count;
  auto base_idx = sample_indices(full_base.count, base_take, seed + 11);
  eval.base_identity = eval_base_identity(full_base, base_idx);
  eval.base = subset_vectors(full_base, base_idx);

  if (!full_query.has_value()) {
    std::size_t query_take = query_limit > 0 ? static_cast<std::size_t>(query_limit)
                                             : eval.base.count;
    auto query_pos = prefix_query
        ? prefix_indices(eval.base.count, query_take)
        : sample_indices(eval.base.count, query_take, seed + 29);
    eval.query = subset_vectors(eval.base, query_pos);
    eval.exclude_self = true;
    eval.self_base_indices.resize(query_pos.size(), -1);
    for (std::size_t i = 0; i < query_pos.size(); ++i) {
      eval.self_base_indices[i] = static_cast<faiss::idx_t>(query_pos[i]);
    }
  } else {
    std::size_t query_take = query_limit > 0 ? static_cast<std::size_t>(query_limit)
                                             : full_query->count;
    auto query_idx = prefix_query
        ? prefix_indices(full_query->count, query_take)
        : sample_indices(full_query->count, query_take, seed + 37);
    eval.query = subset_vectors(*full_query, query_idx);
    eval.exclude_self = false;
    eval.self_base_indices.assign(eval.query.count, -1);
  }

  if (eval.base.dim != eval.query.dim) {
    throw std::runtime_error("Base/query dimensions differ in selector evaluation");
  }
  if (eval.base.count == 0 || eval.query.count == 0) {
    throw std::runtime_error("Selector evaluation subsets must not be empty");
  }

  eval.window = requested_window;
  if (eval.window == 0) {
    eval.window = static_cast<uint64_t>(std::min<std::size_t>(1000, eval.base.count / 10 + 1));
  }
  if (eval.window >= eval.base.count) {
    eval.window = eval.base.count > 0 ? static_cast<uint64_t>(eval.base.count - 1) : 0;
  }
  eval.windows = normalize_eval_windows(requested_windows, eval.window, eval.base.count);
  eval.node_counts = normalize_eval_node_counts(requested_node_counts, eval.base.count);

  uint32_t search_k = eval_k + (eval.exclude_self ? 1U : 0U);
  if (search_k > eval.base.count) {
    search_k = static_cast<uint32_t>(eval.base.count);
  }
  if (search_k == 0) {
    throw std::runtime_error("Selector eval_k produced empty search window");
  }
  eval.truth_k = search_k;

  faiss::IndexFlatL2 index(static_cast<int>(eval.base.dim));
  index.add(static_cast<faiss::idx_t>(eval.base.count), eval.base.values.data());
  eval.truth_labels.resize(eval.query.count * search_k);
  std::vector<float> distances(eval.query.count * search_k, 0.0f);
  index.search(static_cast<faiss::idx_t>(eval.query.count),
               eval.query.values.data(),
               static_cast<int>(search_k),
               distances.data(),
               eval.truth_labels.data());
  return eval;
}

} // namespace vortex::selector_internal
