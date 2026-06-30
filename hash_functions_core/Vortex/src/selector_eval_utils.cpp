#include "selector_eval_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vortex::selector_internal {

vector_io::VectorStorage<float> load_vectors_or_throw(const std::filesystem::path& path) {
  if (path.extension() != ".fvecs") {
    throw std::runtime_error("Unsupported vector dataset (expected .fvecs): " + path.string());
  }
  auto vectors = vector_io::read_fvecs(path.string());
  if (vectors.empty()) {
    throw std::runtime_error("Vector dataset is empty: " + path.string());
  }
  return vectors;
}

std::size_t lower_bound_hash(const std::vector<HashEntry64>& entries, uint64_t hash) {
  auto it = std::lower_bound(entries.begin(), entries.end(), hash,
                             [](const HashEntry64& entry, uint64_t value) {
                               return entry.hash < value;
                             });
  return static_cast<std::size_t>(it - entries.begin());
}

std::size_t lower_bound_hash(const std::vector<HashEntry256>& entries,
                             const UInt256& hash) {
  std::size_t lo = 0;
  std::size_t hi = entries.size();
  while (lo < hi) {
    std::size_t mid = lo + (hi - lo) / 2;
    int cmp = compare(entries[mid].hash, hash);
    if (cmp < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

double percentile_ms(std::vector<uint64_t> values, double quantile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  double pos = quantile * static_cast<double>(values.size());
  std::size_t index = static_cast<std::size_t>(std::ceil(pos));
  if (index == 0) {
    index = 1;
  }
  if (index > values.size()) {
    index = values.size();
  }
  uint64_t ns = values[index - 1];
  return static_cast<double>(ns) / 1e6;
}

double percentile_value(std::vector<double> values, double quantile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  double pos = quantile * static_cast<double>(values.size());
  std::size_t index = static_cast<std::size_t>(std::ceil(pos));
  if (index == 0) {
    index = 1;
  }
  if (index > values.size()) {
    index = values.size();
  }
  return values[index - 1];
}

double clamp01(double value) {
  if (value < 0.0) {
    return 0.0;
  }
  if (value > 1.0) {
    return 1.0;
  }
  return value;
}

uint64_t saturating_add_u64(uint64_t lhs, uint64_t rhs) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs + rhs;
}

uint64_t byte_count_u64(std::size_t count, std::size_t item_size) {
  if (item_size != 0 && count > std::numeric_limits<uint64_t>::max() / item_size) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(count) * static_cast<uint64_t>(item_size);
}

uint32_t eval_worker_count(std::size_t item_count, uint32_t requested_threads) {
  if (item_count == 0) {
    return 1;
  }
  uint32_t threads = std::max<uint32_t>(1, requested_threads);
  constexpr std::size_t kMinItemsPerEvalWorker = 1024;
  std::size_t useful_threads =
      (item_count + kMinItemsPerEvalWorker - 1) / kMinItemsPerEvalWorker;
  useful_threads = std::max<std::size_t>(1, useful_threads);
  useful_threads = std::min<std::size_t>(
      useful_threads, static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()));
  return std::min<uint32_t>(threads, static_cast<uint32_t>(useful_threads));
}

uint64_t saturating_mul_u64(uint64_t lhs, uint64_t rhs) {
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs * rhs;
}

void subtract_cache_bytes(uint64_t& value, uint64_t amount) {
  value = amount >= value ? 0 : value - amount;
}

} // namespace vortex::selector_internal
