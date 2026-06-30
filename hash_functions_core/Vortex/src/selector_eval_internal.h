#ifndef VORTEX_V1_SELECTOR_EVAL_INTERNAL_H
#define VORTEX_V1_SELECTOR_EVAL_INTERNAL_H

#include "selector_internal.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace vortex::selector_internal {

std::size_t lower_bound_hash(const std::vector<HashEntry64>& entries, uint64_t hash);
std::size_t lower_bound_hash(const std::vector<HashEntry256>& entries,
                             const UInt256& hash);
double percentile_value(std::vector<double> values, double quantile);
double clamp01(double value);
uint64_t saturating_add_u64(uint64_t lhs, uint64_t rhs);
uint64_t saturating_mul_u64(uint64_t lhs, uint64_t rhs);
uint64_t byte_count_u64(std::size_t count, std::size_t item_size);
void subtract_cache_bytes(uint64_t& value, uint64_t amount);

template <typename Fn>
void parallel_for_chunks(std::size_t item_count, uint32_t threads, Fn&& fn) {
  if (threads <= 1 || item_count == 0) {
    fn(0, item_count);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(threads);
  std::mutex exception_mutex;
  std::exception_ptr first_exception;
  for (uint32_t worker = 0; worker < threads; ++worker) {
    std::size_t begin = item_count * worker / threads;
    std::size_t end = item_count * (worker + 1) / threads;
    workers.emplace_back([&, begin, end]() {
      try {
        fn(begin, end);
      } catch (...) {
        std::lock_guard<std::mutex> lock(exception_mutex);
        if (!first_exception) {
          first_exception = std::current_exception();
        }
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
}

inline uint64_t fnv1a_update(uint64_t hash, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= kFnvPrime;
  }
  return hash;
}

template <typename T>
uint64_t fnv1a_update_value(uint64_t hash, const T& value) {
  return fnv1a_update(hash, &value, sizeof(T));
}

template <typename T>
uint64_t fnv1a_update_vector(uint64_t hash, const std::vector<T>& values) {
  static_assert(std::is_trivially_copyable_v<T>,
                "fnv1a_update_vector hashes raw element bytes");
  std::size_t count = values.size();
  hash = fnv1a_update_value(hash, count);
  if (!values.empty()) {
    hash = fnv1a_update(hash, values.data(), values.size() * sizeof(T));
  }
  return hash;
}

uint64_t eval_nearest_sampled_cache_budget_bytes(TrainCache* cache);
void enforce_eval_nearest_sampled_cache_budget_locked(
    TrainCache* cache,
    const std::string* protected_key = nullptr);
void evict_eval_nearest_sampled_cache_locked(TrainCache* cache);
bool can_use_eval_base_rank_cache(const PhaseLimits& phase, const EvalDataset& eval);
bool is_eval_base_rank_calibration_phase(const PhaseLimits& phase);
bool can_use_eval_nearest_cache(const VortexModel& model,
                                const PhaseLimits& phase,
                                const EvalDataset& eval);
bool should_bypass_inflight_eval_nearest_cache(const VortexModel& model,
                                               const EvalDataset& eval,
                                               TrainCache* cache,
                                               bool calibration_tier);
std::shared_ptr<const EvalNearestCache> get_or_build_eval_nearest_cache(
    const VortexModel& model,
    const EvalDataset& eval,
    uint32_t eval_threads,
    TrainCache* cache,
    bool calibration_tier);

struct EvalHash64CentroidPlan {
  uint64_t range_start_low = 0;
  uint64_t range_size_low = 0;
  uint64_t range_size_high = 0;
  uint64_t cdf_rows = 0;
  double min_dist = 0.0;
  double max_dist = 0.0;
  const double* top_params = nullptr;
  const double* leaf_params = nullptr;
};

enum class EvalHash64CdfKernel {
  generic,
  linear_linear,
  cubic_linear,
  linear_cubic,
  cubic_cubic,
};

struct EvalHash64Plan {
  const VortexModel* model = nullptr;
  uint32_t hash_bits = 0;
  CdfModelType cdf_top_type = CdfModelType::Linear;
  CdfModelType cdf_leaf_type = CdfModelType::Linear;
  uint32_t cdf_leaf_count = 0;
  uint32_t cdf_top_param_count = 0;
  uint32_t cdf_leaf_param_count = 0;
  bool has_cdf = false;
  EvalHash64CdfKernel cdf_kernel = EvalHash64CdfKernel::generic;
  std::vector<EvalHash64CentroidPlan> centroids;
};

uint64_t eval_hash64_from_centroid(const EvalHash64Plan& plan,
                                   uint32_t centroid,
                                   double dist2);
uint64_t eval_hash64_from_centroid_unchecked(const EvalHash64Plan& plan,
                                             uint32_t centroid,
                                             double dist2);
void fill_eval_hash64_from_nearest(const EvalHash64Plan& plan,
                                   const EvalNearestCache& nearest_cache,
                                   std::vector<HashEntry64>* out,
                                   uint32_t threads);
uint64_t eval_hash64_vector(const EvalHash64Plan& plan, const float* vector);
EvalHash64Plan make_eval_hash64_plan(const VortexModel& model);
void verify_eval_hash64_plan_sample(const EvalHash64Plan& plan,
                                    const VortexModel& model,
                                    const EvalDataset& eval,
                                    const EvalNearestCache* nearest_cache);

SelectorBaseRankCacheAttribution make_eval_base_rank_cache_attribution(
    const SelectorCandidateConfig& config,
    const PhaseLimits& phase,
    const EvalDataset& eval,
    const NswSelectorContext* context,
    bool calibration_tier);
std::shared_ptr<const EvalBaseRankCache> get_or_build_eval_base_rank_cache(
    const VortexModel& model,
    const EvalDataset& eval,
    uint32_t eval_threads,
    TrainCache* cache,
    bool use_nearest_cache,
    bool calibration_tier,
    const SelectorBaseRankCacheAttribution* attribution,
    bool* cache_hit,
    double* wait_ms);

} // namespace vortex::selector_internal

#endif // VORTEX_V1_SELECTOR_EVAL_INTERNAL_H
