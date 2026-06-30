#include "selector_eval_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace vortex::selector_internal {

std::string eval_nearest_cache_key(const VortexModel& model, const EvalDataset& eval) {
  constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
  uint64_t fingerprint = kFnvOffset;
  fingerprint = fnv1a_update_value(fingerprint, model.dim);
  uint32_t metric = static_cast<uint32_t>(model.metric);
  fingerprint = fnv1a_update_value(fingerprint, metric);
  fingerprint = fnv1a_update_value(fingerprint, model.hash_bits);
  std::size_t centroid_count = model.centroid_count();
  fingerprint = fnv1a_update_value(fingerprint, centroid_count);
  if (!model.centroids.empty()) {
    fingerprint = fnv1a_update(fingerprint,
                               model.centroids.data(),
                               model.centroids.size() * sizeof(float));
  }
  const auto& active = model.router.active_centroids();
  std::size_t active_count = active.size();
  fingerprint = fnv1a_update_value(fingerprint, active_count);
  if (!active.empty()) {
    fingerprint = fnv1a_update(fingerprint,
                               active.data(),
                               active.size() * sizeof(uint32_t));
  }

  auto base_address = reinterpret_cast<std::uintptr_t>(eval.base.values.data());
  std::string key;
  key.reserve(96);
  key += eval.base_identity != 0 ? std::to_string(eval.base_identity)
                                 : std::to_string(base_address);
  key += '|';
  key += std::to_string(eval.base.count);
  key += '|';
  key += std::to_string(eval.base.dim);
  key += '|';
  key += std::to_string(fingerprint);
  return key;
}

rm_model::CentroidRouter::QueryResult nearest_for_eval_cache_exact(
    const VortexModel& model,
    const float* vector,
    uint64_t* distance_terms_evaluated) {
  if (model.dim == 0 || model.centroids.empty()) {
    throw std::runtime_error("Selector eval nearest cache: model centroids are empty");
  }
  if (vector == nullptr) {
    throw std::runtime_error("Selector eval nearest cache: input vector is null");
  }

  const uint32_t dim = model.dim;
  const std::size_t centroid_count = model.centroid_count();
  const float* centroids = model.centroids.data();
  const auto& active = model.router.active_centroids();
  double best = std::numeric_limits<double>::infinity();
  uint32_t best_idx = 0;

  auto scan_centroid = [&](uint32_t idx) {
    const float* centroid = centroids + static_cast<std::size_t>(idx) * dim;
    double acc = 0.0;
    for (uint32_t d = 0; d < dim; ++d) {
      double diff = static_cast<double>(vector[d]) - static_cast<double>(centroid[d]);
      acc += diff * diff;
      if (distance_terms_evaluated != nullptr) {
        *distance_terms_evaluated += 1;
      }
      if (acc >= best) {
        break;
      }
    }
    if (acc < best) {
      best = acc;
      best_idx = idx;
    }
  };

  if (active.empty() || active.size() == centroid_count) {
    for (std::size_t idx = 0; idx < centroid_count; ++idx) {
      scan_centroid(static_cast<uint32_t>(idx));
    }
  } else {
    for (uint32_t idx : active) {
      scan_centroid(idx);
    }
  }

  return {best_idx, best};
}

uint64_t estimate_eval_nearest_cache_memory_bytes(const EvalNearestCache& cache) {
  uint64_t bytes = static_cast<uint64_t>(sizeof(EvalNearestCache));
  bytes += saturating_mul_u64(static_cast<uint64_t>(cache.centroid.capacity()),
                              static_cast<uint64_t>(sizeof(uint32_t)));
  bytes += saturating_mul_u64(static_cast<uint64_t>(cache.dist2.capacity()),
                              static_cast<uint64_t>(sizeof(double)));
  return bytes;
}

uint64_t eval_nearest_sampled_cache_budget_bytes(TrainCache* cache) {
  constexpr uint64_t kMiB = 1024ULL * 1024ULL;
  constexpr uint64_t kMinBudget = 8ULL * kMiB;
  constexpr uint64_t kMaxBudget = 64ULL * kMiB;
  uint64_t budget = kMaxBudget;
  if (cache != nullptr && cache->memory_budget_bytes > 0) {
    budget = cache->memory_budget_bytes / 256;
    if (budget == 0) {
      budget = cache->memory_budget_bytes;
    }
    budget = std::max<uint64_t>(kMinBudget, std::min<uint64_t>(budget, kMaxBudget));
    budget = std::min<uint64_t>(budget, cache->memory_budget_bytes);
  }
  return budget;
}

void erase_eval_nearest_cache_entry_locked(TrainCache* cache,
                                           const std::string& key) {
  auto meta_it = cache->nearest_cache_meta_by_key.find(key);
  if (meta_it != cache->nearest_cache_meta_by_key.end()) {
    const auto& meta = meta_it->second;
    if (meta.ready) {
      if (cache->eval_nearest_cache_entries > 0) {
        cache->eval_nearest_cache_entries -= 1;
      }
      subtract_cache_bytes(cache->eval_nearest_cache_memory_bytes,
                           meta.memory_bytes);
      if (meta.calibration_tier) {
        subtract_cache_bytes(cache->eval_nearest_cache_calibration_memory_bytes,
                             meta.memory_bytes);
      } else {
        subtract_cache_bytes(cache->eval_nearest_cache_sampled_memory_bytes,
                             meta.memory_bytes);
      }
      cache->eval_nearest_cache_evictions += 1;
      cache->eval_nearest_cache_evicted_bytes =
          saturating_add_u64(cache->eval_nearest_cache_evicted_bytes,
                             meta.memory_bytes);
    }
    cache->nearest_cache_meta_by_key.erase(meta_it);
  }
  cache->nearest_cache_by_key.erase(key);
}

void enforce_eval_nearest_sampled_cache_budget_locked(
    TrainCache* cache,
    const std::string* protected_key) {
  if (cache == nullptr) {
    return;
  }
  uint64_t budget = eval_nearest_sampled_cache_budget_bytes(cache);
  cache->eval_nearest_cache_sampled_budget_bytes = budget;
  while (cache->eval_nearest_cache_sampled_memory_bytes > budget) {
    const std::string* victim_key = nullptr;
    const EvalNearestCacheMeta* victim_meta = nullptr;
    for (const auto& [key, meta] : cache->nearest_cache_meta_by_key) {
      if (!meta.ready || meta.calibration_tier) {
        continue;
      }
      if (protected_key != nullptr && key == *protected_key) {
        continue;
      }
      if (victim_meta == nullptr ||
          meta.hits < victim_meta->hits ||
          (meta.hits == victim_meta->hits &&
           meta.last_access_tick < victim_meta->last_access_tick) ||
          (meta.hits == victim_meta->hits &&
           meta.last_access_tick == victim_meta->last_access_tick &&
           meta.memory_bytes > victim_meta->memory_bytes)) {
        victim_key = &key;
        victim_meta = &meta;
      }
    }
    if (victim_key == nullptr) {
      break;
    }
    std::string key_to_erase = *victim_key;
    erase_eval_nearest_cache_entry_locked(cache, key_to_erase);
  }
}

void evict_eval_nearest_sampled_cache_locked(TrainCache* cache) {
  std::vector<std::string> keys;
  keys.reserve(cache->nearest_cache_meta_by_key.size());
  for (const auto& [key, meta] : cache->nearest_cache_meta_by_key) {
    if (meta.ready && !meta.calibration_tier) {
      keys.push_back(key);
    }
  }
  for (const auto& key : keys) {
    erase_eval_nearest_cache_entry_locked(cache, key);
  }
}

bool should_use_eval_early_abandon(const VortexModel& model,
                                   const EvalDataset& eval,
                                   uint64_t active_count) {
  constexpr std::size_t kMinBaseCount = 100000;
  // The sampled timing guard below keeps this exact kernel off when branch
  // overhead outweighs skipped distance terms; lower the gate enough for
  // full-SIFT K=512 calibration caches to opt in when the data benefits.
  constexpr uint64_t kMinActiveCentroids = 256;
  constexpr uint32_t kMinDim = 16;
  if (eval.base.count < kMinBaseCount ||
      active_count < kMinActiveCentroids ||
      eval.base.dim < kMinDim) {
    return false;
  }

  constexpr std::size_t kSampleCount = 128;
  std::size_t samples = std::min<std::size_t>(kSampleCount, eval.base.count);
  if (samples == 0) {
    return false;
  }

  std::vector<std::size_t> indices;
  indices.reserve(samples);
  for (std::size_t i = 0; i < samples; ++i) {
    indices.push_back(i * eval.base.count / samples);
  }

  auto router_start = SelectorClock::now();
  std::vector<rm_model::CentroidRouter::QueryResult> router_results;
  router_results.reserve(samples);
  for (std::size_t index : indices) {
    const float* vec = eval.base.values.data() + index * eval.base.dim;
    router_results.push_back(model.router.nearest(vec));
  }
  double router_ms = elapsed_ms(router_start);

  auto early_start = SelectorClock::now();
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const float* vec = eval.base.values.data() + indices[i] * eval.base.dim;
    auto early = nearest_for_eval_cache_exact(model, vec, nullptr);
    if (early.centroid != router_results[i].centroid ||
        std::abs(early.dist2 - router_results[i].dist2) > 1e-9) {
      return false;
    }
  }
  double early_ms = elapsed_ms(early_start);

  return early_ms * 1.10 < router_ms;
}

std::shared_ptr<const EvalNearestCache> build_eval_nearest_cache(const VortexModel& model,
                                                                 const EvalDataset& eval,
                                                                 uint32_t eval_threads) {
  auto out = std::make_shared<EvalNearestCache>();
  out->dim = eval.base.dim;
  out->count = eval.base.count;
  out->centroid.resize(eval.base.count);
  out->dist2.resize(eval.base.count);

  uint32_t threads = eval_worker_count(eval.base.count, eval_threads);
  const auto& active = model.router.active_centroids();
  uint64_t centroid_terms = static_cast<uint64_t>(
      active.empty() ? model.centroid_count() : active.size());
  uint64_t max_terms = saturating_mul_u64(
      saturating_mul_u64(static_cast<uint64_t>(eval.base.count), centroid_terms),
      static_cast<uint64_t>(eval.base.dim));
  bool use_early_abandon = should_use_eval_early_abandon(model, eval, centroid_terms);
  out->used_early_abandon = use_early_abandon;

  if (!use_early_abandon) {
    parallel_for_chunks(eval.base.count, threads, [&](std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; ++i) {
        const float* vec = eval.base.values.data() + i * eval.base.dim;
        auto nearest = model.router.nearest(vec);
        out->centroid[i] = nearest.centroid;
        out->dist2[i] = nearest.dist2;
      }
    });
    out->distance_terms_evaluated = max_terms;
    out->distance_terms_skipped = 0;
    out->memory_bytes = estimate_eval_nearest_cache_memory_bytes(*out);
    return out;
  }

  std::atomic<uint64_t> distance_terms_evaluated{0};
  parallel_for_chunks(eval.base.count, threads, [&](std::size_t begin, std::size_t end) {
    uint64_t local_distance_terms = 0;
    for (std::size_t i = begin; i < end; ++i) {
      const float* vec = eval.base.values.data() + i * eval.base.dim;
      auto nearest = nearest_for_eval_cache_exact(model, vec, &local_distance_terms);
      out->centroid[i] = nearest.centroid;
      out->dist2[i] = nearest.dist2;
    }
    distance_terms_evaluated.fetch_add(local_distance_terms, std::memory_order_relaxed);
  });
  out->distance_terms_evaluated =
      distance_terms_evaluated.load(std::memory_order_relaxed);
  out->distance_terms_skipped =
      max_terms >= out->distance_terms_evaluated
          ? max_terms - out->distance_terms_evaluated
          : 0;
  out->memory_bytes = estimate_eval_nearest_cache_memory_bytes(*out);
  return out;
}

std::shared_ptr<const EvalNearestCache> get_or_build_eval_nearest_cache(
    const VortexModel& model,
    const EvalDataset& eval,
    uint32_t eval_threads,
    TrainCache* cache,
    bool calibration_tier) {
  if (cache == nullptr) {
    return build_eval_nearest_cache(model, eval, eval_threads);
  }

  std::string key = eval_nearest_cache_key(model, eval);
  std::shared_future<std::shared_ptr<const EvalNearestCache>> nearest_future;
  std::optional<std::promise<std::shared_ptr<const EvalNearestCache>>> build_promise;
  bool cache_hit = false;
  {
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->eval_nearest_cache_sampled_budget_bytes =
        eval_nearest_sampled_cache_budget_bytes(cache);
    auto found = cache->nearest_cache_by_key.find(key);
    if (found != cache->nearest_cache_by_key.end()) {
      nearest_future = found->second;
      cache->eval_nearest_cache_hits += 1;
      auto meta = cache->nearest_cache_meta_by_key.find(key);
      if (meta != cache->nearest_cache_meta_by_key.end()) {
        meta->second.hits += 1;
        meta->second.last_access_tick = ++cache->eval_nearest_cache_access_tick;
      }
      cache_hit = true;
    } else {
      build_promise.emplace();
      nearest_future = build_promise->get_future().share();
      auto [meta_it, inserted] =
          cache->nearest_cache_meta_by_key.emplace(key, EvalNearestCacheMeta{});
      meta_it->second.last_access_tick = ++cache->eval_nearest_cache_access_tick;
      meta_it->second.calibration_tier = calibration_tier;
      cache->nearest_cache_by_key.emplace(key, nearest_future);
      cache->eval_nearest_cache_misses += 1;
    }
  }

  if (build_promise.has_value()) {
    try {
      auto build_start = SelectorClock::now();
      auto nearest = build_eval_nearest_cache(model, eval, eval_threads);
      double build_ms = elapsed_ms(build_start);
      {
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->eval_nearest_cache_build_ms += build_ms;
        if (calibration_tier) {
          cache->eval_nearest_cache_calibration_build_ms += build_ms;
        } else {
          cache->eval_nearest_cache_sampled_build_ms += build_ms;
        }
        cache->eval_nearest_cache_distance_terms += nearest->distance_terms_evaluated;
        cache->eval_nearest_cache_skipped_distance_terms += nearest->distance_terms_skipped;
        if (nearest->used_early_abandon) {
          cache->eval_nearest_cache_early_abandon_builds += 1;
        }
        auto meta = cache->nearest_cache_meta_by_key.find(key);
        if (meta != cache->nearest_cache_meta_by_key.end()) {
          meta->second.ready = true;
          meta->second.memory_bytes = nearest->memory_bytes;
          meta->second.calibration_tier = calibration_tier;
        }
        cache->eval_nearest_cache_entries += 1;
        cache->eval_nearest_cache_memory_bytes =
            saturating_add_u64(cache->eval_nearest_cache_memory_bytes,
                               nearest->memory_bytes);
        if (calibration_tier) {
          cache->eval_nearest_cache_calibration_memory_bytes =
              saturating_add_u64(cache->eval_nearest_cache_calibration_memory_bytes,
                                 nearest->memory_bytes);
        } else {
          cache->eval_nearest_cache_sampled_memory_bytes =
              saturating_add_u64(cache->eval_nearest_cache_sampled_memory_bytes,
                                 nearest->memory_bytes);
        }
      }
      build_promise->set_value(nearest);
      if (!calibration_tier) {
        std::lock_guard<std::mutex> lock(cache->mutex);
        enforce_eval_nearest_sampled_cache_budget_locked(cache, &key);
      }
      return nearest;
    } catch (...) {
      auto exception = std::current_exception();
      {
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->nearest_cache_by_key.erase(key);
        cache->nearest_cache_meta_by_key.erase(key);
      }
      build_promise->set_exception(exception);
      throw;
    }
  }

  auto wait_start = SelectorClock::now();
  auto nearest = nearest_future.get();
  if (cache_hit) {
    double wait_ms = elapsed_ms(wait_start);
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->eval_nearest_cache_wait_ms += wait_ms;
    if (calibration_tier) {
      cache->eval_nearest_cache_calibration_wait_ms += wait_ms;
    } else {
      cache->eval_nearest_cache_sampled_wait_ms += wait_ms;
    }
  }
  return nearest;
}

bool can_use_eval_base_rank_cache(const PhaseLimits&, const EvalDataset& eval) {
  return eval.base.count > 0;
}

bool is_eval_base_rank_calibration_phase(const PhaseLimits& phase) {
  return phase.name == "final_calibration" ||
         phase.name == "final_calibration_screen";
}

bool can_use_eval_nearest_cache(const VortexModel& model,
                                const PhaseLimits& phase,
                                const EvalDataset& eval) {
  if (eval.base.count == 0) {
    return false;
  }
  if (is_eval_base_rank_calibration_phase(phase)) {
    return true;
  }
  // Beam refinement often compares many CDF/branch/order variants over the same
  // centroids. Reusing the exact nearest-centroid layer keeps those sampled
  // evaluations exact while avoiding repeated centroid scans.
  bool sampled_reuse_phase =
      phase.name == "phase1_beam" ||
      phase.name == "phase2" ||
      phase.name == "post_exploration";
  if (!sampled_reuse_phase) {
    return false;
  }
  constexpr std::size_t kMinSampledBaseCount = 10000;
  constexpr std::size_t kMinCentroidCount = 256;
  return eval.base.count >= kMinSampledBaseCount &&
         model.centroid_count() >= kMinCentroidCount;
}

bool should_bypass_inflight_eval_nearest_cache(const VortexModel& model,
                                               const EvalDataset& eval,
                                               TrainCache* cache,
                                               bool calibration_tier) {
  if (cache == nullptr || calibration_tier || eval.base.count == 0) {
    return false;
  }

  std::string key = eval_nearest_cache_key(model, eval);
  std::lock_guard<std::mutex> lock(cache->mutex);
  auto found = cache->nearest_cache_by_key.find(key);
  if (found == cache->nearest_cache_by_key.end()) {
    return false;
  }
  auto meta = cache->nearest_cache_meta_by_key.find(key);
  if (meta != cache->nearest_cache_meta_by_key.end() && meta->second.ready) {
    return false;
  }
  if (found->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    return false;
  }
  cache->eval_nearest_cache_inflight_bypasses += 1;
  return true;
}

} // namespace vortex::selector_internal
