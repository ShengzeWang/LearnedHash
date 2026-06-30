#include "selector_eval_internal.h"

#include <algorithm>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace vortex::selector_internal {

uint64_t eval_base_rank_model_fingerprint(const VortexModel& model) {
  constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
  uint64_t fingerprint = kFnvOffset;
  fingerprint = fnv1a_update_value(fingerprint, model.dim);
  uint32_t metric = static_cast<uint32_t>(model.metric);
  fingerprint = fnv1a_update_value(fingerprint, metric);
  fingerprint = fnv1a_update_value(fingerprint, model.hash_bits);
  std::size_t centroid_count = model.centroid_count();
  fingerprint = fnv1a_update_value(fingerprint, centroid_count);
  fingerprint = fnv1a_update_vector(fingerprint, model.centroids);
  fingerprint = fnv1a_update_vector(fingerprint, model.range_start);
  fingerprint = fnv1a_update_vector(fingerprint, model.range_size);
  fingerprint = fnv1a_update_vector(fingerprint, model.range_full);
  fingerprint = fnv1a_update_vector(fingerprint, model.min_dist);
  fingerprint = fnv1a_update_vector(fingerprint, model.max_dist);
  uint32_t cdf_top = static_cast<uint32_t>(model.cdf_top_type);
  uint32_t cdf_leaf = static_cast<uint32_t>(model.cdf_leaf_type);
  fingerprint = fnv1a_update_value(fingerprint, cdf_top);
  fingerprint = fnv1a_update_value(fingerprint, cdf_leaf);
  fingerprint = fnv1a_update_value(fingerprint, model.cdf_leaf_count);
  fingerprint = fnv1a_update_value(fingerprint, model.cdf_top_param_count);
  fingerprint = fnv1a_update_value(fingerprint, model.cdf_leaf_param_count);
  fingerprint = fnv1a_update_vector(fingerprint, model.cdf_rows);
  fingerprint = fnv1a_update_vector(fingerprint, model.cdf_top_params);
  fingerprint = fnv1a_update_vector(fingerprint, model.cdf_leaf_params);
  const auto& active = model.router.active_centroids();
  fingerprint = fnv1a_update_vector(fingerprint, active);
  return fingerprint == 0 ? 1 : fingerprint;
}

std::string eval_base_rank_cache_key(const VortexModel& model, const EvalDataset& eval) {
  auto base_address = reinterpret_cast<std::uintptr_t>(eval.base.values.data());
  std::string key;
  key.reserve(128);
  key += eval.base_identity != 0 ? std::to_string(eval.base_identity)
                                 : std::to_string(base_address);
  key += '|';
  key += std::to_string(eval.base.count);
  key += '|';
  key += std::to_string(eval.base.dim);
  key += '|';
  key += std::to_string(eval_base_rank_model_fingerprint(model));
  return key;
}

uint64_t estimate_eval_base_rank_cache_memory_bytes(const EvalBaseRankCache& cache) {
  uint64_t bytes = static_cast<uint64_t>(sizeof(EvalBaseRankCache));
  bytes += saturating_mul_u64(static_cast<uint64_t>(cache.hashed64.capacity()),
                              static_cast<uint64_t>(sizeof(HashEntry64)));
  bytes += saturating_mul_u64(static_cast<uint64_t>(cache.hashed256.capacity()),
                              static_cast<uint64_t>(sizeof(HashEntry256)));
  bytes += saturating_mul_u64(static_cast<uint64_t>(cache.rank_by_index.capacity()),
                              static_cast<uint64_t>(sizeof(std::size_t)));
  return bytes;
}

uint64_t eval_base_rank_sampled_cache_budget_bytes(TrainCache* cache) {
  constexpr uint64_t kMiB = 1024ULL * 1024ULL;
  constexpr uint64_t kMinBudget = 32ULL * kMiB;
  constexpr uint64_t kMaxBudget = 256ULL * kMiB;
  uint64_t budget = kMaxBudget;
  if (cache != nullptr && cache->memory_budget_bytes > 0) {
    budget = cache->memory_budget_bytes / 128;
    if (budget == 0) {
      budget = cache->memory_budget_bytes;
    }
    budget = std::max<uint64_t>(kMinBudget, std::min<uint64_t>(budget, kMaxBudget));
    budget = std::min<uint64_t>(budget, cache->memory_budget_bytes);
  }
  return budget;
}

SelectorBaseRankCacheAttribution make_eval_base_rank_cache_attribution(
    const SelectorCandidateConfig& config,
    const PhaseLimits& phase,
    const EvalDataset& eval,
    const NswSelectorContext* context,
    bool calibration_tier) {
  SelectorBaseRankCacheAttribution out;
  out.phase = phase.name;
  if (context != nullptr) {
    out.nsw_path = context->display_nsw_path;
    out.skeleton_node_count = static_cast<uint32_t>(context->csr.node_ids.size());
    out.skeleton_layer = context->csr.layer;
    if (out.target_skeleton == 0) {
      out.target_skeleton = context->target_skeleton;
    }
  }
  out.target_skeleton = config.target_skeleton != 0
      ? config.target_skeleton
      : out.target_skeleton;
  out.K = config.K;
  out.centroid_knn = config.centroid_knn;
  out.cdf_model_spec = config.cdf_model_spec;
  out.cdf_branching_factor = config.cdf_branching_factor;
  out.enable_2opt = config.enable_2opt;
  out.two_opt_iterations = config.two_opt_iterations;
  out.enable_graph_centroid_order = config.enable_graph_centroid_order;
  out.base_count = static_cast<uint64_t>(eval.base.count);
  out.query_count = static_cast<uint64_t>(eval.query.count);
  out.calibration_tier = calibration_tier;
  return out;
}

std::string eval_base_rank_cache_attribution_key(
    const SelectorBaseRankCacheAttribution& item) {
  std::string key;
  key.reserve(item.phase.size() + item.nsw_path.size() +
              item.cdf_model_spec.size() + 160);
  key += item.phase;
  key += '|';
  key += item.nsw_path;
  key += '|';
  key += std::to_string(item.target_skeleton);
  key += '|';
  key += std::to_string(item.skeleton_node_count);
  key += '|';
  key += std::to_string(item.skeleton_layer);
  key += '|';
  key += std::to_string(item.K);
  key += '|';
  key += std::to_string(item.centroid_knn);
  key += '|';
  key += item.cdf_model_spec;
  key += '|';
  key += std::to_string(item.cdf_branching_factor);
  key += '|';
  key += item.enable_2opt ? "1" : "0";
  key += '|';
  key += std::to_string(item.two_opt_iterations);
  key += '|';
  key += item.enable_graph_centroid_order ? "1" : "0";
  key += '|';
  key += std::to_string(item.base_count);
  key += '|';
  key += std::to_string(item.query_count);
  key += '|';
  key += item.calibration_tier ? "1" : "0";
  return key;
}

SelectorBaseRankCacheAttribution& eval_base_rank_attribution_row_locked(
    TrainCache* cache,
    const std::string& attribution_key,
    const SelectorBaseRankCacheAttribution& seed) {
  auto result =
      cache->base_rank_cache_attribution_by_key.emplace(attribution_key, seed);
  return result.first->second;
}

void erase_eval_base_rank_cache_entry_locked(TrainCache* cache,
                                             const std::string& key) {
  auto meta_it = cache->base_rank_cache_meta_by_key.find(key);
  if (meta_it != cache->base_rank_cache_meta_by_key.end()) {
    const auto& meta = meta_it->second;
    if (meta.ready) {
      if (cache->eval_base_rank_cache_entries > 0) {
        cache->eval_base_rank_cache_entries -= 1;
      }
      subtract_cache_bytes(cache->eval_base_rank_cache_memory_bytes,
                           meta.memory_bytes);
      if (meta.calibration_tier) {
        subtract_cache_bytes(cache->eval_base_rank_cache_calibration_memory_bytes,
                             meta.memory_bytes);
      } else {
        subtract_cache_bytes(cache->eval_base_rank_cache_sampled_memory_bytes,
                             meta.memory_bytes);
      }
      cache->eval_base_rank_cache_evictions += 1;
      cache->eval_base_rank_cache_evicted_bytes =
          saturating_add_u64(cache->eval_base_rank_cache_evicted_bytes,
                             meta.memory_bytes);
      if (!meta.attribution_key.empty()) {
        auto attribution =
            cache->base_rank_cache_attribution_by_key.find(meta.attribution_key);
        if (attribution != cache->base_rank_cache_attribution_by_key.end()) {
          subtract_cache_bytes(attribution->second.retained_memory_bytes,
                               meta.memory_bytes);
          attribution->second.evictions += 1;
          attribution->second.evicted_bytes =
              saturating_add_u64(attribution->second.evicted_bytes,
                                 meta.memory_bytes);
        }
      }
    }
    cache->base_rank_cache_meta_by_key.erase(meta_it);
  }
  cache->base_rank_cache_by_key.erase(key);
}

void enforce_eval_base_rank_sampled_cache_budget_locked(
    TrainCache* cache,
    const std::string* protected_key = nullptr) {
  if (cache == nullptr) {
    return;
  }
  uint64_t budget = eval_base_rank_sampled_cache_budget_bytes(cache);
  cache->eval_base_rank_cache_sampled_budget_bytes = budget;
  while (cache->eval_base_rank_cache_sampled_memory_bytes > budget) {
    const std::string* victim_key = nullptr;
    const EvalBaseRankCacheMeta* victim_meta = nullptr;
    for (const auto& [key, meta] : cache->base_rank_cache_meta_by_key) {
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
    erase_eval_base_rank_cache_entry_locked(cache, key_to_erase);
  }
}

void evict_eval_base_rank_sampled_cache_locked(TrainCache* cache) {
  std::vector<std::string> keys;
  keys.reserve(cache->base_rank_cache_meta_by_key.size());
  for (const auto& [key, meta] : cache->base_rank_cache_meta_by_key) {
    if (meta.ready && !meta.calibration_tier) {
      keys.push_back(key);
    }
  }
  for (const auto& key : keys) {
    erase_eval_base_rank_cache_entry_locked(cache, key);
  }
}

void prepare_eval_base_rank_cache_for_phase(TrainCache* cache,
                                            const PhaseLimits& phase) {
  if (cache == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(cache->mutex);
  cache->eval_nearest_cache_sampled_budget_bytes =
      eval_nearest_sampled_cache_budget_bytes(cache);
  cache->eval_base_rank_cache_sampled_budget_bytes =
      eval_base_rank_sampled_cache_budget_bytes(cache);
  if (is_eval_base_rank_calibration_phase(phase)) {
    evict_eval_nearest_sampled_cache_locked(cache);
    evict_eval_base_rank_sampled_cache_locked(cache);
  } else {
    enforce_eval_nearest_sampled_cache_budget_locked(cache);
    enforce_eval_base_rank_sampled_cache_budget_locked(cache);
  }
}

std::vector<SelectorBaseRankCacheAttribution> snapshot_eval_base_rank_cache_attribution(
    TrainCache* cache) {
  if (cache == nullptr) {
    return {};
  }
  std::vector<SelectorBaseRankCacheAttribution> out;
  {
    std::lock_guard<std::mutex> lock(cache->mutex);
    out.reserve(cache->base_rank_cache_attribution_by_key.size());
    for (const auto& [_, row] : cache->base_rank_cache_attribution_by_key) {
      out.push_back(row);
    }
  }
  std::sort(out.begin(), out.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.hash_ms != rhs.hash_ms) return lhs.hash_ms > rhs.hash_ms;
    if (lhs.misses != rhs.misses) return lhs.misses > rhs.misses;
    if (lhs.build_ms != rhs.build_ms) return lhs.build_ms > rhs.build_ms;
    if (lhs.phase != rhs.phase) return lhs.phase < rhs.phase;
    if (lhs.target_skeleton != rhs.target_skeleton) {
      return lhs.target_skeleton < rhs.target_skeleton;
    }
    if (lhs.K != rhs.K) return lhs.K < rhs.K;
    if (lhs.centroid_knn != rhs.centroid_knn) {
      return lhs.centroid_knn < rhs.centroid_knn;
    }
    if (lhs.enable_graph_centroid_order != rhs.enable_graph_centroid_order) {
      return lhs.enable_graph_centroid_order < rhs.enable_graph_centroid_order;
    }
    return lhs.cdf_model_spec < rhs.cdf_model_spec;
  });
  return out;
}

std::shared_ptr<const EvalBaseRankCache> build_eval_base_rank_cache(
    const VortexModel& model,
    const EvalDataset& eval,
    uint32_t eval_threads,
    TrainCache* cache,
    bool use_nearest_cache,
    bool nearest_cache_calibration_tier) {
  auto out = std::make_shared<EvalBaseRankCache>();
  out->dim = eval.base.dim;
  out->count = eval.base.count;
  out->hash_bits_64 = model.hash_bits <= 64;
  out->hash_threads = eval_worker_count(eval.base.count, eval_threads);
  out->rank_by_index.resize(eval.base.count, 0);

  std::shared_ptr<const EvalNearestCache> nearest_cache;
  bool bypass_inflight_nearest =
      use_nearest_cache &&
      should_bypass_inflight_eval_nearest_cache(model,
                                                eval,
                                                cache,
                                                nearest_cache_calibration_tier);
  if (use_nearest_cache && eval.base.count > 0 && !bypass_inflight_nearest) {
    auto nearest_cache_start = SelectorClock::now();
    nearest_cache = get_or_build_eval_nearest_cache(model,
                                                    eval,
                                                    out->hash_threads,
                                                    cache,
                                                    nearest_cache_calibration_tier);
    out->nearest_cache_ms = elapsed_ms(nearest_cache_start);
    if (nearest_cache->count != eval.base.count ||
        nearest_cache->centroid.size() != eval.base.count ||
        nearest_cache->dist2.size() != eval.base.count) {
      throw std::runtime_error("Selector eval nearest cache size mismatch");
    }
  }

  auto hash_base_start = SelectorClock::now();
  if (out->hash_bits_64) {
    EvalHash64Plan hash_plan = make_eval_hash64_plan(model);
    verify_eval_hash64_plan_sample(hash_plan, model, eval, nearest_cache.get());
    out->hashed64.resize(eval.base.count);
    if (nearest_cache) {
      fill_eval_hash64_from_nearest(hash_plan,
                                    *nearest_cache,
                                    &out->hashed64,
                                    out->hash_threads);
    } else {
      parallel_for_chunks(eval.base.count, out->hash_threads, [&](std::size_t begin,
                                                                  std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          uint64_t hash =
              eval_hash64_vector(hash_plan,
                                 eval.base.values.data() + i * eval.base.dim);
          out->hashed64[i] = HashEntry64{hash, static_cast<uint32_t>(i)};
        }
      });
    }
  } else {
    out->hashed256.resize(eval.base.count);
    parallel_for_chunks(eval.base.count, out->hash_threads, [&](std::size_t begin,
                                                                std::size_t end) {
      for (std::size_t i = begin; i < end; ++i) {
        UInt256 hash = nearest_cache
            ? model.hash_from_centroid(nearest_cache->centroid[i],
                                       nearest_cache->dist2[i]).value
            : model.hash(eval.base.values.data() + i * eval.base.dim).value;
        out->hashed256[i] = HashEntry256{hash, static_cast<uint32_t>(i)};
      }
    });
  }
  out->hash_base_ms = elapsed_ms(hash_base_start);

  auto sort_base_start = SelectorClock::now();
  if (out->hash_bits_64) {
    std::sort(out->hashed64.begin(), out->hashed64.end(),
              [](const HashEntry64& a, const HashEntry64& b) {
                if (a.hash != b.hash) return a.hash < b.hash;
                return a.index < b.index;
              });
  } else {
    std::sort(out->hashed256.begin(), out->hashed256.end(),
              [](const HashEntry256& a, const HashEntry256& b) {
                int cmp = compare(a.hash, b.hash);
                if (cmp != 0) return cmp < 0;
                return a.index < b.index;
              });
  }
  out->sort_base_ms = elapsed_ms(sort_base_start);

  auto rank_index_start = SelectorClock::now();
  std::size_t entry_count = out->hash_bits_64 ? out->hashed64.size()
                                              : out->hashed256.size();
  out->rank_threads = eval_worker_count(entry_count, eval_threads);
  if (out->hash_bits_64) {
    parallel_for_chunks(out->hashed64.size(), out->rank_threads,
                        [&](std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; ++i) {
        out->rank_by_index[out->hashed64[i].index] = i;
      }
    });
  } else {
    parallel_for_chunks(out->hashed256.size(), out->rank_threads,
                        [&](std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; ++i) {
        out->rank_by_index[out->hashed256[i].index] = i;
      }
    });
  }
  out->rank_index_ms = elapsed_ms(rank_index_start);
  out->memory_bytes = estimate_eval_base_rank_cache_memory_bytes(*out);
  return out;
}

std::shared_ptr<const EvalBaseRankCache> get_or_build_eval_base_rank_cache(
    const VortexModel& model,
    const EvalDataset& eval,
    uint32_t eval_threads,
    TrainCache* cache,
    bool use_nearest_cache,
    bool calibration_tier,
    const SelectorBaseRankCacheAttribution* attribution,
    bool* cache_hit,
    double* wait_ms) {
  if (cache_hit != nullptr) {
    *cache_hit = false;
  }
  if (wait_ms != nullptr) {
    *wait_ms = 0.0;
  }
  if (cache == nullptr) {
    return build_eval_base_rank_cache(model,
                                      eval,
                                      eval_threads,
                                      cache,
                                      use_nearest_cache,
                                      calibration_tier);
  }

  std::string key = eval_base_rank_cache_key(model, eval);
  std::string attribution_key;
  if (attribution != nullptr) {
    attribution_key = eval_base_rank_cache_attribution_key(*attribution);
  }
  std::shared_future<std::shared_ptr<const EvalBaseRankCache>> rank_future;
  std::optional<std::promise<std::shared_ptr<const EvalBaseRankCache>>> build_promise;
  bool hit = false;
  {
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->eval_base_rank_cache_sampled_budget_bytes =
        eval_base_rank_sampled_cache_budget_bytes(cache);
    auto found = cache->base_rank_cache_by_key.find(key);
    if (found != cache->base_rank_cache_by_key.end()) {
      rank_future = found->second;
      cache->eval_base_rank_cache_hits += 1;
      if (attribution != nullptr) {
        auto& row =
            eval_base_rank_attribution_row_locked(cache, attribution_key, *attribution);
        row.hits += 1;
      }
      auto meta = cache->base_rank_cache_meta_by_key.find(key);
      if (meta != cache->base_rank_cache_meta_by_key.end()) {
        meta->second.hits += 1;
        meta->second.last_access_tick = ++cache->eval_base_rank_cache_access_tick;
      }
      hit = true;
    } else {
      build_promise.emplace();
      rank_future = build_promise->get_future().share();
      auto [meta_it, inserted] =
          cache->base_rank_cache_meta_by_key.emplace(key, EvalBaseRankCacheMeta{});
      meta_it->second.last_access_tick = ++cache->eval_base_rank_cache_access_tick;
      meta_it->second.calibration_tier = calibration_tier;
      meta_it->second.attribution_key = attribution_key;
      cache->base_rank_cache_by_key.emplace(key, rank_future);
      cache->eval_base_rank_cache_misses += 1;
      if (attribution != nullptr) {
        auto& row =
            eval_base_rank_attribution_row_locked(cache, attribution_key, *attribution);
        row.misses += 1;
      }
    }
  }

  if (cache_hit != nullptr) {
    *cache_hit = hit;
  }
  if (build_promise.has_value()) {
    try {
      auto build_start = SelectorClock::now();
      auto rank_cache =
          build_eval_base_rank_cache(model,
                                     eval,
                                     eval_threads,
                                     cache,
                                     use_nearest_cache,
                                     calibration_tier);
      double build_ms = elapsed_ms(build_start);
      {
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->eval_base_rank_cache_build_ms += build_ms;
        cache->eval_base_rank_cache_nearest_ms += rank_cache->nearest_cache_ms;
        cache->eval_base_rank_cache_hash_ms += rank_cache->hash_base_ms;
        cache->eval_base_rank_cache_sort_ms += rank_cache->sort_base_ms;
        cache->eval_base_rank_cache_rank_index_ms += rank_cache->rank_index_ms;
        double component_ms = rank_cache->nearest_cache_ms +
                              rank_cache->hash_base_ms +
                              rank_cache->sort_base_ms +
                              rank_cache->rank_index_ms;
        if (build_ms > component_ms) {
          cache->eval_base_rank_cache_overhead_ms += build_ms - component_ms;
        }
        if (attribution != nullptr) {
          auto& row =
              eval_base_rank_attribution_row_locked(cache, attribution_key, *attribution);
          row.builds += 1;
          row.build_ms += build_ms;
          row.nearest_ms += rank_cache->nearest_cache_ms;
          row.hash_ms += rank_cache->hash_base_ms;
          row.sort_ms += rank_cache->sort_base_ms;
          row.rank_index_ms += rank_cache->rank_index_ms;
          if (build_ms > component_ms) {
            row.overhead_ms += build_ms - component_ms;
          }
          row.added_memory_bytes =
              saturating_add_u64(row.added_memory_bytes, rank_cache->memory_bytes);
          row.retained_memory_bytes =
              saturating_add_u64(row.retained_memory_bytes, rank_cache->memory_bytes);
        }
        auto meta = cache->base_rank_cache_meta_by_key.find(key);
        if (meta != cache->base_rank_cache_meta_by_key.end()) {
          meta->second.ready = true;
          meta->second.memory_bytes = rank_cache->memory_bytes;
          meta->second.calibration_tier = calibration_tier;
          meta->second.attribution_key = attribution_key;
        }
        cache->eval_base_rank_cache_entries += 1;
        cache->eval_base_rank_cache_memory_bytes =
            saturating_add_u64(cache->eval_base_rank_cache_memory_bytes,
                               rank_cache->memory_bytes);
        if (calibration_tier) {
          cache->eval_base_rank_cache_calibration_memory_bytes =
              saturating_add_u64(cache->eval_base_rank_cache_calibration_memory_bytes,
                                 rank_cache->memory_bytes);
        } else {
          cache->eval_base_rank_cache_sampled_memory_bytes =
              saturating_add_u64(cache->eval_base_rank_cache_sampled_memory_bytes,
                                 rank_cache->memory_bytes);
        }
      }
      build_promise->set_value(rank_cache);
      if (!calibration_tier) {
        std::lock_guard<std::mutex> lock(cache->mutex);
        enforce_eval_base_rank_sampled_cache_budget_locked(cache, &key);
      }
      return rank_cache;
    } catch (...) {
      auto exception = std::current_exception();
      {
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->base_rank_cache_by_key.erase(key);
        cache->base_rank_cache_meta_by_key.erase(key);
      }
      build_promise->set_exception(exception);
      throw;
    }
  }

  auto wait_start = SelectorClock::now();
  auto rank_cache = rank_future.get();
  double local_wait_ms = elapsed_ms(wait_start);
  if (hit) {
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->eval_base_rank_cache_wait_ms += local_wait_ms;
    if (attribution != nullptr) {
      auto& row =
          eval_base_rank_attribution_row_locked(cache, attribution_key, *attribution);
      row.wait_ms += local_wait_ms;
    }
  }
  if (wait_ms != nullptr) {
    *wait_ms = local_wait_ms;
  }
  return rank_cache;
}

} // namespace vortex::selector_internal
