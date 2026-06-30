#include "selector_internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace vortex::selector_internal {

namespace {

enum class TrainingCacheState {
  missing,
  in_flight,
  ready,
};

enum class TrainingComponentState {
  missing,
  in_flight,
  ready,
};

struct CandidateCacheObservation {
  TrainingCacheState model = TrainingCacheState::missing;
  TrainingComponentState base = TrainingComponentState::missing;
  TrainingComponentState order = TrainingComponentState::missing;
  TrainingComponentState cdf = TrainingComponentState::missing;
  uint32_t component_reuse_score = 0;
  bool fresh_base = true;
};

struct CandidateLaunchBudget {
  double guard_seconds = 0.0;
  bool apply_final_calibration_reserve = false;
};

struct CandidateCachePartition {
  std::vector<std::size_t> ready;
  std::vector<std::size_t> in_flight;
  std::vector<std::size_t> missing;
  uint32_t reuse_prioritized_candidates = 0;
  uint32_t fresh_base_candidates = 0;
};

double observed_train_seconds(TrainCache* cache) {
  if (cache == nullptr) {
    return 0.0;
  }
  std::lock_guard<std::mutex> lock(cache->mutex);
  return cache->max_observed_train_seconds;
}

double observed_eval_hash_seconds_per_vector(TrainCache* cache) {
  if (cache == nullptr) {
    return 0.0;
  }
  std::lock_guard<std::mutex> lock(cache->mutex);
  return cache->max_observed_eval_hash_seconds_per_vector;
}

uint32_t final_calibration_nearest_cache_prewarm_threads(
    const SelectorOptions& options,
    const EvalDataset& eval) {
  uint32_t requested_threads = std::max<uint32_t>(1, options.threads);
  uint32_t hardware_threads = std::thread::hardware_concurrency();
  if (hardware_threads == 0) {
    hardware_threads = requested_threads;
  }
  if (options.selector_parallelism == 0 || eval.base.count >= 500'000) {
    // Final-calibration cache prewarm is sequential across promoted models;
    // use internal parallelism for large bases even when phase search used
    // single-threaded candidates to control memory pressure.
    requested_threads = std::max(requested_threads, hardware_threads);
  }
  return eval_worker_count(eval.base.count, requested_threads);
}

std::shared_future<TrainedCandidate> ready_trained_candidate_future(
    const SelectorCandidateConfig& config,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    TrainCache* cache) {
  if (cache == nullptr) {
    return {};
  }

  std::shared_future<TrainedCandidate> future;
  {
    std::lock_guard<std::mutex> lock(cache->mutex);
    auto found = cache->future_by_key.find(
        selector_training_cache_key(context, config, options));
    if (found == cache->future_by_key.end()) {
      return {};
    }
    future = found->second;
  }
  if (!future.valid() ||
      future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    return {};
  }
  return future;
}

void prewarm_final_calibration_nearest_caches(
    const std::vector<SelectorCandidateConfig>& candidates,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    const EvalDataset& eval,
    const PhaseLimits& phase,
    TrainCache* cache,
    SearchController* search) {
  if (cache == nullptr ||
      candidates.empty() ||
      eval.base.count == 0 ||
      !is_eval_base_rank_calibration_phase(phase)) {
    return;
  }

  uint32_t prewarm_threads =
      final_calibration_nearest_cache_prewarm_threads(options, eval);
  uint32_t requests = 0;
  uint32_t ready_models = 0;
  uint32_t skipped = 0;
  uint32_t failures = 0;
  auto start = SelectorClock::now();

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto& config = candidates[i];
    if (search != nullptr && !search->can_launch_more_work()) {
      skipped += static_cast<uint32_t>(
          std::min<std::size_t>(candidates.size() - i,
                                std::numeric_limits<uint32_t>::max()));
      break;
    }
    std::shared_future<TrainedCandidate> future =
        ready_trained_candidate_future(config, options, context, cache);
    if (!future.valid()) {
      skipped += 1;
      continue;
    }

    try {
      const TrainedCandidate& trained = future.get();
      ready_models += 1;
      if (!can_use_eval_nearest_cache(trained.model, phase, eval)) {
        skipped += 1;
        continue;
      }
      requests += 1;
      (void)get_or_build_eval_nearest_cache(trained.model,
                                            eval,
                                            prewarm_threads,
                                            cache,
                                            true);
    } catch (...) {
      // Prewarm is an optimization. Candidate evaluation will surface the
      // underlying training/cache error with the normal per-candidate handling.
      failures += 1;
    }
  }

  double prewarm_ms = elapsed_ms(start);
  if (requests == 0 && ready_models == 0 && skipped == 0 && failures == 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(cache->mutex);
  cache->eval_nearest_cache_prewarm_requests += requests;
  cache->eval_nearest_cache_prewarm_ready_models += ready_models;
  cache->eval_nearest_cache_prewarm_skipped += skipped;
  cache->eval_nearest_cache_prewarm_failures += failures;
  cache->eval_nearest_cache_prewarm_threads =
      std::max(cache->eval_nearest_cache_prewarm_threads, prewarm_threads);
  cache->eval_nearest_cache_prewarm_ms += prewarm_ms;
}

std::string context_k_observation_key(const std::string& context_key, uint32_t k) {
  return context_key + "||K|" + std::to_string(k);
}

std::string selector_context_key(const NswSelectorContext& context) {
  if (!context.display_nsw_path.empty()) {
    return context.display_nsw_path;
  }
  if (!context.skeleton_identity.empty()) {
    return context.skeleton_identity;
  }
  return context.nsw_path.string();
}

std::size_t nearest_k_index(const std::vector<uint32_t>& values, uint32_t k) {
  if (values.empty()) {
    return 0;
  }
  auto it = std::lower_bound(values.begin(), values.end(), k);
  if (it == values.begin()) {
    return 0;
  }
  if (it == values.end()) {
    return values.size() - 1;
  }
  std::size_t hi = static_cast<std::size_t>(it - values.begin());
  std::size_t lo = hi - 1;
  uint32_t lo_delta = values[lo] > k ? values[lo] - k : k - values[lo];
  uint32_t hi_delta = values[hi] > k ? values[hi] - k : k - values[hi];
  return lo_delta <= hi_delta ? lo : hi;
}

TrainingCacheState training_cache_state(const SelectorCandidateConfig& config,
                                        const SelectorOptions& options,
                                        const NswSelectorContext& context,
                                        TrainCache* cache) {
  if (cache == nullptr) {
    return TrainingCacheState::missing;
  }
  std::shared_future<TrainedCandidate> candidate_future;
  {
    std::lock_guard<std::mutex> lock(cache->mutex);
    auto found = cache->future_by_key.find(
        selector_training_cache_key(context, config, options));
    if (found == cache->future_by_key.end()) {
      return TrainingCacheState::missing;
    }
    candidate_future = found->second;
  }
  return candidate_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready
             ? TrainingCacheState::ready
             : TrainingCacheState::in_flight;
}

TrainingComponentState cdf_component_state(
    const std::shared_future<std::shared_ptr<const training_internal::VortexCdfFit>>& future) {
  if (!future.valid()) {
    return TrainingComponentState::missing;
  }
  return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready
             ? TrainingComponentState::ready
             : TrainingComponentState::in_flight;
}

TrainingCacheState model_future_state(const std::shared_future<TrainedCandidate>& future) {
  if (!future.valid()) {
    return TrainingCacheState::missing;
  }
  return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready
             ? TrainingCacheState::ready
             : TrainingCacheState::in_flight;
}

CandidateCacheObservation observe_candidate_cache(
    const SelectorCandidateConfig& config,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    TrainCache* cache) {
  CandidateCacheObservation observation;
  if (cache == nullptr) {
    return observation;
  }

  std::shared_future<TrainedCandidate> model_future;
  std::shared_future<std::shared_ptr<const training_internal::VortexCdfFit>> cdf_future;
  bool base_ready = false;
  bool order_ready = false;
  {
    std::lock_guard<std::mutex> lock(cache->mutex);
    auto model_it = cache->future_by_key.find(
        selector_training_cache_key(context, config, options));
    if (model_it != cache->future_by_key.end()) {
      model_future = model_it->second;
    }

    std::string base_key = selector_training_base_cache_key(
        context,
        config,
        options.seed,
        options.use_full_dataset_for_assignments,
        options.assignment_sample_limit,
        selector_training_base_includes_centroid_graph(options, config));
    base_ready = cache->training_base_by_key.find(base_key) !=
                 cache->training_base_by_key.end();
    order_ready = cache->training_order_by_key.find(
                      selector_training_order_cache_key(base_key, config)) !=
                  cache->training_order_by_key.end();
    auto cdf_it = cache->training_cdf_by_key.find(
        selector_training_cdf_cache_key(base_key, config));
    if (cdf_it != cache->training_cdf_by_key.end()) {
      cdf_future = cdf_it->second;
    }
  }

  observation.model = model_future_state(model_future);
  observation.base = base_ready ? TrainingComponentState::ready
                                : TrainingComponentState::missing;
  observation.order = order_ready ? TrainingComponentState::ready
                                  : TrainingComponentState::missing;
  observation.cdf = cdf_component_state(cdf_future);
  observation.fresh_base = observation.base == TrainingComponentState::missing;
  if (observation.base == TrainingComponentState::ready) {
    observation.component_reuse_score += 8;
  }
  if (observation.order == TrainingComponentState::ready) {
    observation.component_reuse_score += 3;
  }
  if (observation.cdf == TrainingComponentState::ready) {
    observation.component_reuse_score += 3;
  } else if (observation.cdf == TrainingComponentState::in_flight) {
    observation.component_reuse_score += 1;
  }
  return observation;
}

bool should_skip_progressive_k_candidate(const SelectorCandidateConfig& config,
                                         const SelectorOptions& options,
                                         const NswSelectorContext& context,
                                         TrainCache* cache) {
  if (cache == nullptr ||
      options.dataset_count_hint < 500000 ||
      options.K_values.size() <= 2 ||
      training_cache_state(config, options, context, cache) != TrainingCacheState::missing) {
    return false;
  }
  std::size_t k_idx = nearest_k_index(options.K_values, config.K);
  if (k_idx < 2 || k_idx >= options.K_values.size()) {
    return false;
  }

  std::string context_key = selector_context_key(context);
  std::lock_guard<std::mutex> lock(cache->mutex);
  const auto& observations = options.optimize_for_recall
      ? cache->best_recall_by_context_k
      : cache->best_quality_by_context_k;
  double best_below = -1.0;
  double nearest_lower = -1.0;
  for (std::size_t i = 0; i < k_idx; ++i) {
    auto found = observations.find(context_k_observation_key(context_key,
                                                             options.K_values[i]));
    if (found == observations.end()) {
      continue;
    }
    best_below = std::max(best_below, found->second);
    nearest_lower = found->second;
  }
  if (best_below < 0.0 || nearest_lower < 0.0) {
    return false;
  }
  double drop_tolerance =
      options.optimize_for_recall
          ? std::max(0.030, options.min_recall_improvement * 12.0)
          : std::max(0.020, options.min_quality_improvement * 8.0);
  return nearest_lower + drop_tolerance < best_below;
}

bool is_coarse_exploration_phase(const PhaseLimits& phase) {
  return phase.name == "phase1" || phase.name == "phase1_probe" ||
         phase.name == "phase1_beam";
}

double estimate_training_guard_seconds(const SelectorCandidateConfig& config,
                                       const EvalDataset& eval,
                                       TrainCache* cache) {
  std::size_t train_count = eval.base.count;
  if (cache != nullptr && cache->dataset != nullptr && cache->dataset->count > 0) {
    train_count = cache->dataset->count;
  }

  double data_scale = std::max(1.0, static_cast<double>(train_count) / 1'000'000.0);
  double k_scale = std::max(1.0, static_cast<double>(std::max<uint32_t>(config.K, 1)) / 1024.0);
  double guard = 30.0 * data_scale * k_scale;
  if (config.enable_2opt) {
    guard *= 1.0 + 0.02 * static_cast<double>(config.two_opt_iterations);
  }
  if (train_count >= 500'000) {
    if (config.K >= 8192) {
      guard = std::max(guard, 1800.0);
    } else if (config.K >= 4096) {
      guard = std::max(guard, 1200.0);
    } else if (config.K >= 2048) {
      guard = std::max(guard, 600.0);
    } else {
      guard = std::max(guard, 120.0);
    }
  }

  double observed = observed_train_seconds(cache);
  if (observed > 0.0) {
    guard = std::max(guard, observed * 1.5);
  }
  return guard;
}

double estimate_eval_guard_seconds(const EvalDataset& eval,
                                   const PhaseLimits& phase,
                                   TrainCache* cache) {
  if (is_coarse_exploration_phase(phase)) {
    return 0.0;
  }

  double eval_scale =
      static_cast<double>(eval.base.count) * static_cast<double>(eval.query.count) / 30'000'000.0;
  double static_guard = std::min(180.0, 15.0 * std::max(1.0, eval_scale));
  double observed_per_vector = observed_eval_hash_seconds_per_vector(cache);
  if (observed_per_vector <= 0.0 || eval.base.count == 0) {
    return static_guard;
  }

  double observed_hash_guard = observed_per_vector * static_cast<double>(eval.base.count) * 4.0;
  double query_guard = std::min(30.0, static_cast<double>(eval.query.count) * 0.01);
  double observed_guard = std::max(5.0, observed_hash_guard + query_guard + 5.0);
  return std::min(static_guard, observed_guard);
}

void record_eval_observation(TrainCache* cache,
                             const EvalDataset& eval,
                             const SelectorCandidateMetrics& metrics) {
  if (cache == nullptr || !metrics.ok) {
    return;
  }
  std::lock_guard<std::mutex> lock(cache->mutex);
  if (eval.base.count > 0 && metrics.eval_hash_base_ms > 0.0) {
    double seconds_per_vector =
        (metrics.eval_hash_base_ms / 1000.0) / static_cast<double>(eval.base.count);
    cache->max_observed_eval_hash_seconds_per_vector =
        std::max(cache->max_observed_eval_hash_seconds_per_vector, seconds_per_vector);
  }
  std::string key = context_k_observation_key(metrics.nsw_path, metrics.config.K);
  auto found = cache->best_recall_by_context_k.find(key);
  if (found == cache->best_recall_by_context_k.end()) {
    cache->best_recall_by_context_k.emplace(std::move(key), metrics.recall_at_k_in_window);
  } else {
    found->second = std::max(found->second, metrics.recall_at_k_in_window);
  }
  std::string quality_key = context_k_observation_key(metrics.nsw_path, metrics.config.K);
  auto quality_found = cache->best_quality_by_context_k.find(quality_key);
  double quality = selector_quality_score(metrics);
  if (quality_found == cache->best_quality_by_context_k.end()) {
    cache->best_quality_by_context_k.emplace(std::move(quality_key), quality);
  } else {
    quality_found->second = std::max(quality_found->second, quality);
  }
}

bool is_calibration_phase(const PhaseLimits& phase) {
  return phase.name == "final_calibration" ||
         phase.name == "final_calibration_screen";
}

CandidateLaunchBudget estimate_candidate_launch_budget(const SelectorCandidateConfig& config,
                                                       const SelectorOptions& options,
                                                       const NswSelectorContext& context,
                                                       const EvalDataset& eval,
                                                       const PhaseLimits& phase,
                                                       TrainCache* cache) {
  double eval_guard = estimate_eval_guard_seconds(eval, phase, cache);
  TrainingCacheState state = training_cache_state(config, options, context, cache);
  if (phase.name == "final_calibration" && state == TrainingCacheState::ready) {
    return CandidateLaunchBudget{0.0, false};
  }
  switch (state) {
  case TrainingCacheState::ready:
    return CandidateLaunchBudget{eval_guard, false};
  case TrainingCacheState::in_flight:
    return CandidateLaunchBudget{
        std::min(observed_train_seconds(cache), 300.0) + eval_guard,
        !is_calibration_phase(phase)};
  case TrainingCacheState::missing:
    break;
  }
  double guard = estimate_training_guard_seconds(config, eval, cache) + eval_guard;
  return CandidateLaunchBudget{std::min(guard, 2400.0), !is_calibration_phase(phase)};
}

CandidateCachePartition partition_candidate_cache_states(
    const std::vector<SelectorCandidateConfig>& candidates,
    const NswSelectorContext& context,
    const SelectorOptions& options,
    TrainCache* cache) {
  CandidateCachePartition partition;
  partition.ready.reserve(candidates.size());
  partition.in_flight.reserve(candidates.size());
  partition.missing.reserve(candidates.size());
  std::vector<CandidateCacheObservation> observations;
  observations.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    observations.push_back(
        observe_candidate_cache(candidates[i], options, context, cache));
    switch (observations.back().model) {
    case TrainingCacheState::ready:
      partition.ready.push_back(i);
      break;
    case TrainingCacheState::in_flight:
      partition.in_flight.push_back(i);
      break;
    case TrainingCacheState::missing:
      partition.missing.push_back(i);
      if (observations.back().component_reuse_score > 0) {
        partition.reuse_prioritized_candidates += 1;
      }
      if (observations.back().fresh_base) {
        partition.fresh_base_candidates += 1;
      }
      break;
    }
  }
  std::stable_sort(partition.missing.begin(),
                   partition.missing.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     const auto& a = observations[lhs];
                     const auto& b = observations[rhs];
                     if (a.component_reuse_score != b.component_reuse_score) {
                       return a.component_reuse_score > b.component_reuse_score;
                     }
                     if (a.fresh_base != b.fresh_base) {
                       return !a.fresh_base;
                     }
                     return false;
                   });
  return partition;
}

void record_scheduler_partition(TrainCache* cache,
                                const CandidateCachePartition& partition,
                                bool single_admission_batch) {
  if (cache == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(cache->mutex);
  cache->scheduler_ready_cached_candidates += static_cast<uint32_t>(
      std::min<std::size_t>(partition.ready.size(), std::numeric_limits<uint32_t>::max()));
  cache->scheduler_in_flight_candidates += static_cast<uint32_t>(
      std::min<std::size_t>(partition.in_flight.size(), std::numeric_limits<uint32_t>::max()));
  cache->scheduler_missing_candidates += static_cast<uint32_t>(
      std::min<std::size_t>(partition.missing.size(), std::numeric_limits<uint32_t>::max()));
  cache->scheduler_reuse_prioritized_candidates +=
      partition.reuse_prioritized_candidates;
  cache->scheduler_fresh_base_candidates += partition.fresh_base_candidates;
  if (single_admission_batch) {
    cache->scheduler_single_admission_batches += 1;
  }
}

std::string classify_candidate_error_stage(const std::string& error) {
  if (error.empty()) {
    return {};
  }
  if (error.find("remaining-time guard") != std::string::npos ||
      error.find("progressive-K guard") != std::string::npos ||
      error.find("time budget") != std::string::npos ||
      error.find("budget reached") != std::string::npos) {
    return "launch_guard";
  }
  if (error.find("CDF training failed") != std::string::npos ||
      error.find("cdf_model_spec") != std::string::npos ||
      error.find("CDF branching") != std::string::npos ||
      error.find("Not enough centroid assignments") != std::string::npos) {
    return "cdf_fit";
  }
  if (error.find("clustering") != std::string::npos ||
      error.find("Cluster count") != std::string::npos ||
      error.find("Skeleton size") != std::string::npos ||
      error.find("NSW skeleton") != std::string::npos ||
      error.find("CSR") != std::string::npos ||
      error.find("Dataset") != std::string::npos) {
    return "training_base";
  }
  if (error.find("Centroid order") != std::string::npos ||
      error.find("Centroid ordering") != std::string::npos) {
    return "centroid_order";
  }
  if (error.find("Selector eval") != std::string::npos ||
      error.find("evaluation") != std::string::npos) {
    return "evaluation";
  }
  return "unknown";
}

} // namespace

SelectorCandidateMetrics evaluate_candidate(
    const SelectorCandidateConfig& config,
    const SelectorOptions& options,
    uint32_t eval_threads,
    const NswSelectorContext& context,
    const EvalDataset& eval,
    const PhaseLimits& phase,
    TrainCache* cache,
    SearchController* search) {
  SelectorCandidateMetrics metrics;
  metrics.config = config;
  metrics.phase = phase.name;
  metrics.nsw_path = context.display_nsw_path;
  metrics.skeleton_node_count = static_cast<uint32_t>(context.csr.node_ids.size());
  metrics.skeleton_layer = context.csr.layer;
  if (should_skip_progressive_k_candidate(config, options, context, cache)) {
    metrics.ok = false;
    metrics.error =
        "selector progressive-K guard skipped high-K candidate after lower-K probes showed no recall gain";
    metrics.error_stage = classify_candidate_error_stage(metrics.error);
    return metrics;
  }
  if (search != nullptr && !search->can_launch_more_work()) {
    metrics.ok = false;
    metrics.error = "selector search time budget reached";
    metrics.error_stage = classify_candidate_error_stage(metrics.error);
    return metrics;
  }
  CandidateLaunchBudget launch_budget =
      estimate_candidate_launch_budget(config, options, context, eval, phase, cache);
  double reserve_seconds = launch_budget.apply_final_calibration_reserve && search != nullptr
      ? search->final_calibration_reserve_seconds
      : 0.0;
  if (search != nullptr &&
      !search->can_launch_work_with_min_seconds(launch_budget.guard_seconds,
                                                reserve_seconds)) {
    metrics.ok = false;
    metrics.error =
        "selector remaining-time guard prevented launching expensive candidate";
    metrics.error_stage = classify_candidate_error_stage(metrics.error);
    return metrics;
  }
  try {
    const TrainedCandidate& trained =
        get_or_train_candidate(config, options, context, cache, search);
    metrics = evaluate_model_metrics(config,
                                     trained,
                                     eval,
                                     phase,
                                     eval_threads,
                                     options.recall_target,
                                     cache,
                                     &context);
    metrics.nsw_path = context.display_nsw_path;
    metrics.skeleton_node_count = static_cast<uint32_t>(context.csr.node_ids.size());
    metrics.skeleton_layer = context.csr.layer;
    record_eval_observation(cache, eval, metrics);
  } catch (const std::exception& ex) {
    metrics.ok = false;
    metrics.error = ex.what();
    metrics.error_stage = classify_candidate_error_stage(metrics.error);
    metrics.phase = phase.name;
  }
  return metrics;
}

uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs + rhs;
}

uint64_t byte_count(std::size_t count, std::size_t item_size) {
  if (item_size != 0 && count > std::numeric_limits<uint64_t>::max() / item_size) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(count) * static_cast<uint64_t>(item_size);
}

uint64_t vector_storage_bytes(const vector_io::VectorStorage<float>& storage) {
  return byte_count(storage.values.size(), sizeof(float));
}

uint64_t csr_bytes(const NswCsr& csr) {
  uint64_t total = byte_count(csr.node_ids.size(), sizeof(uint64_t));
  total = saturating_add(total, byte_count(csr.offsets.size(), sizeof(uint64_t)));
  total = saturating_add(total, byte_count(csr.neighbors.size(), sizeof(uint32_t)));
  return total;
}

uint64_t estimate_candidate_working_set_bytes(const NswSelectorContext& context,
                                              const EvalDataset& eval,
                                              const TrainCache* cache) {
  uint64_t total = 64ULL * 1024ULL * 1024ULL;
  total = saturating_add(total, vector_storage_bytes(eval.base));
  total = saturating_add(total, vector_storage_bytes(eval.query));
  total = saturating_add(total, csr_bytes(context.csr));
  if (cache != nullptr && cache->shared_dataset_bytes > 0) {
    // Training builds candidate-local structures over the dataset even though
    // the immutable vector storage itself is shared by the selector.
    total = saturating_add(total, cache->shared_dataset_bytes);
  }
  return std::max<uint64_t>(total, 1);
}

void record_resource_plan(TrainCache* cache,
                          const EvalResourcePlan& plan,
                          bool memory_limited) {
  if (cache == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(cache->mutex);
  cache->peak_parallelism = std::max(cache->peak_parallelism, plan.parallelism);
  cache->peak_threads_per_candidate =
      std::max(cache->peak_threads_per_candidate, plan.threads_per_candidate);
  cache->memory_budget_limited = cache->memory_budget_limited || memory_limited;
}

EvalResourcePlan compute_eval_resource_plan(const SelectorOptions& options,
                                            const NswSelectorContext& context,
                                            const EvalDataset& eval,
                                            TrainCache* cache,
                                            std::size_t candidate_count) {
  EvalResourcePlan plan;
  if (candidate_count == 0) {
    return plan;
  }
  uint32_t hardware_threads = std::thread::hardware_concurrency();
  if (hardware_threads == 0) {
    hardware_threads = 1;
  }

  uint32_t requested_parallelism = options.selector_parallelism;
  uint32_t requested_threads = std::max<uint32_t>(1, options.threads);
  bool auto_threads = requested_parallelism == 0 && requested_threads == 1;
  if (requested_parallelism > 0) {
    plan.parallelism = std::min<uint32_t>(requested_parallelism,
                                          static_cast<uint32_t>(candidate_count));
    plan.threads_per_candidate = requested_threads;
  } else if (requested_threads > 1) {
    uint32_t auto_parallelism = hardware_threads / requested_threads;
    if (auto_parallelism == 0) {
      auto_parallelism = 1;
    }
    plan.parallelism =
        std::min<uint32_t>(auto_parallelism, static_cast<uint32_t>(candidate_count));
    if (plan.parallelism == 0) {
      plan.parallelism = 1;
    }
    plan.threads_per_candidate = requested_threads;
  } else {
    plan.parallelism =
        std::min<uint32_t>(hardware_threads, static_cast<uint32_t>(candidate_count));
    if (plan.parallelism == 0) {
      plan.parallelism = 1;
    }
    plan.threads_per_candidate = 1;
  }

  bool memory_limited = false;
  uint64_t memory_budget = cache != nullptr ? cache->memory_budget_bytes : 0;
  if (memory_budget == 0) {
    memory_budget = options.selector_memory_budget_bytes;
  }
  if (memory_budget > 0) {
    uint64_t shared_bytes = cache != nullptr ? cache->shared_dataset_bytes : 0;
    uint64_t available_bytes = memory_budget > shared_bytes
        ? memory_budget - shared_bytes
        : memory_budget / 2;
    uint64_t per_candidate_bytes =
        estimate_candidate_working_set_bytes(context, eval, cache);
    uint64_t memory_parallelism = available_bytes / per_candidate_bytes;
    if (memory_parallelism == 0) {
      memory_parallelism = 1;
    }
    if (memory_parallelism < plan.parallelism) {
      plan.parallelism = static_cast<uint32_t>(
          std::min<uint64_t>(memory_parallelism, std::numeric_limits<uint32_t>::max()));
      memory_limited = true;
    }
  }

  if (auto_threads) {
    plan.threads_per_candidate =
        std::max<uint32_t>(1, hardware_threads / plan.parallelism);
  }
  plan.memory_limited = memory_limited;
  return plan;
}

std::vector<SelectorCandidateMetrics> evaluate_candidates(
    const std::vector<SelectorCandidateConfig>& candidates,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    const EvalDataset& eval,
    const PhaseLimits& phase,
    TrainCache* cache,
    SearchController* search) {
  std::vector<SelectorCandidateMetrics> metrics(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    metrics[i].config = candidates[i];
    metrics[i].phase = phase.name;
    metrics[i].nsw_path = context.display_nsw_path;
    metrics[i].skeleton_node_count = static_cast<uint32_t>(context.csr.node_ids.size());
    metrics[i].skeleton_layer = context.csr.layer;
  }

  auto mark_time_budget = [&](std::size_t from_index) {
    for (std::size_t i = from_index; i < metrics.size(); ++i) {
      metrics[i].ok = false;
      if (metrics[i].error.empty()) {
        metrics[i].error = "selector search time budget reached";
      }
      metrics[i].error_stage = classify_candidate_error_stage(metrics[i].error);
    }
  };

  if (search != nullptr && !search->can_launch_more_work()) {
    mark_time_budget(0);
    return metrics;
  }
  prepare_eval_base_rank_cache_for_phase(cache, phase);
  prewarm_final_calibration_nearest_caches(candidates,
                                           options,
                                           context,
                                           eval,
                                           phase,
                                           cache,
                                           search);

  CandidateCachePartition partition =
      partition_candidate_cache_states(candidates, context, options, cache);
  uint32_t cached_candidates = static_cast<uint32_t>(
      std::min<std::size_t>(partition.ready.size() + partition.in_flight.size(),
                            std::numeric_limits<uint32_t>::max()));
  uint32_t uncached_candidates = static_cast<uint32_t>(
      std::min<std::size_t>(partition.missing.size(),
                            std::numeric_limits<uint32_t>::max()));
  bool single_admission_batch =
      options.selector_parallelism == 0 && uncached_candidates > 0 && candidates.size() > 1;
  record_scheduler_partition(cache, partition, single_admission_batch);

  auto mark_time_budget_indices = [&](const std::vector<std::size_t>& indices,
                                      std::size_t from_position) {
    for (std::size_t pos = from_position; pos < indices.size(); ++pos) {
      std::size_t index = indices[pos];
      metrics[index].ok = false;
      if (metrics[index].error.empty()) {
        metrics[index].error = "selector search time budget reached";
      }
      metrics[index].error_stage = classify_candidate_error_stage(metrics[index].error);
    }
  };

  auto evaluate_index_list = [&](const std::vector<std::size_t>& indices,
                                 uint32_t list_parallelism,
                                 uint32_t list_eval_threads) {
    if (indices.empty()) {
      return;
    }
    if (search != nullptr && !search->can_launch_more_work()) {
      mark_time_budget_indices(indices, 0);
      return;
    }
    if (list_parallelism <= 1 || indices.size() <= 1) {
      for (std::size_t pos = 0; pos < indices.size(); ++pos) {
        if (search != nullptr && !search->can_launch_more_work()) {
          mark_time_budget_indices(indices, pos);
          break;
        }
        std::size_t index = indices[pos];
        metrics[index] = evaluate_candidate(candidates[index],
                                            options,
                                            list_eval_threads,
                                            context,
                                            eval,
                                            phase,
                                            cache,
                                            search);
      }
      return;
    }

    std::atomic<std::size_t> next_position{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;
    workers.reserve(list_parallelism);
    for (uint32_t worker = 0; worker < list_parallelism; ++worker) {
      workers.emplace_back([&]() {
        while (true) {
          if (stop.load(std::memory_order_relaxed)) {
            break;
          }
          if (search != nullptr && !search->can_launch_more_work()) {
            stop.store(true, std::memory_order_relaxed);
            break;
          }
          std::size_t pos = next_position.fetch_add(1);
          if (pos >= indices.size()) {
            break;
          }
          std::size_t index = indices[pos];
          metrics[index] = evaluate_candidate(candidates[index],
                                              options,
                                              list_eval_threads,
                                              context,
                                              eval,
                                              phase,
                                              cache,
                                              search);
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    if (search != nullptr && !search->can_launch_more_work()) {
      std::size_t done = std::min<std::size_t>(
          next_position.load(std::memory_order_relaxed), indices.size());
      mark_time_budget_indices(indices, done);
    }
  };

  if (single_admission_batch) {
    std::vector<std::size_t> cached_indices = partition.ready;
    cached_indices.insert(cached_indices.end(),
                          partition.in_flight.begin(),
                          partition.in_flight.end());
    auto evaluate_missing = [&]() {
      if (partition.missing.empty()) {
        return;
      }
      EvalResourcePlan missing_plan =
          compute_eval_resource_plan(options, context, eval, cache, 1);
      missing_plan.parallelism = 1;
      record_resource_plan(cache, missing_plan, missing_plan.memory_limited);
      evaluate_index_list(partition.missing, 1, missing_plan.threads_per_candidate);
    };

    bool can_overlap_single_admission =
        !cached_indices.empty() && !partition.missing.empty() &&
        partition.in_flight.empty();
    std::thread missing_worker;
    if (can_overlap_single_admission) {
      // Admit at most one fresh training stream while cached evaluations keep
      // the remaining cores busy. Avoid overlapping when prior in-flight
      // training exists because that would simply recreate lane wait.
      missing_worker = std::thread(evaluate_missing);
    }
    if (!cached_indices.empty()) {
      EvalResourcePlan cached_plan =
          compute_eval_resource_plan(options, context, eval, cache, cached_indices.size());
      if (options.threads == 1 && options.selector_parallelism == 0 &&
          cached_indices.size() > 1) {
        uint32_t hardware_threads = std::thread::hardware_concurrency();
        if (hardware_threads == 0) {
          hardware_threads = 1;
        }
        uint32_t useful_eval_threads = eval_worker_count(eval.base.count, hardware_threads);
        if (useful_eval_threads > cached_plan.threads_per_candidate) {
          uint32_t useful_parallelism =
              std::max<uint32_t>(1, hardware_threads / useful_eval_threads);
          useful_parallelism =
              std::min<uint32_t>(useful_parallelism,
                                 static_cast<uint32_t>(cached_indices.size()));
          if (useful_parallelism > 0 && useful_parallelism < cached_plan.parallelism) {
            cached_plan.parallelism = useful_parallelism;
            cached_plan.threads_per_candidate =
                std::max<uint32_t>(useful_eval_threads,
                                   hardware_threads / cached_plan.parallelism);
          }
        }
      }
      record_resource_plan(cache, cached_plan, cached_plan.memory_limited);
      evaluate_index_list(cached_indices,
                          cached_plan.parallelism,
                          cached_plan.threads_per_candidate);
    }
    if (missing_worker.joinable()) {
      missing_worker.join();
    } else {
      evaluate_missing();
    }
    return metrics;
  }

  EvalResourcePlan resource_plan =
      compute_eval_resource_plan(options, context, eval, cache, candidates.size());
  uint32_t parallelism = resource_plan.parallelism;
  if (options.selector_parallelism == 0 && uncached_candidates > 0 && parallelism > 2) {
    // Candidate training is intentionally serialized in get_or_train_candidate()
    // to avoid nested FAISS/OpenMP contention. Extra workers beyond cached
    // evaluations plus one training lane mostly park on the training mutex.
    uint32_t useful_parallelism =
        std::max<uint32_t>(2, std::min<uint32_t>(parallelism, cached_candidates + 1));
    parallelism = std::min<uint32_t>(parallelism, useful_parallelism);
    resource_plan.parallelism = parallelism;
    if (options.threads == 1) {
      uint32_t hardware_threads = std::thread::hardware_concurrency();
      if (hardware_threads == 0) {
        hardware_threads = 1;
      }
      resource_plan.threads_per_candidate =
          std::max<uint32_t>(1, hardware_threads / parallelism);
    }
  }
  if (options.selector_parallelism == 0 && options.threads == 1 &&
      uncached_candidates == 0 && candidates.size() > 1) {
    uint32_t hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads == 0) {
      hardware_threads = 1;
    }
    uint32_t useful_eval_threads = eval_worker_count(eval.base.count, hardware_threads);
    if (useful_eval_threads > resource_plan.threads_per_candidate) {
      uint32_t useful_parallelism =
          std::max<uint32_t>(1, hardware_threads / useful_eval_threads);
      useful_parallelism =
          std::min<uint32_t>(useful_parallelism, static_cast<uint32_t>(candidates.size()));
      if (useful_parallelism > 0 && useful_parallelism < parallelism) {
        parallelism = useful_parallelism;
        resource_plan.parallelism = parallelism;
        resource_plan.threads_per_candidate =
            std::max<uint32_t>(useful_eval_threads, hardware_threads / parallelism);
      }
    }
  }
  record_resource_plan(cache, resource_plan, resource_plan.memory_limited);
  uint32_t eval_threads = resource_plan.threads_per_candidate;
  if (parallelism <= 1 || candidates.size() <= 1) {
    std::vector<std::size_t> all_indices(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      all_indices[i] = i;
    }
    evaluate_index_list(all_indices, 1, eval_threads);
    return metrics;
  }

  std::atomic<std::size_t> next_index{0};
  std::atomic<bool> stop{false};
  std::vector<std::thread> workers;
  workers.reserve(parallelism);
  for (uint32_t worker = 0; worker < parallelism; ++worker) {
    workers.emplace_back([&]() {
      while (true) {
        if (stop.load(std::memory_order_relaxed)) {
          break;
        }
        if (search != nullptr && !search->can_launch_more_work()) {
          stop.store(true, std::memory_order_relaxed);
          break;
        }
        std::size_t index = next_index.fetch_add(1);
        if (index >= candidates.size()) {
          break;
        }
        metrics[index] = evaluate_candidate(candidates[index],
                                            options,
                                            eval_threads,
                                            context,
                                            eval,
                                            phase,
                                            cache,
                                            search);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  if (search != nullptr && !search->can_launch_more_work()) {
    std::size_t done = std::min<std::size_t>(next_index.load(std::memory_order_relaxed), candidates.size());
    mark_time_budget(done);
  }
  return metrics;
}

std::vector<std::size_t> evenly_spaced_indices(std::size_t count, std::size_t keep) {
  std::vector<std::size_t> out;
  if (count == 0 || keep == 0) {
    return out;
  }
  if (keep >= count) {
    return all_indices(count);
  }
  if (keep == 1) {
    out.push_back(0);
    return out;
  }
  out.reserve(keep);
  for (std::size_t i = 0; i < keep; ++i) {
    double pos = static_cast<double>(i) * static_cast<double>(count - 1) /
                 static_cast<double>(keep - 1);
    std::size_t idx = static_cast<std::size_t>(std::llround(pos));
    idx = std::min<std::size_t>(idx, count - 1);
    if (out.empty() || out.back() != idx) {
      out.push_back(idx);
    }
  }
  while (out.size() < keep) {
    std::size_t next = out.empty() ? 0 : std::min<std::size_t>(count - 1, out.back() + 1);
    if (!out.empty() && next == out.back()) {
      break;
    }
    out.push_back(next);
  }
  return out;
}

} // namespace vortex::selector_internal
