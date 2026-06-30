#include "selector_eval_internal.h"

#include "training_internal.h"

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
#include <utility>
#include <vector>

namespace vortex::selector_internal {

TrainedCandidate train_candidate(const SelectorCandidateConfig& config,
                                 const SelectorOptions& options,
                                 const NswSelectorContext& context,
                                 TrainCache* cache) {
  TrainOptions train_opts;
  train_opts.dataset_path = options.dataset_path;
  train_opts.nsw_path = context.nsw_path;
  train_opts.K = config.K;
  train_opts.hash_bits = options.hash_bits;
  train_opts.cdf_model_spec = config.cdf_model_spec;
  train_opts.cdf_branching_factor = config.cdf_branching_factor;
  train_opts.centroid_knn = config.centroid_knn;
  train_opts.enable_2opt = config.enable_2opt;
  train_opts.two_opt_iterations = config.two_opt_iterations;
  train_opts.seed = options.seed;
  train_opts.threads = options.threads;
  train_opts.use_full_dataset_for_assignments = options.use_full_dataset_for_assignments;
  train_opts.assignment_sample_limit = options.assignment_sample_limit;
  train_opts.enable_graph_centroid_order = config.enable_graph_centroid_order;

  auto get_training_base = [&](
      const std::string& key,
      const TrainOptions& base_train_opts,
      training_internal::VortexTrainingProfile* profile)
      -> std::shared_ptr<const training_internal::VortexTrainingBase> {
    if (cache == nullptr || cache->dataset == nullptr) {
      return {};
    }
    auto wait_start = SelectorClock::now();
    {
      std::lock_guard<std::mutex> lock(cache->mutex);
      auto found = cache->training_base_by_key.find(key);
      if (found != cache->training_base_by_key.end()) {
        double wait_ms = elapsed_ms(wait_start);
        cache->training_base_cache_hits += 1;
        cache->training_base_cache_wait_ms += wait_ms;
        if (profile != nullptr) {
          profile->base_cache_hit = true;
          profile->base_cache_wait_ms += wait_ms;
        }
        return found->second;
      }
    }

    training_internal::VortexTrainingProfile build_profile;
    auto base = training_internal::build_vortex_training_base(
        base_train_opts, *cache->dataset, context.csr, &build_profile);
    if (profile != nullptr) {
      profile->base_build_ms += build_profile.base_build_ms;
      profile->gather_skeleton_ms += build_profile.gather_skeleton_ms;
      profile->cluster_assign_ms += build_profile.cluster_assign_ms;
      profile->assign_centroids_ms += build_profile.assign_centroids_ms;
    }
    {
      std::lock_guard<std::mutex> lock(cache->mutex);
      auto [inserted_it, inserted] = cache->training_base_by_key.emplace(key, base);
      if (inserted) {
        cache->training_base_cache_misses += 1;
        cache->training_base_cache_build_ms += build_profile.base_build_ms;
        cache->training_base_cache_memory_bytes =
            saturating_add_u64(cache->training_base_cache_memory_bytes, base->memory_bytes());
      } else {
        base = inserted_it->second;
        cache->training_base_cache_hits += 1;
      }
    }
    return base;
  };

  auto get_training_order = [&](
      const std::string& key,
      const training_internal::VortexTrainingBase& base,
      training_internal::VortexTrainingProfile* profile)
      -> std::shared_ptr<const std::vector<uint32_t>> {
    if (cache == nullptr) {
      return {};
    }
    auto wait_start = SelectorClock::now();
    {
      std::lock_guard<std::mutex> lock(cache->mutex);
      auto found = cache->training_order_by_key.find(key);
      if (found != cache->training_order_by_key.end()) {
        double wait_ms = elapsed_ms(wait_start);
        cache->training_order_cache_hits += 1;
        cache->training_order_cache_wait_ms += wait_ms;
        if (profile != nullptr) {
          profile->order_cache_hit = true;
          profile->order_cache_wait_ms += wait_ms;
        }
        return found->second;
      }
    }

    training_internal::VortexTrainingProfile order_profile;
    std::vector<uint32_t> built_order =
        training_internal::build_vortex_centroid_order(train_opts, base, &order_profile);
    std::shared_ptr<const std::vector<uint32_t>> order =
        std::make_shared<const std::vector<uint32_t>>(std::move(built_order));
    if (profile != nullptr) {
      profile->order_build_ms += order_profile.order_build_ms;
      profile->centroid_order_ms += order_profile.centroid_order_ms;
    }
    {
      std::lock_guard<std::mutex> lock(cache->mutex);
      auto [inserted_it, inserted] = cache->training_order_by_key.emplace(key, order);
      if (inserted) {
        cache->training_order_cache_misses += 1;
        cache->training_order_cache_build_ms += order_profile.order_build_ms;
        cache->training_order_cache_memory_bytes =
            saturating_add_u64(cache->training_order_cache_memory_bytes,
                               byte_count_u64(order->size(), sizeof(uint32_t)));
      } else {
        order = inserted_it->second;
        cache->training_order_cache_hits += 1;
      }
    }
    return order;
  };

  auto get_training_cdf = [&](
      const std::string& key,
      const training_internal::VortexTrainingBase& base,
      training_internal::VortexTrainingProfile* profile)
      -> std::shared_ptr<const training_internal::VortexCdfFit> {
    if (cache == nullptr) {
      return {};
    }

    auto wait_start = SelectorClock::now();
    std::shared_future<std::shared_ptr<const training_internal::VortexCdfFit>> cdf_future;
    std::optional<std::promise<std::shared_ptr<const training_internal::VortexCdfFit>>>
        build_promise;
    {
      std::lock_guard<std::mutex> lock(cache->mutex);
      auto found = cache->training_cdf_by_key.find(key);
      if (found != cache->training_cdf_by_key.end()) {
        cdf_future = found->second;
        cache->training_cdf_cache_hits += 1;
      } else {
        build_promise.emplace();
        cdf_future = build_promise->get_future().share();
        cache->training_cdf_by_key.emplace(key, cdf_future);
        cache->training_cdf_cache_misses += 1;
      }
    }
    if (!build_promise.has_value()) {
      try {
        auto cdf = cdf_future.get();
        double wait_ms = elapsed_ms(wait_start);
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->training_cdf_cache_wait_ms += wait_ms;
        if (profile != nullptr) {
          profile->cdf_cache_hit = true;
          profile->cdf_cache_wait_ms += wait_ms;
        }
        return cdf;
      } catch (...) {
        double wait_ms = elapsed_ms(wait_start);
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->training_cdf_cache_wait_ms += wait_ms;
        if (profile != nullptr) {
          profile->cdf_cache_hit = true;
          profile->cdf_cache_wait_ms += wait_ms;
        }
        throw;
      }
    }

    {
      auto build_start = SelectorClock::now();
      try {
        training_internal::VortexTrainingProfile cdf_profile;
        auto cdf = training_internal::fit_vortex_cdf_models(
            train_opts, base, &cdf_profile);
        double build_ms = elapsed_ms(build_start);
        if (profile != nullptr) {
          profile->cdf_cache_build_ms += build_ms;
          profile->cdf_fit_ms += cdf_profile.cdf_fit_ms;
        }
        {
          std::lock_guard<std::mutex> lock(cache->mutex);
          cache->training_cdf_cache_build_ms += build_ms;
          cache->training_cdf_cache_memory_bytes =
              saturating_add_u64(cache->training_cdf_cache_memory_bytes,
                                 cdf->memory_bytes());
        }
        build_promise->set_value(cdf);
        return cdf;
      } catch (...) {
        double build_ms = elapsed_ms(build_start);
        auto exception = std::current_exception();
        {
          std::lock_guard<std::mutex> lock(cache->mutex);
          cache->training_cdf_cache_build_ms += build_ms;
          cache->training_cdf_cache_failures += 1;
        }
        if (profile != nullptr) {
          profile->cdf_cache_build_ms += build_ms;
        }
        build_promise->set_exception(exception);
        throw;
      }
    }
  };

  auto train_start = std::chrono::steady_clock::now();
  VortexModel model;
  training_internal::VortexTrainingProfile train_profile;
  if (cache != nullptr && cache->dataset != nullptr) {
    const bool include_centroid_graph =
        selector_training_base_includes_centroid_graph(options, config);
    std::string base_key = selector_training_base_cache_key(
        context,
        config,
        options.seed,
        options.use_full_dataset_for_assignments,
        options.assignment_sample_limit,
        include_centroid_graph);
    TrainOptions base_train_opts = train_opts;
    base_train_opts.enable_graph_centroid_order = include_centroid_graph;
    auto base = get_training_base(base_key, base_train_opts, &train_profile);
    auto order = get_training_order(selector_training_order_cache_key(base_key, config),
                                    *base,
                                    &train_profile);
    auto cdf = get_training_cdf(selector_training_cdf_cache_key(base_key, config),
                                *base,
                                &train_profile);
    model = training_internal::train_vortex_from_training_base(
        train_opts, *base, order.get(), cdf.get(), &train_profile);
  } else {
    model = train_vortex(train_opts);
  }
  auto train_end = std::chrono::steady_clock::now();
  double train_ms = static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start).count()) /
      1000.0;

  TrainedCandidate candidate;
  candidate.model = std::move(model);
  candidate.train_time_ms = train_ms;
  candidate.train_profile = train_profile;
  candidate.model_memory_bytes = estimate_model_memory_bytes(candidate.model);
  candidate.model_size_bytes = serialized_model_size_bytes(candidate.model);
  return candidate;
}

const TrainedCandidate& get_or_train_candidate(
    const SelectorCandidateConfig& config,
    const SelectorOptions& options,
    const NswSelectorContext& context,
    TrainCache* cache,
    SearchController* search) {
  if (search != nullptr && !search->can_launch_more_work()) {
    throw std::runtime_error(
        "selector search time budget reached before launching model training");
  }
  std::string key = selector_training_cache_key(context, config, options);
  std::shared_future<TrainedCandidate> candidate_future;
  std::optional<std::promise<TrainedCandidate>> train_promise;
  bool cache_hit = false;
  {
    std::lock_guard<std::mutex> lock(cache->mutex);
    auto found = cache->future_by_key.find(key);
    if (found != cache->future_by_key.end()) {
      candidate_future = found->second;
      cache->cache_hits += 1;
      cache_hit = true;
    } else {
      if (cache->max_model_trains > 0 &&
          cache->model_trains_started >= cache->max_model_trains) {
        cache->model_train_budget_hit = true;
        throw std::runtime_error(
            "selector model-train budget reached; reduce search scope or increase --max-model-trains");
      }
      if (cache->context_model_train_limit > 0) {
        uint32_t context_started = 0;
        if (cache->model_trains_started >= cache->context_model_train_start) {
          context_started = cache->model_trains_started - cache->context_model_train_start;
        }
        if (context_started >= cache->context_model_train_limit) {
          cache->context_model_train_budget_hit = true;
          throw std::runtime_error(
              "selector context model-train budget reached; increase --max-model-trains "
              "or reduce target_skeleton search scope");
        }
      }
      if (search != nullptr && !search->can_launch_more_work()) {
        throw std::runtime_error(
            "selector search time budget reached before launching model training");
      }
      train_promise.emplace();
      candidate_future = train_promise->get_future().share();
      cache->future_by_key.emplace(key, candidate_future);
      cache->model_trains_started += 1;
      cache->cache_misses += 1;
    }
  }

  if (train_promise.has_value()) {
    try {
      // Training can enter OpenMP/FAISS parallel regions even when the Vortex
      // training option requests one thread. Keep candidate workers from
      // launching concurrent nested training jobs; cached model evaluation can
      // still run in parallel after the model is materialized.
      auto lane_wait_start = SelectorClock::now();
      std::unique_lock<std::mutex> train_lock(cache->training_mutex);
      double lane_wait_ms = elapsed_ms(lane_wait_start);
      {
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->training_lane_wait_ms += lane_wait_ms;
      }
      TrainedCandidate trained = train_candidate(config, options, context, cache);
      {
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->max_observed_train_seconds =
            std::max(cache->max_observed_train_seconds, trained.train_time_ms / 1000.0);
        cache->train_time_ms_sum += trained.train_time_ms;
        cache->train_gather_skeleton_ms_sum += trained.train_profile.gather_skeleton_ms;
        cache->train_cluster_assign_ms_sum += trained.train_profile.cluster_assign_ms;
        cache->train_assign_centroids_ms_sum += trained.train_profile.assign_centroids_ms;
        cache->train_centroid_order_ms_sum += trained.train_profile.centroid_order_ms;
        cache->train_range_alloc_ms_sum += trained.train_profile.range_alloc_ms;
        cache->train_cdf_fit_ms_sum += trained.train_profile.cdf_fit_ms;
        cache->train_model_assembly_ms_sum += trained.train_profile.model_assembly_ms;
      }
      train_promise->set_value(std::move(trained));
    } catch (...) {
      train_promise->set_exception(std::current_exception());
    }
  }

  auto cache_wait_start = SelectorClock::now();
  const TrainedCandidate& trained = candidate_future.get();
  if (cache_hit) {
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->cache_wait_ms += elapsed_ms(cache_wait_start);
  }
  return trained;
}

struct LocalityMetricAccumulator {
  explicit LocalityMetricAccumulator(const EvalDataset& eval_data)
      : eval(eval_data),
        hits_by_window(eval_data.windows.size(), 0),
        same_node_hits(eval_data.node_counts.size(), 0),
        near_1_node_hits(eval_data.node_counts.size(), 0),
        near_2_node_hits(eval_data.node_counts.size(), 0),
        near_4_node_hits(eval_data.node_counts.size(), 0),
        total_node_distance_norm(eval_data.node_counts.size(), 0.0L),
        overlay_match_sum_by_count(eval_data.node_counts.size(), 0.0L),
        overlay_match_weight_by_count(eval_data.node_counts.size(), 0.0L) {
    per_query_recall.reserve(eval.query.count);
    per_query_overlay_match.reserve(eval.query.count);
    rank_distance_norm_samples.reserve(eval.query.count * eval.truth_k);
    auto it = std::find(eval.windows.begin(), eval.windows.end(), eval.window);
    if (it != eval.windows.end()) {
      default_window_index = static_cast<std::size_t>(it - eval.windows.begin());
    }
  }

  void record_query(std::size_t qi, std::size_t pos, const std::vector<std::size_t>& rank_by_index) {
    std::size_t query_neighbors = 0;
    std::size_t default_window_hits = 0;
    long double query_overlay_match_sum = 0.0L;
    long double query_overlay_match_weight = 0.0L;
    faiss::idx_t self_index = eval.self_base_indices[qi];
    for (uint32_t j = 0; j < eval.truth_k; ++j) {
      faiss::idx_t nb = eval.truth_labels[qi * eval.truth_k + j];
      if (nb < 0) {
        continue;
      }
      if (eval.exclude_self && self_index >= 0 && nb == self_index) {
        continue;
      }
      std::size_t nb_index = static_cast<std::size_t>(nb);
      if (nb_index >= eval.base.count) {
        continue;
      }
      query_neighbors += 1;
      total_neighbors += 1;
      double neighbor_weight =
          1.0 / std::log2(static_cast<double>(query_neighbors) + 1.0);
      std::size_t rank = rank_by_index[nb_index];
      uint64_t diff = rank > pos
          ? static_cast<uint64_t>(rank - pos)
          : static_cast<uint64_t>(pos - rank);
      for (std::size_t i = 0; i < eval.windows.size(); ++i) {
        if (diff <= eval.windows[i]) {
          hits_by_window[i] += 1;
          if (i == default_window_index) {
            default_window_hits += 1;
          }
        }
      }
      total_rank_distance += static_cast<long double>(diff);
      if (eval.base.count > 0) {
        rank_distance_norm_samples.push_back(
            static_cast<double>(diff) / static_cast<double>(eval.base.count));
      }
      for (std::size_t i = 0; i < eval.node_counts.size(); ++i) {
        uint32_t node_count = eval.node_counts[i];
        if (node_count < 2 || eval.base.count == 0) {
          continue;
        }
        uint32_t query_node = static_cast<uint32_t>(
            (pos * static_cast<std::size_t>(node_count)) / eval.base.count);
        uint32_t neighbor_node = static_cast<uint32_t>(
            (rank * static_cast<std::size_t>(node_count)) / eval.base.count);
        if (query_node >= node_count) {
          query_node = node_count - 1;
        }
        if (neighbor_node >= node_count) {
          neighbor_node = node_count - 1;
        }
        uint32_t direct_distance = query_node > neighbor_node
            ? query_node - neighbor_node
            : neighbor_node - query_node;
        uint32_t node_distance =
            std::min(direct_distance, node_count - direct_distance);
        if (node_distance == 0) {
          same_node_hits[i] += 1;
        }
        if (node_distance <= 1) {
          near_1_node_hits[i] += 1;
        }
        if (node_distance <= 2) {
          near_2_node_hits[i] += 1;
        }
        if (node_distance <= 4) {
          near_4_node_hits[i] += 1;
        }
        double max_distance = std::max(1.0, static_cast<double>(node_count) * 0.5);
        total_node_distance_norm[i] +=
            static_cast<long double>(static_cast<double>(node_distance) / max_distance);
        double match = 1.0 / (1.0 + static_cast<double>(node_distance));
        overlay_match_sum_by_count[i] +=
            static_cast<long double>(neighbor_weight * match);
        overlay_match_weight_by_count[i] +=
            static_cast<long double>(neighbor_weight);
        query_overlay_match_sum += static_cast<long double>(neighbor_weight * match);
        query_overlay_match_weight += static_cast<long double>(neighbor_weight);
      }
    }
    if (query_neighbors > 0) {
      per_query_recall.push_back(static_cast<double>(default_window_hits) /
                                 static_cast<double>(query_neighbors));
      if (query_overlay_match_weight > 0.0L) {
        per_query_overlay_match.push_back(
            static_cast<double>(query_overlay_match_sum / query_overlay_match_weight));
      }
    }
  }

  void finish(SelectorCandidateMetrics* metrics,
              const std::optional<double>& recall_target) {
    metrics->recall_windows = eval.windows;
    metrics->recall_at_k_by_window.assign(eval.windows.size(), 0.0);
    metrics->node_counts = eval.node_counts;
    metrics->same_node_hit_by_count.assign(eval.node_counts.size(), 0.0);
    metrics->near_1_node_hit_by_count.assign(eval.node_counts.size(), 0.0);
    metrics->near_2_node_hit_by_count.assign(eval.node_counts.size(), 0.0);
    metrics->near_4_node_hit_by_count.assign(eval.node_counts.size(), 0.0);
    metrics->mean_node_distance_norm_by_count.assign(eval.node_counts.size(), 0.0);
    metrics->overlay_match_by_count.assign(eval.node_counts.size(), 0.0);
    if (total_neighbors == 0) {
      metrics->locality_quality_score = 0.0;
      return;
    }

    for (std::size_t i = 0; i < eval.windows.size(); ++i) {
      metrics->recall_at_k_by_window[i] =
          static_cast<double>(hits_by_window[i]) /
          static_cast<double>(total_neighbors);
    }
    if (default_window_index < metrics->recall_at_k_by_window.size()) {
      metrics->recall_at_k_in_window =
          metrics->recall_at_k_by_window[default_window_index];
    }
    metrics->mean_rank_distance_norm =
        static_cast<double>(total_rank_distance /
                            (static_cast<long double>(total_neighbors) * eval.base.count));
    metrics->query_recall_p05 = percentile_value(per_query_recall, 0.05);
    metrics->query_recall_p50 = percentile_value(per_query_recall, 0.50);
    metrics->query_recall_p95 = percentile_value(per_query_recall, 0.95);
    metrics->rank_distance_norm_p50 =
        percentile_value(rank_distance_norm_samples, 0.50);
    metrics->rank_distance_norm_p95 =
        percentile_value(rank_distance_norm_samples, 0.95);
    metrics->rank_distance_norm_p99 =
        percentile_value(rank_distance_norm_samples, 0.99);

    long double node_score_sum = 0.0L;
    long double node_score_weight = 0.0L;
    for (std::size_t i = 0; i < eval.node_counts.size(); ++i) {
      const double denom = static_cast<double>(total_neighbors);
      metrics->same_node_hit_by_count[i] =
          static_cast<double>(same_node_hits[i]) / denom;
      metrics->near_1_node_hit_by_count[i] =
          static_cast<double>(near_1_node_hits[i]) / denom;
      metrics->near_2_node_hit_by_count[i] =
          static_cast<double>(near_2_node_hits[i]) / denom;
      metrics->near_4_node_hit_by_count[i] =
          static_cast<double>(near_4_node_hits[i]) / denom;
      metrics->mean_node_distance_norm_by_count[i] =
          static_cast<double>(total_node_distance_norm[i] /
                              static_cast<long double>(total_neighbors));
      if (overlay_match_weight_by_count[i] > 0.0L) {
        metrics->overlay_match_by_count[i] =
            static_cast<double>(overlay_match_sum_by_count[i] /
                                overlay_match_weight_by_count[i]);
      }
      double count_score = clamp01(
          0.35 * metrics->same_node_hit_by_count[i] +
          0.30 * metrics->near_1_node_hit_by_count[i] +
          0.20 * metrics->near_2_node_hit_by_count[i] +
          0.10 * metrics->near_4_node_hit_by_count[i] +
          0.05 * (1.0 - clamp01(metrics->mean_node_distance_norm_by_count[i])));
      node_score_sum += static_cast<long double>(count_score);
      node_score_weight += 1.0L;
    }
    metrics->node_locality_score =
        node_score_weight > 0.0L
            ? static_cast<double>(node_score_sum / node_score_weight)
            : 0.0;
    if (!metrics->overlay_match_by_count.empty()) {
      long double overlay_sum = 0.0L;
      for (double score : metrics->overlay_match_by_count) {
        overlay_sum += static_cast<long double>(score);
      }
      metrics->overlay_match_score =
          static_cast<double>(overlay_sum /
                              static_cast<long double>(metrics->overlay_match_by_count.size()));
    }
    metrics->query_overlay_match_p05 =
        percentile_value(per_query_overlay_match, 0.05);
    metrics->query_overlay_match_p50 =
        percentile_value(per_query_overlay_match, 0.50);
    metrics->query_overlay_match_p95 =
        percentile_value(per_query_overlay_match, 0.95);

    if (metrics->recall_at_k_by_window.size() <= 1) {
      metrics->recall_auc_log_window = metrics->recall_at_k_in_window;
    } else {
      long double area = 0.0L;
      for (std::size_t i = 1; i < metrics->recall_windows.size(); ++i) {
        double x0 = std::log(static_cast<double>(
            std::max<uint64_t>(1, metrics->recall_windows[i - 1])));
        double x1 = std::log(static_cast<double>(
            std::max<uint64_t>(1, metrics->recall_windows[i])));
        double y0 = metrics->recall_at_k_by_window[i - 1];
        double y1 = metrics->recall_at_k_by_window[i];
        area += static_cast<long double>((x1 - x0) * (y0 + y1) * 0.5);
      }
      double x_min = std::log(static_cast<double>(
          std::max<uint64_t>(1, metrics->recall_windows.front())));
      double x_max = std::log(static_cast<double>(
          std::max<uint64_t>(1, metrics->recall_windows.back())));
      metrics->recall_auc_log_window =
          x_max > x_min ? static_cast<double>(area / (x_max - x_min))
                        : metrics->recall_at_k_in_window;
    }

    if (recall_target.has_value()) {
      for (std::size_t i = 0; i < metrics->recall_windows.size(); ++i) {
        if (metrics->recall_at_k_by_window[i] + kNormEpsilon >= *recall_target) {
          metrics->min_window_for_recall_target = metrics->recall_windows[i];
          metrics->recall_target_met_by_curve = true;
          break;
        }
      }
    }

    metrics->locality_quality_score = clamp01(
        0.50 * metrics->overlay_match_score +
        0.20 * metrics->node_locality_score +
        0.15 * metrics->recall_auc_log_window +
        0.10 * metrics->query_overlay_match_p05 +
        0.05 * (1.0 - clamp01(metrics->rank_distance_norm_p50)));
  }

  const EvalDataset& eval;
  std::vector<std::size_t> hits_by_window;
  std::vector<std::size_t> same_node_hits;
  std::vector<std::size_t> near_1_node_hits;
  std::vector<std::size_t> near_2_node_hits;
  std::vector<std::size_t> near_4_node_hits;
  std::vector<double> per_query_recall;
  std::vector<double> per_query_overlay_match;
  std::vector<double> rank_distance_norm_samples;
  std::vector<long double> total_node_distance_norm;
  std::vector<long double> overlay_match_sum_by_count;
  std::vector<long double> overlay_match_weight_by_count;
  std::size_t default_window_index = 0;
  std::size_t total_neighbors = 0;
  long double total_rank_distance = 0.0L;
};

void begin_context_train_budget(TrainCache* cache, uint32_t train_limit) {
  if (cache == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(cache->mutex);
  cache->context_model_train_start = cache->model_trains_started;
  cache->context_model_train_limit = train_limit;
  cache->context_model_train_budget_hit = false;
}

void end_context_train_budget(TrainCache* cache) {
  if (cache == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(cache->mutex);
  cache->context_model_train_limit = 0;
}

SelectorCandidateMetrics evaluate_model_metrics(const SelectorCandidateConfig& config,
                                                const TrainedCandidate& trained,
                                                const EvalDataset& eval,
                                                const PhaseLimits& phase,
                                                uint32_t eval_threads,
                                                const std::optional<double>& recall_target,
                                                TrainCache* cache,
                                                const NswSelectorContext* context) {
  auto eval_total_start = SelectorClock::now();
  SelectorCandidateMetrics metrics;
  metrics.config = config;
  metrics.phase = phase.name;
  metrics.ok = true;
  metrics.train_time_ms = trained.train_time_ms;
  metrics.train_base_cache_hit = trained.train_profile.base_cache_hit;
  metrics.train_order_cache_hit = trained.train_profile.order_cache_hit;
  metrics.train_cdf_cache_hit = trained.train_profile.cdf_cache_hit;
  metrics.train_base_cache_wait_ms = trained.train_profile.base_cache_wait_ms;
  metrics.train_order_cache_wait_ms = trained.train_profile.order_cache_wait_ms;
  metrics.train_cdf_cache_wait_ms = trained.train_profile.cdf_cache_wait_ms;
  metrics.train_base_build_ms = trained.train_profile.base_build_ms;
  metrics.train_order_build_ms = trained.train_profile.order_build_ms;
  metrics.train_cdf_cache_build_ms = trained.train_profile.cdf_cache_build_ms;
  metrics.train_gather_skeleton_ms = trained.train_profile.gather_skeleton_ms;
  metrics.train_cluster_assign_ms = trained.train_profile.cluster_assign_ms;
  metrics.train_assign_centroids_ms = trained.train_profile.assign_centroids_ms;
  metrics.train_centroid_order_ms = trained.train_profile.centroid_order_ms;
  metrics.train_range_alloc_ms = trained.train_profile.range_alloc_ms;
  metrics.train_cdf_fit_ms = trained.train_profile.cdf_fit_ms;
  metrics.train_model_assembly_ms = trained.train_profile.model_assembly_ms;
  metrics.model_memory_bytes = trained.model_memory_bytes;
  metrics.model_size_bytes = trained.model_size_bytes;
  metrics.centroid_count = static_cast<uint32_t>(trained.model.centroid_count());
  metrics.active_centroid_count =
      static_cast<uint32_t>(trained.model.router.active_centroids().size());

  if (trained.model.hash_bits <= 64) {
    uint32_t hash_threads = eval_worker_count(eval.base.count, eval_threads);
    metrics.eval_hash_base_threads = hash_threads;
    EvalHash64Plan hash_plan = make_eval_hash64_plan(trained.model);
    bool hash_plan_verified = false;
    auto verify_hash_plan = [&](const EvalNearestCache* nearest_cache) {
      if (!hash_plan_verified) {
        verify_eval_hash64_plan_sample(hash_plan, trained.model, eval, nearest_cache);
        hash_plan_verified = true;
      }
    };
    std::vector<HashEntry64> hashed_base_storage;
    std::vector<std::size_t> rank_by_index_storage;
    const std::vector<HashEntry64>* hashed_base_ptr = &hashed_base_storage;
    const std::vector<std::size_t>* rank_by_index_ptr = &rank_by_index_storage;

    bool base_rank_cache_hit = false;
    double base_rank_cache_wait_ms = 0.0;
    std::shared_ptr<const EvalBaseRankCache> base_rank_cache;
    if (cache != nullptr && can_use_eval_base_rank_cache(phase, eval)) {
      bool calibration_tier = is_eval_base_rank_calibration_phase(phase);
      SelectorBaseRankCacheAttribution attribution =
          make_eval_base_rank_cache_attribution(config,
                                                phase,
                                                eval,
                                                context,
                                                calibration_tier);
      base_rank_cache =
          get_or_build_eval_base_rank_cache(trained.model,
                                            eval,
                                            hash_threads,
                                            cache,
                                            can_use_eval_nearest_cache(trained.model,
                                                                       phase,
                                                                       eval),
                                            calibration_tier,
                                            &attribution,
                                            &base_rank_cache_hit,
                                            &base_rank_cache_wait_ms);
      if (!base_rank_cache->hash_bits_64 ||
          base_rank_cache->count != eval.base.count ||
          base_rank_cache->hashed64.size() != eval.base.count ||
          base_rank_cache->rank_by_index.size() != eval.base.count) {
        throw std::runtime_error("Selector eval base-rank cache size mismatch");
      }
      metrics.eval_hash_base_threads = base_rank_cache->hash_threads;
      metrics.eval_rank_index_threads = base_rank_cache->rank_threads;
      metrics.eval_hash_base_ms = base_rank_cache_hit
          ? base_rank_cache_wait_ms
          : base_rank_cache->hash_base_ms;
      metrics.eval_sort_base_ms = base_rank_cache_hit ? 0.0
                                                       : base_rank_cache->sort_base_ms;
      metrics.eval_rank_index_ms = base_rank_cache_hit ? 0.0
                                                       : base_rank_cache->rank_index_ms;
      hashed_base_ptr = &base_rank_cache->hashed64;
      rank_by_index_ptr = &base_rank_cache->rank_by_index;
    } else {
      rank_by_index_storage.assign(eval.base.count, 0);

      auto hash_base_start = SelectorClock::now();
      std::shared_ptr<const EvalNearestCache> nearest_cache;
      bool calibration_tier = is_eval_base_rank_calibration_phase(phase);
      if (cache != nullptr &&
          can_use_eval_nearest_cache(trained.model, phase, eval) &&
          !should_bypass_inflight_eval_nearest_cache(trained.model,
                                                     eval,
                                                     cache,
                                                     calibration_tier)) {
        nearest_cache = get_or_build_eval_nearest_cache(
            trained.model,
            eval,
            hash_threads,
            cache,
            calibration_tier);
        if (nearest_cache->count != eval.base.count ||
            nearest_cache->centroid.size() != eval.base.count ||
            nearest_cache->dist2.size() != eval.base.count) {
          throw std::runtime_error("Selector eval nearest cache size mismatch");
        }
      }
      verify_hash_plan(nearest_cache.get());

      if (nearest_cache) {
        hashed_base_storage.resize(eval.base.count);
        fill_eval_hash64_from_nearest(hash_plan,
                                      *nearest_cache,
                                      &hashed_base_storage,
                                      hash_threads);
      } else if (hash_threads <= 1) {
        hashed_base_storage.reserve(eval.base.count);
        for (std::size_t i = 0; i < eval.base.count; ++i) {
          uint64_t hash = eval_hash64_vector(
              hash_plan,
              eval.base.values.data() + i * eval.base.dim);
          hashed_base_storage.push_back(HashEntry64{hash, static_cast<uint32_t>(i)});
        }
      } else {
        hashed_base_storage.resize(eval.base.count);
        parallel_for_chunks(eval.base.count, hash_threads, [&](std::size_t begin,
                                                               std::size_t end) {
          for (std::size_t i = begin; i < end; ++i) {
            uint64_t hash =
                eval_hash64_vector(hash_plan,
                                   eval.base.values.data() + i * eval.base.dim);
            hashed_base_storage[i] = HashEntry64{hash, static_cast<uint32_t>(i)};
          }
        });
      }
      metrics.eval_hash_base_ms = elapsed_ms(hash_base_start);

      auto sort_base_start = SelectorClock::now();
      std::sort(hashed_base_storage.begin(), hashed_base_storage.end(),
                [](const HashEntry64& a, const HashEntry64& b) {
                  if (a.hash != b.hash) return a.hash < b.hash;
                  return a.index < b.index;
                });
      metrics.eval_sort_base_ms = elapsed_ms(sort_base_start);

      auto rank_index_start = SelectorClock::now();
      uint32_t rank_threads = eval_worker_count(hashed_base_storage.size(), eval_threads);
      metrics.eval_rank_index_threads = rank_threads;
      parallel_for_chunks(hashed_base_storage.size(), rank_threads,
                          [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          rank_by_index_storage[hashed_base_storage[i].index] = i;
        }
      });
      metrics.eval_rank_index_ms = elapsed_ms(rank_index_start);
    }

    LocalityMetricAccumulator accumulator(eval);

    auto query_start = SelectorClock::now();
    const auto& hashed_base = *hashed_base_ptr;
    const auto& rank_by_index = *rank_by_index_ptr;
    verify_hash_plan(nullptr);
    for (std::size_t qi = 0; qi < eval.query.count; ++qi) {
      const float* vec = eval.query.values.data() + qi * eval.query.dim;
      uint64_t qhash = eval_hash64_vector(hash_plan, vec);
      std::size_t pos = lower_bound_hash(hashed_base, qhash);
      if (pos >= eval.base.count) {
        pos = eval.base.count - 1;
      }

      accumulator.record_query(qi, pos, rank_by_index);
    }
    metrics.eval_query_ms = elapsed_ms(query_start);
    accumulator.finish(&metrics, recall_target);
  } else {
    uint32_t hash_threads = eval_worker_count(eval.base.count, eval_threads);
    metrics.eval_hash_base_threads = hash_threads;
    std::vector<HashEntry256> hashed_base_storage;
    std::vector<std::size_t> rank_by_index_storage;
    const std::vector<HashEntry256>* hashed_base_ptr = &hashed_base_storage;
    const std::vector<std::size_t>* rank_by_index_ptr = &rank_by_index_storage;

    bool base_rank_cache_hit = false;
    double base_rank_cache_wait_ms = 0.0;
    std::shared_ptr<const EvalBaseRankCache> base_rank_cache;
    if (cache != nullptr && can_use_eval_base_rank_cache(phase, eval)) {
      bool calibration_tier = is_eval_base_rank_calibration_phase(phase);
      SelectorBaseRankCacheAttribution attribution =
          make_eval_base_rank_cache_attribution(config,
                                                phase,
                                                eval,
                                                context,
                                                calibration_tier);
      base_rank_cache =
          get_or_build_eval_base_rank_cache(trained.model,
                                            eval,
                                            hash_threads,
                                            cache,
                                            can_use_eval_nearest_cache(trained.model,
                                                                       phase,
                                                                       eval),
                                            calibration_tier,
                                            &attribution,
                                            &base_rank_cache_hit,
                                            &base_rank_cache_wait_ms);
      if (base_rank_cache->hash_bits_64 ||
          base_rank_cache->count != eval.base.count ||
          base_rank_cache->hashed256.size() != eval.base.count ||
          base_rank_cache->rank_by_index.size() != eval.base.count) {
        throw std::runtime_error("Selector eval base-rank cache size mismatch");
      }
      metrics.eval_hash_base_threads = base_rank_cache->hash_threads;
      metrics.eval_rank_index_threads = base_rank_cache->rank_threads;
      metrics.eval_hash_base_ms = base_rank_cache_hit
          ? base_rank_cache_wait_ms
          : base_rank_cache->hash_base_ms;
      metrics.eval_sort_base_ms = base_rank_cache_hit ? 0.0
                                                       : base_rank_cache->sort_base_ms;
      metrics.eval_rank_index_ms = base_rank_cache_hit ? 0.0
                                                       : base_rank_cache->rank_index_ms;
      hashed_base_ptr = &base_rank_cache->hashed256;
      rank_by_index_ptr = &base_rank_cache->rank_by_index;
    } else {
      rank_by_index_storage.assign(eval.base.count, 0);

      auto hash_base_start = SelectorClock::now();
      std::shared_ptr<const EvalNearestCache> nearest_cache;
      bool calibration_tier = is_eval_base_rank_calibration_phase(phase);
      if (cache != nullptr &&
          can_use_eval_nearest_cache(trained.model, phase, eval) &&
          !should_bypass_inflight_eval_nearest_cache(trained.model,
                                                     eval,
                                                     cache,
                                                     calibration_tier)) {
        nearest_cache = get_or_build_eval_nearest_cache(
            trained.model,
            eval,
            hash_threads,
            cache,
            calibration_tier);
        if (nearest_cache->count != eval.base.count ||
            nearest_cache->centroid.size() != eval.base.count ||
            nearest_cache->dist2.size() != eval.base.count) {
          throw std::runtime_error("Selector eval nearest cache size mismatch");
        }
      }

      if (hash_threads <= 1) {
        hashed_base_storage.reserve(eval.base.count);
        for (std::size_t i = 0; i < eval.base.count; ++i) {
          UInt256 hash = nearest_cache
              ? trained.model.hash_from_centroid(nearest_cache->centroid[i],
                                                 nearest_cache->dist2[i]).value
              : trained.model.hash(eval.base.values.data() + i * eval.base.dim).value;
          hashed_base_storage.push_back(HashEntry256{hash, static_cast<uint32_t>(i)});
        }
      } else {
        hashed_base_storage.resize(eval.base.count);
        parallel_for_chunks(eval.base.count, hash_threads, [&](std::size_t begin,
                                                               std::size_t end) {
          for (std::size_t i = begin; i < end; ++i) {
            UInt256 hash = nearest_cache
                ? trained.model.hash_from_centroid(nearest_cache->centroid[i],
                                                   nearest_cache->dist2[i]).value
                : trained.model.hash(eval.base.values.data() + i * eval.base.dim).value;
            hashed_base_storage[i] = HashEntry256{hash, static_cast<uint32_t>(i)};
          }
        });
      }
      metrics.eval_hash_base_ms = elapsed_ms(hash_base_start);

      auto sort_base_start = SelectorClock::now();
      std::sort(hashed_base_storage.begin(), hashed_base_storage.end(),
                [](const HashEntry256& a, const HashEntry256& b) {
                  int cmp = compare(a.hash, b.hash);
                  if (cmp != 0) return cmp < 0;
                  return a.index < b.index;
                });
      metrics.eval_sort_base_ms = elapsed_ms(sort_base_start);

      auto rank_index_start = SelectorClock::now();
      uint32_t rank_threads = eval_worker_count(hashed_base_storage.size(), eval_threads);
      metrics.eval_rank_index_threads = rank_threads;
      parallel_for_chunks(hashed_base_storage.size(), rank_threads,
                          [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          rank_by_index_storage[hashed_base_storage[i].index] = i;
        }
      });
      metrics.eval_rank_index_ms = elapsed_ms(rank_index_start);
    }

    LocalityMetricAccumulator accumulator(eval);

    auto query_start = SelectorClock::now();
    const auto& hashed_base = *hashed_base_ptr;
    const auto& rank_by_index = *rank_by_index_ptr;
    for (std::size_t qi = 0; qi < eval.query.count; ++qi) {
      const float* vec = eval.query.values.data() + qi * eval.query.dim;
      UInt256 qhash = trained.model.hash(vec).value;
      std::size_t pos = lower_bound_hash(hashed_base, qhash);
      if (pos >= eval.base.count) {
        pos = eval.base.count - 1;
      }

      accumulator.record_query(qi, pos, rank_by_index);
    }
    metrics.eval_query_ms = elapsed_ms(query_start);
    accumulator.finish(&metrics, recall_target);
  }

  auto latency_start = SelectorClock::now();
  std::vector<uint64_t> latency_samples;
  latency_samples.reserve(phase.latency_iterations);
  std::size_t query_count = eval.query.count;
  for (uint32_t i = 0; i < phase.latency_warmup; ++i) {
    const float* vec = eval.query.values.data() + (i % query_count) * eval.query.dim;
    (void)trained.model.hash(vec);
  }
  for (uint32_t i = 0; i < phase.latency_iterations; ++i) {
    const float* vec = eval.query.values.data() + (i % query_count) * eval.query.dim;
    auto begin = std::chrono::steady_clock::now();
    (void)trained.model.hash(vec);
    auto end = std::chrono::steady_clock::now();
    uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    latency_samples.push_back(ns);
  }

  if (!latency_samples.empty()) {
    long double total_ns = 0.0L;
    for (uint64_t ns : latency_samples) {
      total_ns += static_cast<long double>(ns);
    }
    metrics.latency_avg_ms =
        static_cast<double>(total_ns / static_cast<long double>(latency_samples.size())) / 1e6;
    metrics.latency_p95_ms = percentile_ms(latency_samples, 0.95);
    metrics.latency_p99_ms = percentile_ms(latency_samples, 0.99);
  }
  metrics.eval_latency_ms = elapsed_ms(latency_start);
  metrics.eval_total_ms = elapsed_ms(eval_total_start);

  return metrics;
}

} // namespace vortex::selector_internal
