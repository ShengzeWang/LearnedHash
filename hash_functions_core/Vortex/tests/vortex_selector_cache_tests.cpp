#include "vortex_test_common.h"

#include "vortex_v1/model_selector.h"
#include "vortex_v1/nsw_csr.h"
#include "vortex_v1/training.h"

#include "vector_io/vector_io.hpp"

#include "../src/selector_internal.h"
#include "../src/selector_eval_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace vortex_test;

namespace {

void test_eval_nearest_cache_hash_stability(TestContext& ctx) {
  auto model = make_golden_model();

  vortex::selector_internal::EvalDataset eval;
  eval.base = vector_io::VectorStorage<float>(2, 5);
  eval.base.values = {
      0.5f, 0.0f,
      1.0f, 0.0f,
      0.0f, 2.75f,
      100.0f, 100.0f,
      -3.0f, 4.0f,
  };

  auto cache = vortex::selector_internal::build_eval_nearest_cache(model, eval, 2);
  ctx.check(cache->count == eval.base.count,
            "eval nearest cache preserves base vector count");
  ctx.check(cache->distance_terms_evaluated > 0,
            "eval nearest cache reports evaluated distance terms");

  for (std::size_t i = 0; i < eval.base.count; ++i) {
    const float* vec = eval.base.values.data() + i * eval.base.dim;
    auto direct_nearest = model.router.nearest(vec);
    ctx.check(cache->centroid[i] == direct_nearest.centroid,
              "eval nearest cache centroid matches router");
    ctx.check(std::abs(cache->dist2[i] - direct_nearest.dist2) < 1e-12,
              "eval nearest cache distance matches router");
    auto direct_hash = model.hash(vec);
    auto cached_hash = model.hash_from_centroid(cache->centroid[i], cache->dist2[i]);
    ctx.check(cached_hash.to_u64() == direct_hash.to_u64(),
              "eval nearest cache hash matches direct hash");
  }

  auto cubic_linear = make_golden_model();
  cubic_linear.cdf_top_type = vortex::CdfModelType::Cubic;
  cubic_linear.cdf_leaf_type = vortex::CdfModelType::Linear;
  cubic_linear.cdf_top_param_count =
      vortex::cdf_param_count(cubic_linear.cdf_top_type);
  cubic_linear.cdf_leaf_param_count =
      vortex::cdf_param_count(cubic_linear.cdf_leaf_type);
  cubic_linear.cdf_top_params.assign(cubic_linear.centroid_count() *
                                         cubic_linear.cdf_top_param_count,
                                     0.0);
  auto cubic_cache =
      vortex::selector_internal::build_eval_nearest_cache(cubic_linear, eval, 2);
  auto hash_plan = vortex::selector_internal::make_eval_hash64_plan(cubic_linear);
  std::vector<vortex::selector_internal::HashEntry64> filled(eval.base.count);
  vortex::selector_internal::fill_eval_hash64_from_nearest(
      hash_plan, *cubic_cache, &filled, 2);
  for (std::size_t i = 0; i < eval.base.count; ++i) {
    uint64_t expected =
        cubic_linear.hash_from_centroid_u64(cubic_cache->centroid[i],
                                            cubic_cache->dist2[i]);
    ctx.check(filled[i].index == i && filled[i].hash == expected,
              "typed nearest-cache hash fill preserves cubic-linear hashes");
  }

  auto active_subset = make_golden_model();
  active_subset.range_size[0].words = {0, 0, 0, 0};
  active_subset.compute_active();
  uint64_t terms = 0;
  float query[2] = {2.5f, 0.0f};
  uint64_t full_terms =
      static_cast<uint64_t>(active_subset.router.active_centroids().size()) *
      active_subset.dim;
  auto subset_direct = active_subset.router.nearest(query);
  auto subset_cached =
      vortex::selector_internal::nearest_for_eval_cache_exact(active_subset, query, &terms);
  ctx.check(terms > 0, "eval nearest cache helper reports active-subset distance terms");
  ctx.check(terms < full_terms,
            "eval nearest cache helper can early-abandon non-winning centroid distances");
  ctx.check(subset_cached.centroid == subset_direct.centroid,
            "eval nearest cache helper respects active centroid subset");
  ctx.check(std::abs(subset_cached.dist2 - subset_direct.dist2) < 1e-12,
            "eval nearest cache helper preserves active-subset distance");
}

void test_eval_base_rank_cache_metric_parity(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(2, 8);
  storage.values = {
      0.0f, 0.0f,
      0.4f, 0.1f,
      1.5f, 0.0f,
      2.5f, 0.2f,
      0.0f, 1.7f,
      0.2f, 2.5f,
      3.0f, 3.0f,
      -0.5f, 0.3f,
  };

  auto eval = vortex::selector_internal::build_eval_dataset(
      storage,
      std::nullopt,
      2,
      3,
      {1, 2, 3, 5},
      {4, 8},
      0,
      5,
      17,
      true);

  vortex::SelectorCandidateConfig config;
  config.target_skeleton = storage.count;
  config.K = 4;
  config.centroid_knn = 2;
  config.cdf_model_spec = "linear,linear";
  config.cdf_branching_factor = 4;
  config.enable_2opt = false;
  config.two_opt_iterations = 1;

  vortex::selector_internal::TrainedCandidate trained;
  trained.model = make_golden_model();
  trained.model_memory_bytes =
      vortex::selector_internal::estimate_model_memory_bytes(trained.model);
  trained.model_size_bytes =
      vortex::selector_internal::serialized_model_size_bytes(trained.model);

  vortex::selector_internal::PhaseLimits phase2{"phase2", 0, 0, 4, 1};
  vortex::selector_internal::PhaseLimits screen_phase{
      "final_calibration_screen", 0, 0, 4, 1};
  vortex::selector_internal::PhaseLimits final_phase{"final_calibration", 0, 0, 4, 1};

  auto uncached = vortex::selector_internal::evaluate_model_metrics(
      config, trained, eval, phase2, 2, std::nullopt, nullptr);

  vortex::selector_internal::TrainCache phase_cache;
  auto phase2_miss = vortex::selector_internal::evaluate_model_metrics(
      config, trained, eval, phase2, 2, std::nullopt, &phase_cache);
  auto phase2_hit = vortex::selector_internal::evaluate_model_metrics(
      config, trained, eval, phase2, 2, std::nullopt, &phase_cache);

  vortex::selector_internal::TrainCache cache;
  auto cached_miss = vortex::selector_internal::evaluate_model_metrics(
      config, trained, eval, screen_phase, 2, std::nullopt, &cache);
  auto cached_hit = vortex::selector_internal::evaluate_model_metrics(
      config, trained, eval, final_phase, 2, std::nullopt, &cache);

  auto close = [](double lhs, double rhs) {
    return std::abs(lhs - rhs) < 1e-12;
  };

  ctx.check(cached_miss.recall_windows == uncached.recall_windows,
            "base-rank cache preserves recall window ladder on miss");
  ctx.check(cached_hit.recall_windows == uncached.recall_windows,
            "base-rank cache preserves recall window ladder on hit");
  ctx.check(phase2_miss.recall_at_k_by_window == uncached.recall_at_k_by_window,
            "phase2 base-rank cache miss preserves recall curve");
  ctx.check(phase2_hit.recall_at_k_by_window == uncached.recall_at_k_by_window,
            "phase2 base-rank cache hit preserves recall curve");
  ctx.check(cached_miss.recall_at_k_by_window == uncached.recall_at_k_by_window,
            "base-rank cache miss preserves recall curve");
  ctx.check(cached_hit.recall_at_k_by_window == uncached.recall_at_k_by_window,
            "base-rank cache hit preserves recall curve");
  ctx.check(cached_hit.node_counts == uncached.node_counts,
            "base-rank cache hit preserves node count ladder");
  ctx.check(cached_hit.same_node_hit_by_count == uncached.same_node_hit_by_count &&
                cached_hit.near_1_node_hit_by_count == uncached.near_1_node_hit_by_count &&
                cached_hit.near_2_node_hit_by_count == uncached.near_2_node_hit_by_count &&
                cached_hit.near_4_node_hit_by_count == uncached.near_4_node_hit_by_count,
            "base-rank cache hit preserves node-locality curve");
  ctx.check(close(cached_miss.recall_at_k_in_window,
                  uncached.recall_at_k_in_window),
            "base-rank cache miss preserves default-window recall");
  ctx.check(close(cached_hit.recall_at_k_in_window,
                  uncached.recall_at_k_in_window),
            "base-rank cache hit preserves default-window recall");
  ctx.check(close(cached_hit.mean_rank_distance_norm,
                  uncached.mean_rank_distance_norm),
            "base-rank cache hit preserves rank-distance metric");
  ctx.check(close(cached_hit.query_recall_p05, uncached.query_recall_p05),
            "base-rank cache hit preserves tail query recall");
  ctx.check(close(cached_hit.node_locality_score, uncached.node_locality_score),
            "base-rank cache hit preserves node locality score");
  ctx.check(cached_hit.overlay_match_by_count == uncached.overlay_match_by_count &&
                close(cached_hit.overlay_match_score, uncached.overlay_match_score) &&
                close(cached_hit.query_overlay_match_p05,
                      uncached.query_overlay_match_p05),
            "base-rank cache hit preserves overlay match metrics");
  ctx.check(phase_cache.eval_base_rank_cache_misses == 1 &&
                phase_cache.eval_base_rank_cache_hits == 1,
            "base-rank cache reuses identical phase2 base/model");
  ctx.check(phase_cache.eval_nearest_cache_misses == 0 &&
                phase_cache.eval_nearest_cache_hits == 0,
            "phase2 base-rank cache does not build calibration nearest cache");
  auto phase_cache_memory = phase_cache.eval_base_rank_cache_memory_bytes;
  vortex::selector_internal::prepare_eval_base_rank_cache_for_phase(&phase_cache,
                                                                    screen_phase);
  ctx.check(phase_cache.eval_base_rank_cache_entries == 0 &&
                phase_cache.eval_base_rank_cache_memory_bytes == 0,
            "calibration phase evicts sampled base-rank cache entries");
  ctx.check(phase_cache.eval_base_rank_cache_evictions == 1 &&
                phase_cache.eval_base_rank_cache_evicted_bytes == phase_cache_memory,
            "sampled base-rank cache eviction reports evicted bytes");
  ctx.check(phase_cache.base_rank_cache_attribution_by_key.size() == 1,
            "phase2 base-rank attribution is grouped by phase/config");
  if (!phase_cache.base_rank_cache_attribution_by_key.empty()) {
    const auto& row = phase_cache.base_rank_cache_attribution_by_key.begin()->second;
    ctx.check(row.phase == "phase2" &&
                  row.K == config.K &&
                  row.misses == 1 &&
                  row.hits == 1 &&
                  row.builds == 1,
              "phase2 base-rank attribution reports hits and misses");
    ctx.check(row.retained_memory_bytes == 0 &&
                  row.evictions == 1 &&
                  row.evicted_bytes == phase_cache_memory,
              "phase2 base-rank attribution tracks retained and evicted bytes");
  }
  ctx.check(cache.eval_base_rank_cache_misses == 1,
            "base-rank cache builds once for identical calibration base/model");
  ctx.check(cache.eval_base_rank_cache_hits == 1,
            "base-rank cache reuses identical calibration base/model");
  ctx.check(cache.eval_nearest_cache_sampled_build_ms == 0.0 &&
                close(cache.eval_nearest_cache_build_ms,
                      cache.eval_nearest_cache_calibration_build_ms),
            "nearest-cache telemetry separates calibration builds from sampled builds");
  ctx.check(cache.eval_nearest_cache_sampled_wait_ms == 0.0 &&
                cache.eval_nearest_cache_calibration_wait_ms == 0.0,
            "nearest-cache telemetry reports no wait when calibration cache is built once");
  ctx.check(cache.eval_base_rank_cache_entries == 1 &&
                cache.eval_base_rank_cache_memory_bytes > 0,
            "base-rank cache reports retained cache memory");
  bool saw_screen_miss_attribution = false;
  bool saw_final_hit_attribution = false;
  for (const auto& [_, row] : cache.base_rank_cache_attribution_by_key) {
    if (row.phase == "final_calibration_screen") {
      saw_screen_miss_attribution =
          row.calibration_tier &&
          row.misses == 1 &&
          row.builds == 1 &&
          row.hash_ms >= 0.0 &&
          row.retained_memory_bytes == cache.eval_base_rank_cache_memory_bytes;
    }
    if (row.phase == "final_calibration") {
      saw_final_hit_attribution =
          row.calibration_tier &&
          row.hits == 1 &&
          row.wait_ms >= 0.0;
    }
  }
  ctx.check(saw_screen_miss_attribution,
            "base-rank attribution records final-calibration screen miss cost");
  ctx.check(saw_final_hit_attribution,
            "base-rank attribution records final-calibration cache-hit reuse");
  ctx.check(cached_hit.eval_sort_base_ms == 0.0 &&
                cached_hit.eval_rank_index_ms == 0.0,
            "base-rank cache hit skips repeated base sort and rank-index work");
  ctx.check(cached_miss.eval_sort_base_ms >= 0.0 &&
                cached_miss.eval_rank_index_ms >= 0.0,
            "base-rank cache miss reports base sort and rank-index timings");
}

void test_eval_base_rank_sampled_cache_budget(TestContext& ctx) {
  constexpr uint64_t kMiB = 1024ULL * 1024ULL;
  constexpr uint64_t kGiB = 1024ULL * kMiB;

  vortex::selector_internal::TrainCache cache;
  ctx.check(vortex::selector_internal::eval_base_rank_sampled_cache_budget_bytes(
                &cache) == 256ULL * kMiB,
            "base-rank sampled cache defaults to the large-run cap");

  cache.memory_budget_bytes = 1ULL * kMiB;
  ctx.check(vortex::selector_internal::eval_base_rank_sampled_cache_budget_bytes(
                &cache) == 1ULL * kMiB,
            "base-rank sampled cache respects tiny explicit memory budgets");

  cache.memory_budget_bytes = 8ULL * kGiB;
  ctx.check(vortex::selector_internal::eval_base_rank_sampled_cache_budget_bytes(
                &cache) == 64ULL * kMiB,
            "base-rank sampled cache scales with moderate memory budgets");

  cache.memory_budget_bytes = 48ULL * kGiB;
  ctx.check(vortex::selector_internal::eval_base_rank_sampled_cache_budget_bytes(
                &cache) == 256ULL * kMiB,
            "base-rank sampled cache is capped for large memory budgets");
}

void test_phase1_beam_can_reuse_sampled_nearest_cache(TestContext& ctx) {
  auto model = make_golden_model();
  model.centroids.resize(static_cast<std::size_t>(256) * model.dim, 0.0f);

  vortex::selector_internal::EvalDataset small_eval;
  small_eval.base = vector_io::VectorStorage<float>(2, 9999);

  vortex::selector_internal::EvalDataset large_eval;
  large_eval.base = vector_io::VectorStorage<float>(2, 10000);

  vortex::selector_internal::PhaseLimits phase1{"phase1", 0, 0, 0, 0};
  vortex::selector_internal::PhaseLimits phase1_beam{
      "phase1_beam", 0, 0, 0, 0};

  ctx.check(!vortex::selector_internal::can_use_eval_nearest_cache(
                model, phase1, large_eval),
            "plain phase1 does not build sampled nearest cache");
  ctx.check(!vortex::selector_internal::can_use_eval_nearest_cache(
                model, phase1_beam, small_eval),
            "phase1 beam keeps small sampled evals on the direct hash path");
  ctx.check(vortex::selector_internal::can_use_eval_nearest_cache(
                model, phase1_beam, large_eval),
            "phase1 beam reuses nearest cache for large sampled evals");
}

void test_final_calibration_ready_cache_bypasses_launch_guard(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(2, 8);
  storage.values = {
      0.0f, 0.0f,
      0.4f, 0.1f,
      1.5f, 0.0f,
      2.5f, 0.2f,
      0.0f, 1.7f,
      0.2f, 2.5f,
      3.0f, 3.0f,
      -0.5f, 0.3f,
  };

  auto eval = vortex::selector_internal::build_eval_dataset(
      storage,
      std::nullopt,
      2,
      3,
      {3},
      {4},
      0,
      4,
      29,
      true);

  vortex::SelectorCandidateConfig config;
  config.target_skeleton = storage.count;
  config.K = 4;
  config.centroid_knn = 2;
  config.cdf_model_spec = "linear,linear";
  config.cdf_branching_factor = 4;
  config.enable_2opt = false;
  config.two_opt_iterations = 1;

  vortex::selector_internal::NswSelectorContext context;
  context.display_nsw_path = "synthetic-ready-cache.csr";
  context.skeleton_identity = "synthetic-ready-cache";
  context.target_skeleton = storage.count;
  context.csr.dim = storage.dim;
  context.csr.metric = vortex::Metric::L2;
  context.csr.layer = 0;
  context.csr.node_ids.resize(storage.count);
  for (std::size_t i = 0; i < storage.count; ++i) {
    context.csr.node_ids[i] = static_cast<uint64_t>(i);
  }
  context.csr.offsets.assign(storage.count + 1, 0);

  vortex::SelectorOptions opts;
  opts.hash_bits = 64;
  opts.eval_k = 2;
  opts.latency_iterations = 4;
  opts.latency_warmup = 1;

  vortex::selector_internal::TrainedCandidate trained;
  trained.model = make_golden_model();
  trained.model_memory_bytes =
      vortex::selector_internal::estimate_model_memory_bytes(trained.model);
  trained.model_size_bytes =
      vortex::selector_internal::serialized_model_size_bytes(trained.model);

  std::promise<vortex::selector_internal::TrainedCandidate> train_promise;
  auto future = train_promise.get_future().share();
  train_promise.set_value(std::move(trained));

  vortex::selector_internal::TrainCache cache;
  cache.future_by_key.emplace(
      vortex::selector_internal::selector_training_cache_key(context, config, opts),
      future);

  vortex::selector_internal::SearchController search;
  search.start_time = std::chrono::steady_clock::now();
  search.deadline = search.start_time + std::chrono::seconds(1);

  vortex::selector_internal::PhaseLimits phase2{"phase2", 0, 0, 4, 1};
  auto guarded = vortex::selector_internal::evaluate_candidate(
      config, opts, 1, context, eval, phase2, &cache, &search);
  ctx.check(!guarded.ok &&
                guarded.error.find("remaining-time guard") != std::string::npos,
            "ready cached non-final candidate still respects eval launch guard");

  search.remaining_time_guard_hit.store(false, std::memory_order_relaxed);
  vortex::selector_internal::PhaseLimits final_phase{
      "final_calibration", 0, 0, 4, 1};
  auto calibrated = vortex::selector_internal::evaluate_candidate(
      config, opts, 1, context, eval, final_phase, &cache, &search);
  ctx.check(calibrated.ok,
            "ready cached final calibration candidate bypasses launch guard");
  ctx.check(calibrated.phase == "final_calibration",
            "ready cached final calibration preserves phase label");
  ctx.check(calibrated.recall_at_k_in_window >= 0.0 &&
                calibrated.recall_at_k_in_window <= 1.0,
            "ready cached final calibration produces bounded recall");
}

void test_final_calibration_prewarms_ready_nearest_cache(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(2, 8);
  storage.values = {
      0.0f, 0.0f,
      0.4f, 0.1f,
      1.5f, 0.0f,
      2.5f, 0.2f,
      0.0f, 1.7f,
      0.2f, 2.5f,
      3.0f, 3.0f,
      -0.5f, 0.3f,
  };

  auto eval = vortex::selector_internal::build_eval_dataset(
      storage,
      std::nullopt,
      2,
      3,
      {3},
      {4},
      0,
      4,
      31,
      true);

  vortex::SelectorCandidateConfig config;
  config.target_skeleton = storage.count;
  config.K = 4;
  config.centroid_knn = 2;
  config.cdf_model_spec = "linear,linear";
  config.cdf_branching_factor = 4;
  config.enable_2opt = false;
  config.two_opt_iterations = 1;

  vortex::selector_internal::NswSelectorContext context;
  context.display_nsw_path = "synthetic-prewarm-cache.csr";
  context.skeleton_identity = "synthetic-prewarm-cache";
  context.target_skeleton = storage.count;
  context.csr.dim = storage.dim;
  context.csr.metric = vortex::Metric::L2;
  context.csr.layer = 0;
  context.csr.node_ids.resize(storage.count);
  for (std::size_t i = 0; i < storage.count; ++i) {
    context.csr.node_ids[i] = static_cast<uint64_t>(i);
  }
  context.csr.offsets.assign(storage.count + 1, 0);

  vortex::SelectorOptions opts;
  opts.hash_bits = 64;
  opts.threads = 2;
  opts.eval_k = 2;
  opts.latency_iterations = 4;
  opts.latency_warmup = 1;

  vortex::selector_internal::TrainedCandidate trained;
  trained.model = make_golden_model();
  trained.model_memory_bytes =
      vortex::selector_internal::estimate_model_memory_bytes(trained.model);
  trained.model_size_bytes =
      vortex::selector_internal::serialized_model_size_bytes(trained.model);

  std::promise<vortex::selector_internal::TrainedCandidate> train_promise;
  auto future = train_promise.get_future().share();
  train_promise.set_value(std::move(trained));

  vortex::selector_internal::TrainCache cache;
  cache.future_by_key.emplace(
      vortex::selector_internal::selector_training_cache_key(context, config, opts),
      future);

  vortex::selector_internal::PhaseLimits final_phase{
      "final_calibration", 0, 0, 4, 1};
  auto metrics = vortex::selector_internal::evaluate_candidates(
      {config}, opts, context, eval, final_phase, &cache, nullptr);

  ctx.check(metrics.size() == 1 && metrics[0].ok,
            "final calibration prewarm preserves candidate evaluation");
  ctx.check(cache.eval_nearest_cache_prewarm_requests == 1 &&
                cache.eval_nearest_cache_prewarm_ready_models == 1,
            "final calibration prewarm uses ready cached model");
  ctx.check(cache.eval_nearest_cache_prewarm_skipped == 0 &&
                cache.eval_nearest_cache_prewarm_failures == 0,
            "final calibration prewarm reports no skip or failure for ready model");
  ctx.check(cache.eval_nearest_cache_prewarm_threads >= 1,
            "final calibration prewarm records worker thread count");
  ctx.check(cache.eval_nearest_cache_entries == 1 &&
                cache.eval_nearest_cache_calibration_memory_bytes > 0,
            "final calibration prewarm retains one calibration nearest cache");
  ctx.check(cache.eval_nearest_cache_hits >= 1,
            "candidate evaluation reuses the prewarmed nearest cache");
}

void test_training_cache_key_includes_assignment_policy(TestContext& ctx) {
  vortex::selector_internal::NswSelectorContext context;
  context.skeleton_identity = "cache-key-policy";

  vortex::SelectorCandidateConfig config;
  config.target_skeleton = 1000;
  config.K = 16;
  config.centroid_knn = 8;
  config.cdf_model_spec = "linear,linear";
  config.cdf_branching_factor = 16;
  config.enable_2opt = true;
  config.two_opt_iterations = 4;

  vortex::SelectorOptions exact;
  exact.seed = 7;
  exact.use_full_dataset_for_assignments = true;
  exact.assignment_sample_limit = 0;
  exact.enable_graph_centroid_order = true;
  exact.include_disable_graph_centroid_order = true;

  vortex::SelectorOptions sampled = exact;
  sampled.assignment_sample_limit = 1000000;

  vortex::SelectorOptions skeleton = exact;
  skeleton.use_full_dataset_for_assignments = false;

  ctx.check(vortex::selector_internal::selector_training_cache_key(context, config, exact) !=
                vortex::selector_internal::selector_training_cache_key(context, config, sampled),
            "training cache key separates exact and sampled assignment models");
  ctx.check(vortex::selector_internal::selector_training_cache_key(context, config, exact) !=
                vortex::selector_internal::selector_training_cache_key(context, config, skeleton),
            "training cache key separates full-dataset and skeleton-only assignment models");

  vortex::SelectorCandidateConfig graph_disabled_config = config;
  graph_disabled_config.enable_graph_centroid_order = false;
  ctx.check(vortex::selector_internal::selector_training_cache_key(context, config, exact) !=
                vortex::selector_internal::selector_training_cache_key(
                    context, graph_disabled_config, exact),
            "training cache key separates graph-aware and legacy centroid ordering");

  bool graph_base = vortex::selector_internal::selector_training_base_includes_centroid_graph(
      exact, config);
  bool disabled_graph_base =
      vortex::selector_internal::selector_training_base_includes_centroid_graph(
          exact, graph_disabled_config);
  ctx.check(graph_base && disabled_graph_base,
            "mixed graph-order search builds one graph-capable training base");

  auto base_key = vortex::selector_internal::selector_training_base_cache_key(
      context,
      config,
      exact.seed,
      exact.use_full_dataset_for_assignments,
      exact.assignment_sample_limit,
      graph_base);
  auto graph_disabled_base_key =
      vortex::selector_internal::selector_training_base_cache_key(
          context,
          graph_disabled_config,
          exact.seed,
          exact.use_full_dataset_for_assignments,
          exact.assignment_sample_limit,
          disabled_graph_base);
  ctx.check(base_key == graph_disabled_base_key,
            "training base cache reuses graph-capable base across order policies");
  ctx.check(vortex::selector_internal::selector_training_order_cache_key(
                base_key, config) !=
                vortex::selector_internal::selector_training_order_cache_key(
                    graph_disabled_base_key, graph_disabled_config),
            "training order cache key separates graph-aware and legacy ordering");
  ctx.check(vortex::selector_internal::selector_training_cdf_cache_key(
                base_key, config) ==
                vortex::selector_internal::selector_training_cdf_cache_key(
                    graph_disabled_base_key, graph_disabled_config),
            "training CDF cache reuses order-independent CDF fit");

  vortex::SelectorOptions graph_disabled_search = exact;
  graph_disabled_search.enable_graph_centroid_order = false;
  graph_disabled_search.include_disable_graph_centroid_order = false;
  ctx.check(!vortex::selector_internal::selector_training_base_includes_centroid_graph(
                graph_disabled_search, graph_disabled_config),
            "graph-disabled searches keep the cheaper graphless training base");
}

} // namespace

int main() {
  TestContext ctx;

  test_eval_nearest_cache_hash_stability(ctx);
  test_eval_base_rank_cache_metric_parity(ctx);
  test_eval_base_rank_sampled_cache_budget(ctx);
  test_phase1_beam_can_reuse_sampled_nearest_cache(ctx);
  test_final_calibration_ready_cache_bypasses_launch_guard(ctx);
  test_final_calibration_prewarms_ready_nearest_cache(ctx);
  test_training_cache_key_includes_assignment_policy(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All vortex_v1 selector cache tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " tests failed\n";
  return 1;
}
