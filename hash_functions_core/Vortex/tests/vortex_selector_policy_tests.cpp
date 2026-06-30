#include "vortex_test_common.h"

#include "vortex_v1/model_selector.h"
#include "vortex_v1/nsw_csr.h"
#include "vortex_v1/training.h"

#include "vector_io/vector_io.hpp"

#include "../src/selector_internal.h"

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
#include <stdexcept>
#include <string>
#include <vector>

using namespace vortex_test;

namespace {

void test_model_selector_phase1_candidates(TestContext& ctx) {
  vortex::NswCsr csr;
  csr.dim = 4;
  csr.metric = vortex::Metric::L2;
  csr.layer = 0;
  csr.node_ids.resize(32);
  for (std::size_t i = 0; i < csr.node_ids.size(); ++i) {
    csr.node_ids[i] = static_cast<uint64_t>(i);
  }
  csr.offsets.assign(csr.node_ids.size() + 1, 0);
  auto nsw_path = unique_temp_path("vortex_selector_phase1.csr");
  csr.write(nsw_path);

  vortex::SelectorOptions opts;
  opts.dataset_path = "unused.fvecs";
  opts.nsw_path = nsw_path;
  opts.K_values = {4, 8, 16, 64};
  opts.centroid_knn_values = {1, 2, 4, 32};
  opts.cdf_branching_values = {4, 8};
  opts.cdf_model_specs = {"linear,linear", "cubic,linear"};
  opts.two_opt_iterations_values = {2, 4};
  opts.include_disable_2opt = true;

  auto candidates = vortex::build_phase1_selector_candidates(opts);
  ctx.check(!candidates.empty(), "selector phase1 candidate set is non-empty");
  for (const auto& candidate : candidates) {
    ctx.check(candidate.K >= 2 && candidate.K <= 32, "selector candidate K clamped by skeleton");
    ctx.check(candidate.centroid_knn >= 1 && candidate.centroid_knn < candidate.K,
              "selector candidate centroid_knn in [1, K-1]");
    ctx.check(candidate.cdf_branching_factor >= 2,
              "selector candidate cdf branching factor >= 2");
    ctx.check(!candidate.enable_graph_centroid_order,
              "selector phase1 preserves the Euclidean centroid-order baseline");
  }

  std::filesystem::remove(nsw_path);
}
void test_final_calibration_reserve_guard(TestContext& ctx) {
  vortex::selector_internal::SearchController search;
  search.start_time = std::chrono::steady_clock::now();
  search.deadline = search.start_time + std::chrono::seconds(5);
  search.final_calibration_reserve_seconds = 10.0;

  ctx.check(search.can_launch_work_with_min_seconds(1.0, 0.0),
            "search controller allows work that fits without reserve");
  ctx.check(!search.can_launch_work_with_min_seconds(
                1.0, search.final_calibration_reserve_seconds),
            "search controller blocks work that would consume calibration reserve");
  ctx.check(search.remaining_time_guard_hit.load(std::memory_order_relaxed),
            "reserve guard marks remaining-time guard hit");
  ctx.check(search.final_calibration_reserve_hit.load(std::memory_order_relaxed),
            "reserve guard marks final calibration reserve hit");
}

void test_final_calibration_screen_candidate_policy(TestContext& ctx) {
  vortex::SelectorOptions opts;
  opts.recommend_count = 8;
  opts.final_calibration_count = 24;
  ctx.check(vortex::selector_internal::resolve_final_calibration_full_count(
                opts, 1'000'000) == 8,
            "large-dataset final calibration auto-promotes the recommendation count");
  ctx.check(vortex::selector_internal::resolve_final_calibration_screen_candidate_count(
                opts, 1'000'000) == 16,
            "large-dataset calibration screen auto-compacts to two promotion widths");
  ctx.check(vortex::selector_internal::resolve_final_calibration_screen_candidate_count(
                opts, 10'000) == 24,
            "small datasets keep the requested calibration screen width");

  opts.final_calibration_screen_candidate_count = 12;
  ctx.check(vortex::selector_internal::resolve_final_calibration_screen_candidate_count(
                opts, 1'000'000) == 12,
            "explicit calibration screen candidate count overrides the auto cap");
}

void test_auto_target_skeleton_policy(TestContext& ctx) {
  vortex::SelectorOptions opts;
  opts.nsw_index_path = std::filesystem::path("synthetic.index");

  ctx.check(vortex::selector_internal::uses_auto_target_skeleton_policy(opts),
            "index-only selector input enables auto target_skeleton policy");
  ctx.check((vortex::selector_internal::default_target_skeleton_values(opts, 10000) ==
             std::vector<uint64_t>{2000, 5000, 10000}),
            "small-dataset auto target_skeleton policy keeps a compact sweep");
  ctx.check((vortex::selector_internal::resolve_target_skeleton_values(opts, 1000000) ==
             std::vector<uint64_t>{30000, 100000, 300000, 1000000}),
            "full-SIFT auto target_skeleton policy starts larger and reaches the full base layer");
  ctx.check((vortex::selector_internal::default_target_skeleton_values(opts, 50000000) ==
             std::vector<uint64_t>{300000, 1000000, 3000000, 5000000}),
            "very-large default target_skeleton policy stays bounded");

  opts.optimize_for_recall = true;
  ctx.check((vortex::selector_internal::default_target_skeleton_values(opts, 50000000) ==
             std::vector<uint64_t>{300000, 1000000, 3000000, 10000000}),
            "recall-optimized target_skeleton policy raises the large-dataset cap");
  ctx.check((vortex::selector_internal::default_target_skeleton_values(opts, 1000000000) ==
             std::vector<uint64_t>{1000000, 3000000, 10000000, 30000000}),
            "billion-scale target_skeleton policy uses bounded large anchors");

  opts.target_skeleton_percentages = {1.0, 50.0};
  ctx.check(!vortex::selector_internal::uses_auto_target_skeleton_policy(opts),
            "explicit target_skeleton percentages override auto policy");
  ctx.check((vortex::selector_internal::resolve_target_skeleton_values(opts, 1000000) ==
             std::vector<uint64_t>{10000, 500000}),
            "explicit target_skeleton percentages are preserved exactly");

  vortex::SelectorOptions fixed;
  fixed.nsw_path = std::filesystem::path("fixed.csr");
  ctx.check(!vortex::selector_internal::uses_auto_target_skeleton_policy(fixed),
            "fixed NSW input keeps legacy single-skeleton behavior");
  ctx.check(vortex::selector_internal::resolve_target_skeleton_values(fixed, 1000000).empty(),
            "fixed NSW input does not synthesize target_skeleton values");
}

void test_default_k_scale_policy(TestContext& ctx) {
  auto contains = [](const std::vector<uint32_t>& values, uint32_t target) {
    return std::find(values.begin(), values.end(), target) != values.end();
  };

  auto small = vortex::selector_internal::default_k_values(10000, 10000, true);
  ctx.check(!small.empty() && small.front() == 32 && small.back() == 256,
            "small-dataset recall K ladder stays within stable clustering density");

  auto sift = vortex::selector_internal::default_k_values(1000000, 1000000, true);
  ctx.check(!sift.empty() && sift.front() >= 512 && sift.back() == 24576,
            "full-SIFT recall K ladder starts higher and respects stable clustering density");
  ctx.check(contains(sift, 512) && contains(sift, 4096) && contains(sift, 16384),
            "full-SIFT recall K ladder keeps practical midpoints");

  auto mid_scale = vortex::selector_internal::default_k_values(5000000, 3000000, true);
  ctx.check(!mid_scale.empty() && mid_scale.front() >= 1024 && mid_scale.back() == 65536,
            "5M-scale recall K ladder expands beyond full-SIFT caps");

  auto billion = vortex::selector_internal::default_k_values(1000000000, 30000000, true);
  ctx.check(!billion.empty() && billion.front() >= 16384 && billion.back() == 262144,
            "billion-scale recall K ladder remains bounded but large");

  auto balanced = vortex::selector_internal::default_k_values(1000000, 1000000, false);
  ctx.check(!balanced.empty() && balanced.front() >= 256 && balanced.back() == 8192,
            "balanced full-SIFT K ladder is smaller than recall-optimized mode");
}

void test_large_dataset_budget_policy(TestContext& ctx) {
  vortex::SelectorOptions opts;
  ctx.check(vortex::selector_internal::resolve_search_time_budget_seconds(
                opts, "balanced", 1'000'000) == 600,
            "full-SIFT balanced auto budget remains bounded");
  ctx.check(vortex::selector_internal::resolve_search_time_budget_seconds(
                opts, "balanced", 10'000'000) == 1200,
            "10M balanced auto budget allows sustained search");
  ctx.check(vortex::selector_internal::resolve_model_train_budget(
                opts, "balanced", 10'000'000, 4) == 224,
            "10M balanced model-train budget scales with context count");

  opts.optimize_for_recall = true;
  ctx.check(vortex::selector_internal::resolve_search_time_budget_seconds(
                opts, "balanced", 10'000'000) == 2400,
            "10M recall mode doubles the balanced search budget");
  ctx.check(vortex::selector_internal::resolve_model_train_budget(
                opts, "balanced", 10'000'000, 4) == 448,
            "10M recall mode expands model-train budget without becoming unbounded");

  opts.max_search_seconds = 3600;
  opts.max_model_trains = 720;
  ctx.check(vortex::selector_internal::resolve_search_time_budget_seconds(
                opts, "balanced", 10'000'000) == 3600,
            "explicit search budget overrides 10M auto policy");
  ctx.check(vortex::selector_internal::resolve_model_train_budget(
                opts, "balanced", 10'000'000, 4) == 720,
            "explicit model-train budget overrides 10M auto policy");
}

void test_assignment_sample_limit_policy(TestContext& ctx) {
  vortex::SelectorOptions opts;
  opts.nsw_path = std::filesystem::path("fixed.csr");

  ctx.check(vortex::selector_internal::default_assignment_sample_limit(opts, 1000000) == 0,
            "full-SIFT-scale assignment policy remains exact by default");
  ctx.check(vortex::selector_internal::default_assignment_sample_limit(opts, 5000000) ==
                2000000,
            "larger balanced runs get a bounded assignment cap");

  opts.optimize_for_recall = true;
  ctx.check(vortex::selector_internal::default_assignment_sample_limit(opts, 5000000) ==
                3000000,
            "recall-optimized larger runs get a larger assignment cap");

  opts.dataset_count_hint = 5000000;
  auto normalized = vortex::selector_internal::normalize_selector_options(opts, 100000);
  ctx.check(normalized.assignment_sample_limit == 3000000,
            "selector normalization applies the auto assignment cap");

  opts.auto_assignment_sample_limit = false;
  opts.assignment_sample_limit = 0;
  normalized = vortex::selector_internal::normalize_selector_options(opts, 100000);
  ctx.check(normalized.assignment_sample_limit == 0,
            "explicit zero assignment sample limit keeps exact assignment");

  opts.use_full_dataset_for_assignments = false;
  opts.assignment_sample_limit = 1234;
  normalized = vortex::selector_internal::normalize_selector_options(opts, 100000);
  ctx.check(normalized.assignment_sample_limit == 0,
            "skeleton-only assignment ignores sample caps");
}

void test_context_capacity_scaling_preserves_midpoints(TestContext& ctx) {
  std::vector<uint32_t> values = {
      512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
  };
  std::vector<uint32_t> tuned = vortex::selector_internal::context_capacity_values(
      values,
      31254,
      31254.0 / 1000000.0,
      2,
      8192);
  auto contains = [&](uint32_t value) {
    return std::find(tuned.begin(), tuned.end(), value) != tuned.end();
  };
  ctx.check(contains(512) && contains(768) && contains(1024) && contains(1536),
            "small-context capacity scaling preserves low/mid K coverage");
  ctx.check(contains(8192) && contains(12288) && contains(16384),
            "small-context capacity scaling still keeps high-K probes");
}

void test_attribution_guided_compaction_policy(TestContext& ctx) {
  vortex::SelectorOptions opts;
  opts.optimize_for_recall = true;
  opts.dataset_count_hint = 1'000'000;
  opts.min_recall_improvement = 0.001;

  std::vector<vortex::SelectorCandidateMetrics> phase1;
  auto add_phase1 = [&](uint32_t k, double quality, double recall) {
    vortex::SelectorCandidateMetrics metric;
    metric.ok = true;
    metric.config.K = k;
    metric.config.centroid_knn = std::max<uint32_t>(1, k / 4);
    metric.locality_quality_score = quality;
    metric.recall_at_k_in_window = recall;
    phase1.push_back(metric);
  };
  add_phase1(512, 0.60, 0.60);
  add_phase1(1024, 0.55, 0.55);

  std::vector<vortex::SelectorCandidateConfig> candidates;
  for (uint32_t i = 0; i < 4; ++i) {
    vortex::SelectorCandidateConfig config;
    config.K = 512;
    config.centroid_knn = 128 + i;
    config.cdf_branching_factor = 32 + i;
    candidates.push_back(config);
  }
  for (uint32_t i = 0; i < 10; ++i) {
    vortex::SelectorCandidateConfig config;
    config.K = 1024;
    config.centroid_knn = 256 + i;
    config.cdf_branching_factor = 64 + i;
    candidates.push_back(config);
  }

  uint32_t pruned = 0;
  std::vector<vortex::SelectorCandidateConfig> compacted =
      vortex::selector_internal::compact_attribution_guided_candidates(
          candidates, phase1, opts, 3, &pruned);
  uint32_t kept_512 = 0;
  uint32_t kept_1024 = 0;
  for (const auto& config : compacted) {
    if (config.K == 512) ++kept_512;
    if (config.K == 1024) ++kept_1024;
  }
  ctx.check(pruned == 7 && kept_512 == 4 && kept_1024 == 3,
            "attribution-guided compaction caps weak high-K families");

  opts.attribution_guided_compaction = false;
  compacted = vortex::selector_internal::compact_attribution_guided_candidates(
      candidates, phase1, opts, 3, &pruned);
  ctx.check(pruned == 0 && compacted.size() == candidates.size(),
            "disabled attribution-guided compaction preserves all candidates");

  opts.attribution_guided_compaction = true;
  opts.dataset_count_hint = 10'000;
  compacted = vortex::selector_internal::compact_attribution_guided_candidates(
      candidates, phase1, opts, 3, &pruned);
  ctx.check(pruned == 0 && compacted.size() == candidates.size(),
            "small datasets skip high-K compaction");

}

void test_selector_quality_anchor_policy(TestContext& ctx) {
  auto make_metric = [](uint32_t k,
                        double recall,
                        double quality,
                        double objective,
                        double latency_ms) {
    vortex::SelectorCandidateMetrics metric;
    metric.ok = true;
    metric.config.K = k;
    metric.config.centroid_knn = std::max<uint32_t>(1, k / 2);
    metric.recall_at_k_in_window = recall;
    metric.locality_quality_score = quality;
    metric.objective_score = objective;
    metric.latency_avg_ms = latency_ms;
    return metric;
  };

  vortex::SelectorCandidateConfig fallback;
  fallback.K = 1;
  std::vector<vortex::SelectorCandidateMetrics> metrics = {
      make_metric(64, 0.91, 0.60, 0.20, 10.0),
      make_metric(128, 0.86, 0.83, 0.70, 20.0),
      make_metric(256, 0.90, 0.82, 0.90, 1.0),
  };

  auto recall_anchor =
      vortex::selector_internal::choose_recall_anchor(metrics, fallback);
  ctx.check(recall_anchor.K == 64,
            "recall anchor keeps peak fixed-window recall as the first key");

  auto quality_anchor =
      vortex::selector_internal::choose_quality_anchor(metrics, fallback);
  ctx.check(quality_anchor.K == 128,
            "quality anchor keeps locality-quality as the first key");

  metrics[1].locality_quality_score = metrics[2].locality_quality_score;
  quality_anchor = vortex::selector_internal::choose_quality_anchor(metrics, fallback);
  ctx.check(quality_anchor.K == 256,
            "quality anchor uses recall, objective, and latency as deterministic ties");
}

void test_min_quality_improvement_validation(TestContext& ctx) {
  vortex::SelectorOptions opts;
  opts.nsw_path = std::filesystem::path("fixed.csr");
  opts.K_values = {8};
  opts.centroid_knn_values = {2};
  opts.cdf_branching_values = {4};
  opts.cdf_model_specs = {"linear,linear"};
  opts.two_opt_iterations_values = {1};
  opts.min_quality_improvement = -0.01;

  bool threw = false;
  try {
    (void)vortex::selector_internal::normalize_selector_options(opts, 16);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  ctx.check(threw, "selector rejects negative min_quality_improvement");
}

vortex::SelectorCandidateMetrics make_screen_metric(uint32_t k,
                                                    double recall,
                                                    double latency_ms,
                                                    uint64_t model_size_bytes) {
  vortex::SelectorCandidateMetrics metric;
  metric.ok = true;
  metric.phase = "final_calibration_screen";
  metric.nsw_path = "synthetic.csr";
  metric.config.target_skeleton = 100;
  metric.config.K = k;
  metric.config.centroid_knn = std::max<uint32_t>(1, k / 2);
  metric.config.cdf_model_spec = "linear,linear";
  metric.config.cdf_branching_factor = 8;
  metric.config.enable_2opt = false;
  metric.config.two_opt_iterations = 1;
  metric.recall_at_k_in_window = recall;
  metric.recall_auc_log_window = recall;
  metric.query_recall_p05 = recall;
  metric.locality_quality_score = recall;
  metric.mean_rank_distance_norm = 1.0 - recall;
  metric.latency_avg_ms = latency_ms;
  metric.model_size_bytes = model_size_bytes;
  return metric;
}

void test_role_aware_final_calibration_promotion(TestContext& ctx) {
  std::vector<vortex::SelectorCandidateMetrics> screen_metrics = {
      make_screen_metric(10, 0.930, 100.0, 10000),
      make_screen_metric(20, 0.9299, 10.0, 5000),
      make_screen_metric(30, 0.925, 1.0, 9000),
      make_screen_metric(40, 0.925, 50.0, 10),
      make_screen_metric(50, 0.928, 80.0, 8000),
  };
  uint32_t role_promoted = 0;
  uint32_t recall_promoted = 0;
  auto promoted =
      vortex::selector_internal::choose_role_aware_final_calibration_seeds(
          screen_metrics,
          {},
          5,
          std::nullopt,
          std::nullopt,
          &role_promoted,
          &recall_promoted);
  ctx.check(promoted.size() == 5,
            "role-aware calibration promotion fills the requested cap");
  ctx.check(role_promoted == 4,
            "role-aware calibration promotion preserves four role winners");
  ctx.check(recall_promoted == 1,
            "role-aware calibration promotion adds recall challenger after roles");

  std::set<uint32_t> promoted_k;
  for (const auto& item : promoted) {
    promoted_k.insert(item.config.K);
  }
  ctx.check(promoted_k.count(10) == 1,
            "role-aware calibration promotion includes peak-recall winner");
  ctx.check(promoted_k.count(20) == 1,
            "role-aware calibration promotion includes knee winner");
  ctx.check(promoted_k.count(30) == 1,
            "role-aware calibration promotion includes fast winner");
  ctx.check(promoted_k.count(40) == 1,
            "role-aware calibration promotion includes small winner");
  ctx.check(promoted_k.count(50) == 1,
            "role-aware calibration promotion includes top recall challenger");
}

} // namespace

int main() {
  TestContext ctx;

  test_model_selector_phase1_candidates(ctx);
  test_final_calibration_reserve_guard(ctx);
  test_final_calibration_screen_candidate_policy(ctx);
  test_auto_target_skeleton_policy(ctx);
  test_default_k_scale_policy(ctx);
  test_large_dataset_budget_policy(ctx);
  test_assignment_sample_limit_policy(ctx);
  test_context_capacity_scaling_preserves_midpoints(ctx);
  test_attribution_guided_compaction_policy(ctx);
  test_selector_quality_anchor_policy(ctx);
  test_min_quality_improvement_validation(ctx);
  test_role_aware_final_calibration_promotion(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All vortex_v1 selector policy tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " tests failed\n";
  return 1;
}
