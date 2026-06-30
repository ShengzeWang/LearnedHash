#include "vortex_selector_test_common.h"

using namespace vortex_test;

namespace {

void test_model_selector_reuses_equivalent_final_calibration(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(4, 48);
  storage.values.resize(storage.count * storage.dim);
  std::mt19937 rng(456);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& value : storage.values) {
    value = dist(rng);
  }
  auto dataset_path = unique_temp_path("vortex_selector_reuse_dataset.fvecs");
  vector_io::write_fvecs(dataset_path.string(), storage);

  vortex::NswCsr csr;
  csr.dim = storage.dim;
  csr.metric = vortex::Metric::L2;
  csr.layer = 0;
  csr.node_ids.resize(storage.count);
  for (std::size_t i = 0; i < storage.count; ++i) {
    csr.node_ids[i] = static_cast<uint64_t>(i);
  }
  csr.offsets.assign(storage.count + 1, 0);
  auto nsw_path = unique_temp_path("vortex_selector_reuse_dataset.csr");
  csr.write(nsw_path);

  vortex::SelectorOptions opts;
  opts.dataset_path = dataset_path;
  opts.nsw_path = nsw_path;
  opts.hash_bits = 32;
  opts.threads = 1;
  opts.selector_parallelism = 1;
  opts.seed = 19;
  opts.K_values = {4, 8};
  opts.centroid_knn_values = {2, 4};
  opts.cdf_branching_values = {4};
  opts.cdf_model_specs = {"linear,linear"};
  opts.two_opt_iterations_values = {2};
  opts.include_disable_2opt = true;
  opts.phase1_keep = 2;
  opts.max_phase2_candidates = 3;
  opts.recommend_count = 2;
  opts.eval_k = 5;
  opts.eval_window = 12;
  opts.eval_window_values = {6, 12};
  opts.coarse_base_limit = storage.count;
  opts.coarse_query_limit = storage.count;
  opts.eval_base_limit = 0;
  opts.eval_query_limit = 0;
  opts.coarse_latency_iterations = 8;
  opts.coarse_latency_warmup = 2;
  opts.latency_iterations = 8;
  opts.latency_warmup = 2;
  opts.final_calibration_count = 2;
  opts.final_calibration_query_limit = 0;
  opts.post_exploration_candidates = 0;

  auto result = vortex::select_vortex_models(opts);
  ctx.check(result.final_calibration_evaluated > 0,
            "selector equivalent final calibration produces calibrated metrics");
  ctx.check(result.final_calibration_reused == result.final_calibration_evaluated,
            "selector reuses equivalent full-eval metrics for final calibration");
  ctx.check(!result.final_calibration_progressive_probes.empty(),
            "selector equivalent calibration reports stability probe");
  if (!result.final_calibration_progressive_probes.empty()) {
    const auto& probe = result.final_calibration_progressive_probes.front();
    ctx.check(probe.peak_recall_match,
              "equivalent calibration stability probe keeps peak-recall role stable");
    ctx.check(probe.knee_match && probe.fast_match && probe.small_match,
              "equivalent calibration stability probe keeps secondary roles stable");
  }

  for (const auto& item : result.candidates) {
    if (item.phase == "final_calibration" && item.ok) {
      ctx.check(item.eval_hash_base_ms == 0.0,
                "reused final calibration reports zero additional base hashing");
    }
  }

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(nsw_path);
}

} // namespace

int main() {
  TestContext ctx;

  test_model_selector_reuses_equivalent_final_calibration(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All vortex_v1 selector calibration tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " tests failed\n";
  return 1;
}
