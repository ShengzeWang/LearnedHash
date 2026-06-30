#include "vortex_selector_test_common.h"

using namespace vortex_test;

namespace {

void test_model_selector_target_skeleton_search(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(8, 96);
  storage.values.resize(storage.count * storage.dim);
  std::mt19937 rng(321);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& value : storage.values) {
    value = dist(rng);
  }
  auto dataset_path = unique_temp_path("vortex_selector_target_skeleton.fvecs");
  vector_io::write_fvecs(dataset_path.string(), storage);

  vortex::BuildHnswOptions build_opts;
  build_opts.dataset_path = dataset_path;
  build_opts.M = 8;
  build_opts.ef_construction = 80;
  auto index_path = unique_temp_path("vortex_selector_target_skeleton.index");
  vortex::build_hnsw_index(build_opts, index_path);

  vortex::SelectorOptions opts;
  opts.dataset_path = dataset_path;
  opts.nsw_index_path = index_path;
  opts.target_skeleton_values = {16, 32};
  opts.hash_bits = 32;
  opts.threads = 1;
  opts.seed = 17;
  opts.K_values = {8, 16};
  opts.centroid_knn_values = {4, 8};
  opts.cdf_branching_values = {4};
  opts.cdf_model_specs = {"linear,linear"};
  opts.two_opt_iterations_values = {2};
  opts.include_disable_2opt = false;
  opts.phase1_keep = 2;
  opts.max_phase2_candidates = 6;
  opts.recommend_count = 2;
  opts.eval_k = 5;
  opts.eval_window = 16;
  opts.coarse_base_limit = 64;
  opts.coarse_query_limit = 24;
  opts.eval_base_limit = 96;
  opts.eval_query_limit = 24;
  opts.coarse_latency_iterations = 16;
  opts.coarse_latency_warmup = 4;
  opts.latency_iterations = 32;
  opts.latency_warmup = 8;

  auto result = vortex::select_vortex_models(opts);
  ctx.check(result.best.has_value(),
            "selector target-skeleton search returns a best candidate");
  if (result.best.has_value()) {
    uint64_t ts = result.best->config.target_skeleton;
    ctx.check(ts == 16 || ts == 32,
              "selector best target_skeleton is one of requested values");
  }
  for (const auto& item : result.recommended) {
    uint64_t ts = item.config.target_skeleton;
    ctx.check(ts == 16 || ts == 32,
              "selector recommended target_skeleton is one of requested values");
  }

  auto json = vortex::selector_result_to_json(result, opts);
  ctx.check(json.is_object(),
            "selector target-skeleton json output is an object");
  if (json.is_object()) {
    const auto* selected = json.find("selected_target_skeleton");
    ctx.check(selected && selected->is_number(),
              "selector json includes selected_target_skeleton");
    const auto* selection = json.find("selection");
    ctx.check(selection && selection->is_object(),
              "selector json includes selection object");
    if (selection && selection->is_object()) {
      const auto* values = selection->find("target_skeleton_values");
      ctx.check(values && values->is_array(),
                "selector json includes target_skeleton_values");
    }
  }

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(index_path);
}

void test_model_selector_target_skeleton_percent_search(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(8, 100);
  storage.values.resize(storage.count * storage.dim);
  std::mt19937 rng(789);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& value : storage.values) {
    value = dist(rng);
  }
  auto dataset_path = unique_temp_path("vortex_selector_target_skeleton_percent.fvecs");
  vector_io::write_fvecs(dataset_path.string(), storage);

  vortex::BuildHnswOptions build_opts;
  build_opts.dataset_path = dataset_path;
  build_opts.M = 8;
  build_opts.ef_construction = 80;
  auto index_path = unique_temp_path("vortex_selector_target_skeleton_percent.index");
  vortex::build_hnsw_index(build_opts, index_path);

  vortex::SelectorOptions opts;
  opts.dataset_path = dataset_path;
  opts.nsw_index_path = index_path;
  opts.target_skeleton_values = {12};
  opts.target_skeleton_percentages = {10.0, 50.0, 100.0};
  opts.hash_bits = 32;
  opts.threads = 1;
  opts.selector_parallelism = 2;
  opts.seed = 23;
  opts.K_values = {8};
  opts.centroid_knn_values = {4};
  opts.cdf_branching_values = {4};
  opts.cdf_model_specs = {"linear,linear"};
  opts.two_opt_iterations_values = {2};
  opts.include_disable_2opt = false;
  opts.phase1_keep = 2;
  opts.max_phase2_candidates = 4;
  opts.recommend_count = 2;
  opts.max_target_skeleton_contexts = 1;
  opts.eval_k = 5;
  opts.eval_window = 16;
  opts.coarse_base_limit = 64;
  opts.coarse_query_limit = 24;
  opts.eval_base_limit = 96;
  opts.eval_query_limit = 24;
  opts.coarse_latency_iterations = 8;
  opts.coarse_latency_warmup = 2;
  opts.latency_iterations = 16;
  opts.latency_warmup = 4;

  auto result = vortex::select_vortex_models(opts);
  ctx.check(result.best.has_value(),
            "selector target-skeleton percentage search returns a best candidate");
  ctx.check(result.target_skeleton_contexts_requested >= 2,
            "selector percentage search records multiple requested skeleton contexts");
  ctx.check(result.target_skeleton_contexts_selected == 1,
            "selector percentage search honors max_target_skeleton_contexts=1");

  auto is_expected_target = [](uint64_t value) {
    return value == 10 || value == 12 || value == 50 || value == 100;
  };
  for (const auto& item : result.candidates) {
    ctx.check(is_expected_target(item.config.target_skeleton),
              "selector percentage candidate target_skeleton resolved as expected");
  }

  auto json = vortex::selector_result_to_json(result, opts);
  ctx.check(json.is_object(),
            "selector target-skeleton percentage json output is an object");
  if (json.is_object()) {
    const auto* selection = json.find("selection");
    ctx.check(selection && selection->is_object(),
              "selector percentage json includes selection object");
    if (selection && selection->is_object()) {
      const auto* percentages = selection->find("target_skeleton_percentages");
      ctx.check(percentages && percentages->is_array(),
                "selector percentage json includes target_skeleton_percentages");
      const auto* resolved = selection->find("resolved_target_skeleton_values");
      ctx.check(resolved && resolved->is_array(),
                "selector percentage json includes resolved_target_skeleton_values");
      const auto* parallelism = selection->find("selector_parallelism");
      ctx.check(parallelism && parallelism->is_number(),
                "selector percentage json includes selector_parallelism");
    }
  }

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(index_path);
}

} // namespace

int main() {
  TestContext ctx;

  test_model_selector_target_skeleton_search(ctx);
  test_model_selector_target_skeleton_percent_search(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All vortex_v1 selector skeleton tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " tests failed\n";
  return 1;
}
