#include "vortex_selector_test_common.h"

using namespace vortex_test;

namespace {

void test_model_selector_recall_target_and_capacity_scaling(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(128, 128);
  storage.values.resize(storage.count * storage.dim);
  std::mt19937 rng(991);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& value : storage.values) {
    value = dist(rng);
  }
  auto dataset_path = unique_temp_path("vortex_selector_recall_target.fvecs");
  vector_io::write_fvecs(dataset_path.string(), storage);

  vortex::BuildHnswOptions build_opts;
  build_opts.dataset_path = dataset_path;
  build_opts.M = 8;
  build_opts.ef_construction = 80;
  auto index_path = unique_temp_path("vortex_selector_recall_target.index");
  vortex::build_hnsw_index(build_opts, index_path);

  vortex::SelectorOptions opts;
  opts.dataset_path = dataset_path;
  opts.nsw_index_path = index_path;
  opts.target_skeleton_values = {16, 128};
  opts.hash_bits = 32;
  opts.threads = 1;
  opts.selector_parallelism = 2;
  opts.seed = 31;
  opts.K_values = {4, 6, 8, 10, 12, 14, 16, 24, 32};
  opts.centroid_knn_values = {4, 8, 16, 24, 32};
  opts.cdf_branching_values = {4};
  opts.cdf_model_specs = {"linear,linear"};
  opts.two_opt_iterations_values = {2};
  opts.include_disable_2opt = false;
  opts.phase1_keep = 3;
  opts.max_phase2_candidates = 8;
  opts.recommend_count = 3;
  opts.eval_k = 5;
  opts.eval_window = 20;
  opts.eval_window_values = {10, 20, 40};
  opts.coarse_base_limit = 80;
  opts.coarse_query_limit = 24;
  opts.eval_base_limit = 120;
  opts.eval_query_limit = 24;
  opts.coarse_latency_iterations = 8;
  opts.coarse_latency_warmup = 2;
  opts.latency_iterations = 16;
  opts.latency_warmup = 4;
  opts.recall_target = 0.0;
  opts.optimize_for_recall = true;
  opts.max_recall_refine_rounds = 1;

  auto result = vortex::select_vortex_models(opts);
  ctx.check(result.best.has_value(),
            "selector recall-target search returns a best candidate");
  ctx.check(result.recall_target_met,
            "selector recall-target metadata marks target as met");
  ctx.check(result.recall_target_met_recommended == result.recommended.size(),
            "selector recall-target metadata tracks met recommended count");
  ctx.check(result.best->recall_target_met_by_curve,
            "selector recall-target search uses recall curve metadata");
  ctx.check(result.best->min_window_for_recall_target > 0,
            "selector recall-target search reports minimum target window");
  ctx.check(result.effective_weights.recall >= 0.80,
            "selector optimize-for-recall increases effective recall weight");

  std::set<uint32_t> small_ks;
  std::set<uint32_t> large_ks;
  for (const auto& item : result.candidates) {
    if (!item.ok) {
      continue;
    }
    if (item.config.target_skeleton == 16) {
      small_ks.insert(item.config.K);
    }
    if (item.config.target_skeleton == 128) {
      large_ks.insert(item.config.K);
    }
  }
  ctx.check(!small_ks.empty() && !large_ks.empty(),
            "selector capacity scaling produces candidates for both skeleton targets");
  ctx.check(*large_ks.rbegin() > *small_ks.rbegin(),
            "selector capacity scaling gives larger skeleton targets higher K capacity");

  auto json = vortex::selector_result_to_json(result, opts);
  ctx.check(json.is_object(),
            "selector recall-target json output is an object");
  if (json.is_object()) {
    const auto* selection = json.find("selection");
    ctx.check(selection && selection->is_object(),
              "selector recall-target json includes selection object");
    if (selection && selection->is_object()) {
      const auto* recall_target = selection->find("recall_target");
      ctx.check(recall_target && recall_target->is_number(),
                "selector json includes recall_target");
      const auto* recall_target_met = selection->find("recall_target_met");
      ctx.check(recall_target_met && recall_target_met->is_bool(),
                "selector json includes recall_target_met");
      const auto* recall_target_met_recommended = selection->find("recall_target_met_recommended");
      ctx.check(recall_target_met_recommended && recall_target_met_recommended->is_number(),
                "selector json includes recall_target_met_recommended");
      const auto* scaling = selection->find("skeleton_capacity_scaling");
      ctx.check(scaling && scaling->is_bool(),
                "selector json includes skeleton_capacity_scaling");
      const auto* capacity_policy = selection->find("selector_capacity_policy");
      ctx.check(capacity_policy && capacity_policy->is_string(),
                "selector json includes selector_capacity_policy");
      const auto* optimize_for_recall = selection->find("optimize_for_recall");
      ctx.check(optimize_for_recall && optimize_for_recall->is_bool(),
                "selector recall-target json includes optimize_for_recall");
      const auto* max_recall_refine_rounds = selection->find("max_recall_refine_rounds");
      ctx.check(max_recall_refine_rounds && max_recall_refine_rounds->is_number(),
                "selector recall-target json includes max_recall_refine_rounds");
    }
  }

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(index_path);
}

} // namespace

int main() {
  TestContext ctx;

  test_model_selector_recall_target_and_capacity_scaling(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All vortex_v1 selector recall tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " tests failed\n";
  return 1;
}
