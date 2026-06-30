#include "vortex_test_common.h"

#include "../src/training_internal.h"

#include "vortex_v1/nsw_csr.h"
#include "vortex_v1/training.h"

#include "vector_io/vector_io.hpp"

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <memory>

using namespace vortex_test;

namespace {

void test_model_determinism(TestContext& ctx, const std::filesystem::path& dataset_path) {
  if (!std::filesystem::exists(dataset_path)) {
    std::cerr << "[SKIP] dataset not found for determinism test\n";
    return;
  }

  auto data = vector_io::read_fvecs(dataset_path.string());
  if (data.count < 100) {
    std::cerr << "[SKIP] dataset too small for determinism test\n";
    return;
  }

  vortex::NswCsr csr;
  csr.dim = data.dim;
  csr.metric = vortex::Metric::L2;
  csr.layer = 0;
  csr.node_ids.clear();
  for (std::size_t i = 0; i < std::min<std::size_t>(500, data.count); ++i) {
    csr.node_ids.push_back(static_cast<uint64_t>(i));
  }
  csr.offsets.assign(csr.node_ids.size() + 1, 0);

  auto temp = unique_temp_path("vortex_det.csr");
  csr.write(temp);

  vortex::TrainOptions opts;
  opts.dataset_path = dataset_path;
  opts.nsw_path = temp;
  opts.K = 8;
  opts.hash_bits = 32;
  opts.cdf_model_spec = "linear,linear";
  opts.cdf_branching_factor = 4;
  opts.centroid_knn = 8;
  opts.seed = 123;
  opts.threads = 1;

  auto model_a = vortex::train_vortex(opts);
  auto model_b = vortex::train_vortex(opts);

  ctx.check(model_a.centroids.size() == model_b.centroids.size(), "Centroid size mismatch");
  double max_diff = 0.0;
  for (std::size_t i = 0; i < model_a.centroids.size(); ++i) {
    double diff = std::abs(static_cast<double>(model_a.centroids[i]) -
                           static_cast<double>(model_b.centroids[i]));
    max_diff = std::max(max_diff, diff);
  }
  ctx.check(max_diff < 1e-5, "Centroids not deterministic");

  for (std::size_t i = 0; i < 10; ++i) {
    const float* vec = data.values.data() + i * data.dim;
    auto h1 = model_a.hash(vec).to_u64();
    auto h2 = model_b.hash(vec).to_u64();
    ctx.check(h1 == h2, "Hash mismatch in determinism test");
  }

  std::filesystem::remove(temp);
}

void test_train_option_validation(TestContext& ctx) {
  vortex::TrainOptions opts;
  opts.dataset_path = "missing.fvecs";
  opts.nsw_path = "missing.csr";
  opts.K = 8;
  opts.hash_bits = 32;
  opts.cdf_model_spec = "linear,linear";
  opts.cdf_branching_factor = 4;
  opts.centroid_knn = 8;

  expect_throw(
      ctx,
      [&]() {
        auto bad = opts;
        bad.K = 0;
        (void)vortex::train_vortex(bad);
      },
      "train_vortex rejects K=0");

  expect_throw(
      ctx,
      [&]() {
        auto bad = opts;
        bad.hash_bits = 0;
        (void)vortex::train_vortex(bad);
      },
      "train_vortex rejects hash_bits=0");

  expect_throw(
      ctx,
      [&]() {
        auto bad = opts;
        bad.hash_bits = 257;
        (void)vortex::train_vortex(bad);
      },
      "train_vortex rejects hash_bits>256");

  expect_throw(
      ctx,
      [&]() {
        auto bad = opts;
        bad.cdf_model_spec.clear();
        (void)vortex::train_vortex(bad);
      },
      "train_vortex rejects empty cdf_model_spec");

  expect_throw(
      ctx,
      [&]() {
        auto bad = opts;
        bad.cdf_branching_factor = 1;
        (void)vortex::train_vortex(bad);
      },
      "train_vortex rejects cdf_branching_factor<2");

  expect_throw(
      ctx,
      [&]() {
        auto bad = opts;
        bad.centroid_knn = 0;
        (void)vortex::train_vortex(bad);
      },
      "train_vortex rejects centroid_knn=0");

  expect_throw(
      ctx,
      [&]() {
        auto bad = opts;
        bad.enable_2opt = true;
        bad.two_opt_iterations = 0;
        (void)vortex::train_vortex(bad);
      },
      "train_vortex rejects enable_2opt with zero iterations");
}

void test_train_empty_skeleton(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(1, 2);
  storage.values = {0.5f, -1.0f};
  auto dataset_path = unique_temp_path("vortex_train_empty_skel.fvecs");
  vector_io::write_fvecs(dataset_path.string(), storage);

  vortex::NswCsr csr;
  csr.dim = storage.dim;
  csr.metric = vortex::Metric::L2;
  csr.layer = 0;
  csr.node_ids = {};
  csr.offsets = {0};
  csr.neighbors = {};
  auto csr_path = unique_temp_path("vortex_train_empty_skel.csr");
  csr.write(csr_path);

  vortex::TrainOptions opts;
  opts.dataset_path = dataset_path;
  opts.nsw_path = csr_path;
  opts.K = 1;
  opts.hash_bits = 32;
  opts.cdf_model_spec = "linear,linear";
  opts.cdf_branching_factor = 2;
  opts.centroid_knn = 1;
  opts.seed = 123;
  opts.threads = 1;

  expect_throw(
      ctx,
      [&]() { (void)vortex::train_vortex(opts); },
      "train_vortex rejects empty NSW skeleton");

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(csr_path);
}

void test_extract_nsw_layer_membership(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(4, 64);
  storage.values.resize(storage.count * storage.dim);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : storage.values) {
    v = dist(rng);
  }

  auto dataset_path = unique_temp_path("vortex_layer_membership.fvecs");
  vector_io::write_fvecs(dataset_path.string(), storage);

  vortex::BuildHnswOptions build_opts;
  build_opts.dataset_path = dataset_path;
  build_opts.M = 8;
  build_opts.ef_construction = 80;
  auto index_path = unique_temp_path("vortex_layer_membership.index");
  vortex::build_hnsw_index(build_opts, index_path);

  std::unique_ptr<faiss::Index> idx(faiss::read_index(index_path.string().c_str()));
  auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(idx.get());
  ctx.check(hnsw != nullptr, "layer-membership test index type");
  if (!hnsw) {
    std::filesystem::remove(dataset_path);
    std::filesystem::remove(index_path);
    return;
  }

  auto count_nodes = [&](int layer) {
    std::size_t count = 0;
    for (int level : hnsw->hnsw.levels) {
      if (level > layer) {
        ++count;
      }
    }
    return count;
  };

  int max_layer = hnsw->hnsw.max_level;
  for (int layer : {0, max_layer}) {
    vortex::ExtractNswOptions opts;
    opts.index_path = index_path;
    opts.target_skeleton = 1;
    opts.layer = layer;
    auto out = unique_temp_path("vortex_layer_membership_" + std::to_string(layer) + ".csr");
    auto csr = vortex::extract_nsw(opts, out);
    ctx.check(csr.layer == layer, "extract_nsw preserves requested layer");
    ctx.check(csr.node_ids.size() == count_nodes(layer),
              "extract_nsw node count matches FAISS layer membership");
    for (uint64_t id : csr.node_ids) {
      bool id_ok = (id < static_cast<uint64_t>(hnsw->hnsw.levels.size()));
      ctx.check(id_ok, "extract_nsw node id in bounds");
      if (id_ok) {
        ctx.check(hnsw->hnsw.levels[static_cast<std::size_t>(id)] > layer,
                  "extract_nsw node belongs to requested layer");
      }
    }
    std::filesystem::remove(out);
  }

  expect_throw(
      ctx,
      [&]() {
        vortex::ExtractNswOptions opts;
        opts.index_path = index_path;
        opts.target_skeleton = 1;
        opts.layer = -1;
        auto out = unique_temp_path("vortex_layer_membership_neg.csr");
        (void)vortex::extract_nsw(opts, out);
      },
      "extract_nsw rejects negative layer");

  expect_throw(
      ctx,
      [&]() {
        vortex::ExtractNswOptions opts;
        opts.index_path = index_path;
        opts.target_skeleton = 1;
        opts.layer = max_layer + 1;
        auto out = unique_temp_path("vortex_layer_membership_high.csr");
        (void)vortex::extract_nsw(opts, out);
      },
      "extract_nsw rejects layer above max_layer");

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(index_path);
}

void test_train_rejects_non_l2_csr_metric(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(2, 4);
  storage.values = {0.0f, 0.1f, 0.2f, 0.3f,
                    0.4f, 0.5f, 0.6f, 0.7f};
  auto dataset_path = unique_temp_path("vortex_non_l2_metric.fvecs");
  vector_io::write_fvecs(dataset_path.string(), storage);

  vortex::NswCsr csr;
  csr.dim = storage.dim;
  csr.metric = static_cast<vortex::Metric>(999);
  csr.layer = 0;
  csr.node_ids = {0, 1};
  csr.offsets = {0, 0, 0};
  csr.neighbors = {};
  auto csr_path = unique_temp_path("vortex_non_l2_metric.csr");
  csr.write(csr_path);

  vortex::TrainOptions opts;
  opts.dataset_path = dataset_path;
  opts.nsw_path = csr_path;
  opts.K = 1;
  opts.hash_bits = 32;
  opts.cdf_model_spec = "linear,linear";
  opts.cdf_branching_factor = 2;
  opts.centroid_knn = 1;
  opts.seed = 1;
  opts.threads = 1;

  expect_throw(
      ctx,
      [&]() { (void)vortex::train_vortex(opts); },
      "train_vortex rejects non-L2 CSR metric");

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(csr_path);
}

void test_train_duplicate_distances(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(4, 128);
  storage.values.assign(storage.count * storage.dim, 0.0f);
  auto dataset_path = unique_temp_path("vortex_duplicate_distances.fvecs");
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
  auto csr_path = unique_temp_path("vortex_duplicate_distances.csr");
  csr.write(csr_path);

  vortex::TrainOptions opts;
  opts.dataset_path = dataset_path;
  opts.nsw_path = csr_path;
  opts.K = 1;
  opts.hash_bits = 32;
  opts.cdf_model_spec = "linear,linear";
  opts.cdf_branching_factor = 32;
  opts.centroid_knn = 1;
  opts.seed = 9;
  opts.threads = 1;

  bool threw = false;
  try {
    auto model = vortex::train_vortex(opts);
    float query[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto hash = model.hash(query).to_u64();
    ctx.check(hash < (1ULL << 32), "duplicate-distance training hash in range");
  } catch (const std::exception&) {
    threw = true;
  }
  ctx.check(!threw, "train_vortex handles duplicated distances");

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(csr_path);
}

void test_train_uses_full_dataset_assignments_by_default(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(2, 8);
  storage.values = {
      0.0f, 0.0f,
      0.0f, 0.1f,
      0.1f, 0.0f,
      0.1f, 0.1f,
      10.0f, 10.0f,
      10.0f, 10.1f,
      10.1f, 10.0f,
      10.1f, 10.1f,
  };
  auto dataset_path = unique_temp_path("vortex_full_assignment.fvecs");
  vector_io::write_fvecs(dataset_path.string(), storage);

  vortex::NswCsr csr;
  csr.dim = storage.dim;
  csr.metric = vortex::Metric::L2;
  csr.layer = 1;
  csr.node_ids = {0, 1, 4, 5};
  csr.offsets.assign(csr.node_ids.size() + 1, 0);
  auto csr_path = unique_temp_path("vortex_full_assignment.csr");
  csr.write(csr_path);

  vortex::TrainOptions opts;
  opts.dataset_path = dataset_path;
  opts.nsw_path = csr_path;
  opts.K = 2;
  opts.hash_bits = 32;
  opts.cdf_model_spec = "linear,linear";
  opts.cdf_branching_factor = 2;
  opts.centroid_knn = 1;
  opts.seed = 5;
  opts.threads = 1;

  auto full_model = vortex::train_vortex(opts);
  uint64_t full_mass = 0;
  for (uint64_t mass : full_model.mass) {
    full_mass += mass;
  }
  ctx.check(full_mass == storage.count,
            "default training assigns the full dataset for mass/CDF fitting");

  opts.assignment_sample_limit = 6;
  auto sampled_model = vortex::train_vortex(opts);
  uint64_t sampled_mass = 0;
  for (uint64_t mass : sampled_model.mass) {
    sampled_mass += mass;
  }
  ctx.check(sampled_mass == 6,
            "assignment sample limit bounds full-dataset mass/CDF fitting");

  opts.assignment_sample_limit = 0;
  opts.use_full_dataset_for_assignments = false;
  auto skeleton_model = vortex::train_vortex(opts);
  uint64_t skeleton_mass = 0;
  for (uint64_t mass : skeleton_model.mass) {
    skeleton_mass += mass;
  }
  ctx.check(skeleton_mass == csr.node_ids.size(),
            "legacy training flag restricts mass/CDF fitting to skeleton vectors");

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(csr_path);
}

void test_graph_centroid_order_base_cache_payload(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(2, 4);
  storage.values = {
      0.0f, 0.0f,
      0.1f, 0.0f,
      10.0f, 0.0f,
      10.1f, 0.0f,
  };

  vortex::NswCsr csr;
  csr.dim = storage.dim;
  csr.metric = vortex::Metric::L2;
  csr.layer = 0;
  csr.node_ids = {0, 1, 2, 3};
  csr.offsets = {0, 2, 4, 6, 8};
  csr.neighbors = {
      1, 2,
      0, 3,
      0, 3,
      1, 2,
  };

  vortex::TrainOptions opts;
  opts.K = 2;
  opts.hash_bits = 32;
  opts.cdf_model_spec = "linear,linear";
  opts.cdf_branching_factor = 2;
  opts.centroid_knn = 1;
  opts.seed = 11;
  opts.threads = 1;
  opts.use_full_dataset_for_assignments = false;

  auto enabled_base =
      vortex::training_internal::build_vortex_training_base(opts, storage, csr);
  ctx.check(enabled_base->centroid_graph_offsets.size() == opts.K + 1,
            "graph centroid ordering records per-centroid offsets");
  ctx.check(!enabled_base->centroid_graph_neighbors.empty(),
            "graph centroid ordering records sparse centroid affinities");
  auto enabled_order =
      vortex::training_internal::build_vortex_centroid_order(opts, *enabled_base);
  ctx.check(enabled_order.size() == opts.K,
            "graph centroid ordering produces a complete centroid order");

  opts.enable_graph_centroid_order = false;
  auto disabled_base =
      vortex::training_internal::build_vortex_training_base(opts, storage, csr);
  ctx.check(disabled_base->centroid_graph_offsets.empty(),
            "legacy centroid ordering does not retain graph-affinity payload");
  auto disabled_order =
      vortex::training_internal::build_vortex_centroid_order(opts, *disabled_base);
  ctx.check(disabled_order.size() == opts.K,
            "legacy centroid ordering still produces a complete centroid order");
}

void test_train_with_2opt(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(64, 16);
  storage.values.resize(storage.count * storage.dim);
  std::mt19937 rng(19);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : storage.values) {
    v = dist(rng);
  }
  auto dataset_path = unique_temp_path("vortex_train_2opt.fvecs");
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
  auto csr_path = unique_temp_path("vortex_train_2opt.csr");
  csr.write(csr_path);

  vortex::TrainOptions opts;
  opts.dataset_path = dataset_path;
  opts.nsw_path = csr_path;
  opts.K = 8;
  opts.hash_bits = 32;
  opts.cdf_model_spec = "linear,linear";
  opts.cdf_branching_factor = 4;
  opts.centroid_knn = 8;
  opts.enable_2opt = true;
  opts.two_opt_iterations = 4;
  opts.seed = 13;
  opts.threads = 1;

  bool threw = false;
  try {
    auto model = vortex::train_vortex(opts);
    ctx.check(model.order.size() == 8, "2-opt training produced full centroid order");
    auto hash = model.hash(storage.values.data()).to_u64();
    ctx.check(hash < (1ULL << 32), "2-opt training hash in range");
  } catch (const std::exception&) {
    threw = true;
  }
  ctx.check(!threw, "train_vortex succeeds with enable_2opt");

  std::filesystem::remove(dataset_path);
  std::filesystem::remove(csr_path);
}

void test_smoke(TestContext& ctx, const std::filesystem::path& dataset_path) {
  if (!std::filesystem::exists(dataset_path)) {
    std::cerr << "[SKIP] dataset not found for smoke test\n";
    return;
  }

  vortex::BuildHnswOptions build_opts;
  build_opts.dataset_path = dataset_path;
  build_opts.M = 8;
  build_opts.ef_construction = 80;
  auto index_path = unique_temp_path("vortex_smoke.index");
  vortex::build_hnsw_index(build_opts, index_path);

  vortex::ExtractNswOptions extract_opts;
  extract_opts.index_path = index_path;
  extract_opts.target_skeleton = 500;
  auto nsw_path = unique_temp_path("vortex_smoke.csr");
  vortex::extract_nsw(extract_opts, nsw_path);

  vortex::TrainOptions train_opts;
  train_opts.dataset_path = dataset_path;
  train_opts.nsw_path = nsw_path;
  train_opts.K = 8;
  train_opts.hash_bits = 32;
  train_opts.cdf_model_spec = "linear,linear";
  train_opts.cdf_branching_factor = 4;
  train_opts.centroid_knn = 8;
  train_opts.seed = 123;
  train_opts.threads = 1;

  auto model = vortex::train_vortex(train_opts);

  auto data = vector_io::read_fvecs(dataset_path.string());
  std::size_t count = std::min<std::size_t>(100, data.count);
  for (std::size_t i = 0; i < count; ++i) {
    const float* vec = data.values.data() + i * data.dim;
    auto hash = model.hash(vec);
    uint64_t value = hash.to_u64();
    ctx.check(value < (1ULL << 32), "Hash out of range");
  }

  std::filesystem::remove(index_path);
  std::filesystem::remove(nsw_path);
}

} // namespace

int main() {
  TestContext ctx;

  test_train_option_validation(ctx);
  test_train_empty_skeleton(ctx);
  test_extract_nsw_layer_membership(ctx);
  test_train_rejects_non_l2_csr_metric(ctx);
  test_train_duplicate_distances(ctx);
  test_train_uses_full_dataset_assignments_by_default(ctx);
  test_graph_centroid_order_base_cache_payload(ctx);
  test_train_with_2opt(ctx);
  auto root = project_root();
  auto dataset = root / "vector_datasets" / "siftsmall" / "siftsmall_base.fvecs";
  test_model_determinism(ctx, dataset);
  test_smoke(ctx, dataset);

  if (ctx.failures == 0) {
    std::cerr << "All vortex_v1 training tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " tests failed\n";
  return 1;
}
