#include "vortex_test_common.h"

#include "vortex_v1/codegen.h"
#include "vortex_v1/codegen_loader.h"

#include "rm_model/json.h"

using namespace vortex_test;

namespace {

std::size_t count_occurrences(const std::string& text, const std::string& needle) {
  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

void test_codegen_output(TestContext& ctx) {
  vortex::VortexModel model;
  model.dim = 2;
  model.metric = vortex::Metric::L2;
  model.hash_bits = 64;
  model.centroids = {0.0f, 0.0f};
  model.order = {0};
  model.mass = {10};
  model.range_start.resize(1);
  model.range_size.resize(1);
  model.range_start[0].words = {0, 0, 0, 0};
  model.range_size[0].words = {100, 0, 0, 0};
  model.range_full = {0};
  model.min_dist = {0.0};
  model.max_dist = {1.0};
  model.cdf_top_type = vortex::CdfModelType::Linear;
  model.cdf_leaf_type = vortex::CdfModelType::Linear;
  model.cdf_leaf_count = 2;
  model.cdf_top_param_count = vortex::cdf_param_count(model.cdf_top_type);
  model.cdf_leaf_param_count = vortex::cdf_param_count(model.cdf_leaf_type);
  model.cdf_rows = {2};
  model.cdf_top_params = {0.0, 0.0};
  model.cdf_leaf_params = {0.0, 1.0, 0.0, 1.0};
  model.compute_active();

  auto dir = unique_temp_path("vortex_codegen_test");
  std::filesystem::remove_all(dir);

  vortex::CodegenOptions opts;
  opts.output_dir = dir;
  opts.name = "vortex_codegen_test";
  ScopedEnvVar vortex_root("VORTEX_ROOT", project_root().string());
  auto result = vortex::emit_codegen(model, opts);

  ctx.check(std::filesystem::exists(result.manifest_path), "codegen manifest exists");
  ctx.check(std::filesystem::exists(result.model_path), "codegen model exists");
  ctx.check(std::filesystem::exists(result.centroid_index_path), "codegen centroid index exists");
  ctx.check(std::filesystem::exists(result.include_path), "codegen header exists");
  ctx.check(std::filesystem::exists(result.src_path), "codegen source exists");
  ctx.check(std::filesystem::exists(dir / "codegen" / "README.md"),
            "codegen README exists");
  ctx.check(std::filesystem::exists(dir / "codegen" / "runtime" / "src" / "model.cpp"),
            "codegen runtime model exists");
  ctx.check(std::filesystem::exists(dir / "codegen" / "runtime" / "src" / "uint256.cpp"),
            "codegen runtime shared UInt256 source exists");
  ctx.check(std::filesystem::exists(dir / "codegen" / "runtime" / "include" / "rm_model" /
                                    "uint256.h"),
            "codegen runtime shared UInt256 header exists");

  std::ifstream header_in(result.include_path);
  std::ifstream source_in(result.src_path);
  std::string header_text((std::istreambuf_iterator<char>(header_in)),
                          std::istreambuf_iterator<char>());
  std::string source_text((std::istreambuf_iterator<char>(source_in)),
                          std::istreambuf_iterator<char>());
  ctx.check(header_text.find("_infer_load_ex") != std::string::npos,
            "codegen header exposes infer_load_ex");
  ctx.check(header_text.find("_infer_load_ex2") != std::string::npos,
            "codegen header exposes infer_load_ex2");
  ctx.check(header_text.find("_infer_ann_query") != std::string::npos,
            "codegen header exposes infer_ann_query");
  ctx.check(header_text.find("_infer_ann_query_cached") != std::string::npos,
            "codegen header exposes infer_ann_query_cached");
  ctx.check(header_text.find("_infer_nearest_active_centroid") != std::string::npos,
            "codegen header exposes infer_nearest_active_centroid");
  ctx.check(header_text.find("_infer_nearest_active_centroid(\n"
                             "    const float* vector,\n"
                             "    uint32_t* out_centroid,\n"
                             "    float* out_dist2,\n"
                             "    uint64_t* out_cache_token)") != std::string::npos,
            "codegen header emits valid nearest_active_centroid C ABI signature");
  ctx.check(header_text.find("float* out_dist2,\n"
                             "    float* out_dist2") == std::string::npos,
            "codegen header does not duplicate out_dist2 parameters");
  ctx.check(header_text.find("_infer_hash_with_centroid") != std::string::npos,
            "codegen header exposes infer_hash_with_centroid");
  ctx.check(header_text.find("ann_query(const float* vector") != std::string::npos,
            "codegen header exposes C++ ann_query API");
  ctx.check(header_text.find("hash_with_centroid(const float* vector") != std::string::npos,
            "codegen header exposes C++ hash_with_centroid API");
  ctx.check(header_text.find("nearest_active_centroid(const float* vector") != std::string::npos,
            "codegen header exposes C++ nearest_active_centroid API");
  ctx.check(header_text.find("load_centroid_index(") != std::string::npos,
            "codegen header exposes centroid index loader");
  ctx.check(header_text.find("populate_centroid_buffers(") != std::string::npos,
            "codegen header declares shared centroid-buffer helper");
  ctx.check(header_text.find("prepare_query_pivot_l2(") != std::string::npos,
            "codegen header declares shared query-pivot helper");
  ctx.check(header_text.find("AlignedVector<float> active_vectors_") != std::string::npos,
            "codegen header stores active centroids in aligned contiguous buffers");
  ctx.check(header_text.find("AlignedVector<float> index_pivot_vectors_") != std::string::npos,
            "codegen header stores pivot centroids in aligned contiguous buffers");
  ctx.check(header_text.find("scratch_query_pivot_l2_") == std::string::npos,
            "codegen header avoids shared scratch pivot buffer");
  ctx.check(source_text.find("build_centroid_index") != std::string::npos,
            "codegen source builds centroid index");
  ctx.check(source_text.find("load_centroid_index(index_path") != std::string::npos,
            "codegen source loads serialized centroid index");
  ctx.check(source_text.find("thread_local QueryPivotCacheState state") != std::string::npos,
            "codegen source uses thread-local query pivot cache state");
  ctx.check(source_text.find("query_cache_enabled_") != std::string::npos,
            "codegen source includes query cache controls");
  ctx.check(source_text.find("should_consider_centroid_index()") != std::string::npos,
            "codegen source includes adaptive index materialization heuristic");
  ctx.check(source_text.find("use_centroid_index_for_queries()") != std::string::npos,
            "codegen source includes adaptive direct-scan/index query heuristic");
  ctx.check(source_text.find("invalidate_query_pivot_cache(this)") != std::string::npos,
            "codegen source invalidates query cache on index/model lifecycle changes");
  ctx.check(source_text.find("infer_ann_query_cached") != std::string::npos,
            "codegen source exports cached ann_query C ABI");
  ctx.check(source_text.find("lower_bound2 >= best_d2") != std::string::npos,
            "codegen source early-breaks centroid bound loop");
  ctx.check(source_text.find("dot_product(") != std::string::npos,
            "codegen source uses dot-product distance kernel");
  ctx.check(source_text.find("cache_state.query_ptr == vector") != std::string::npos,
            "codegen source ties query cache token reuse to vector identity");
  ctx.check(source_text.find("cache_state.query_norm2 == query_norm2") != std::string::npos,
            "codegen source ties query cache token reuse to query norm");
  ctx.check(source_text.find("bool query_cache_allowed = query_cache_enabled_ &&") !=
                std::string::npos,
            "codegen source skips cache-token bookkeeping for plain hash calls");
  ctx.check(count_occurrences(source_text, "prepare_query_pivot_l2(") == 3,
            "codegen source centralizes query-pivot cache preparation");
  ctx.check(count_occurrences(source_text, "cache_state.pivot_l2.resize") == 1,
            "codegen source emits one query-pivot resize path");
  ctx.check(count_occurrences(source_text, "populate_centroid_buffers(") == 5,
            "codegen source centralizes centroid buffer population");
  ctx.check(source_text.find("l2_dist2_from_dot(query_norm2, index_active_norm2_[i], dot_qc)") !=
                std::string::npos,
            "codegen source uses centroid norms in exact re-ranking");
  ctx.check(source_text.find("cache_state.ann_full_scan") != std::string::npos,
            "codegen source reuses ANN full-scan scratch buffer");
  ctx.check(source_text.find("cache_state.ann_heap") != std::string::npos,
            "codegen source reuses ANN heap scratch buffer");
  ctx.check(source_text.find("std::priority_queue") == std::string::npos,
            "codegen source avoids per-query priority_queue allocations");

  auto manifest = rm_model::json::parse_file(result.manifest_path.string());
  ctx.check(manifest.is_object(), "codegen manifest is object");
  if (manifest.is_object()) {
    const auto* name = manifest.find("name");
    ctx.check(name && name->is_string(), "codegen manifest has name");
    const auto* paths = manifest.find("paths");
    ctx.check(paths && paths->is_object(), "codegen manifest has paths");
    if (paths && paths->is_object()) {
      const auto* centroid_index = paths->find("centroid_index");
      ctx.check(centroid_index && centroid_index->is_string(),
                "codegen manifest includes centroid_index path");
    }
  }

  auto loaded = vortex::VortexModel::read(result.model_path);
  float vec[2] = {0.5f, 0.0f};
  ctx.check(model.hash(vec).to_u64() == loaded.hash(vec).to_u64(), "codegen model roundtrip");

  std::filesystem::remove_all(dir);
}

void test_codegen_artifact_golden_stability(TestContext& ctx) {
  auto model = make_golden_model();
  auto dir_a = unique_temp_path("vortex_codegen_golden_a");
  auto dir_b = unique_temp_path("vortex_codegen_golden_b");
  std::filesystem::remove_all(dir_a);
  std::filesystem::remove_all(dir_b);

  vortex::CodegenOptions opts_a;
  opts_a.output_dir = dir_a;
  opts_a.name = "vortex_codegen_golden";

  vortex::CodegenOptions opts_b;
  opts_b.output_dir = dir_b;
  opts_b.name = "vortex_codegen_golden";

  ScopedEnvVar vortex_root("VORTEX_ROOT", project_root().string());
  auto result_a = vortex::emit_codegen(model, opts_a);
  auto result_b = vortex::emit_codegen(model, opts_b);

  check_same_bytes(ctx, result_a.model_path, result_b.model_path,
                   "codegen serialized model is byte-for-byte stable");
  check_same_bytes(ctx, result_a.centroid_index_path, result_b.centroid_index_path,
                   "codegen centroid index is byte-for-byte stable");
  check_same_bytes(ctx, result_a.include_path, result_b.include_path,
                   "codegen header is byte-for-byte stable");
  check_same_bytes(ctx, result_a.src_path, result_b.src_path,
                   "codegen source is byte-for-byte stable");
  ctx.check(std::filesystem::file_size(result_a.model_path) == 519,
            "codegen model byte size is stable");
  ctx.check(std::filesystem::file_size(result_a.centroid_index_path) == 88,
            "codegen centroid index byte size is stable");

  auto loaded = vortex::VortexModel::read(result_a.model_path);
  float query[2] = {2.5f, 0.0f};
  ctx.check(loaded.hash(query).to_u64() == 1536,
            "codegen model artifact preserves golden hash output");

  std::filesystem::remove_all(dir_a);
  std::filesystem::remove_all(dir_b);
}

void test_codegen_loader_hash_stability(TestContext& ctx) {
  vortex::VortexModel model;
  model.dim = 2;
  model.metric = vortex::Metric::L2;
  model.hash_bits = 64;
  model.centroids = {0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 3.0f};
  model.order = {0, 1, 2};
  model.mass = {20, 30, 50};
  model.range_start.resize(3);
  model.range_size.resize(3);
  model.range_start[0].words = {0, 0, 0, 0};
  model.range_size[0].words = {200, 0, 0, 0};
  model.range_start[1].words = {200, 0, 0, 0};
  model.range_size[1].words = {300, 0, 0, 0};
  model.range_start[2].words = {500, 0, 0, 0};
  model.range_size[2].words = {500, 0, 0, 0};
  model.range_full = {0, 0, 0};
  model.min_dist = {0.0, 0.0, 0.0};
  model.max_dist = {25.0, 25.0, 25.0};
  model.cdf_top_type = vortex::CdfModelType::Linear;
  model.cdf_leaf_type = vortex::CdfModelType::Linear;
  model.cdf_leaf_count = 2;
  model.cdf_top_param_count = vortex::cdf_param_count(model.cdf_top_type);
  model.cdf_leaf_param_count = vortex::cdf_param_count(model.cdf_leaf_type);
  model.cdf_rows = {2, 2, 2};
  model.cdf_top_params = {
      0.0, 1.0 / 25.0,
      0.0, 1.0 / 25.0,
      0.0, 1.0 / 25.0,
  };
  model.cdf_leaf_params = {
      0.0, 1.0 / 25.0, 0.0, 1.0 / 25.0,
      0.0, 1.0 / 25.0, 0.0, 1.0 / 25.0,
      0.0, 1.0 / 25.0, 0.0, 1.0 / 25.0,
  };
  model.compute_active();

  auto dir = unique_temp_path("vortex_codegen_loader_test");
  std::filesystem::remove_all(dir);

  vortex::CodegenOptions opts;
  opts.output_dir = dir;
  opts.name = "vortex_codegen_loader_test";
  ScopedEnvVar vortex_root("VORTEX_ROOT", project_root().string());
  (void)vortex::emit_codegen(model, opts);
  auto build_cache_dir = unique_temp_path("vortex_codegen_loader_cache");
  std::filesystem::remove_all(build_cache_dir);
  ScopedEnvVar codegen_build_cache("VORTEX_CODEGEN_BUILD_CACHE", build_cache_dir.string());

  std::vector<float> queries = {
      0.2f, 0.1f,
      2.9f, 0.2f,
      0.1f, 2.8f,
      1.0f, 1.0f,
      2.0f, 1.0f,
  };

  vortex::CodegenHasher hasher;
  vortex::CodegenLoadOptions load_on;
  load_on.enable_centroid_index = true;
  load_on.centroid_index_pivots = 16;
  hasher.load(dir, load_on);
  ctx.check(hasher.library_status() == "cache_miss_built",
            "codegen loader populates build cache on first load");

  for (std::size_t i = 0; i < queries.size() / model.dim; ++i) {
    const float* vec = queries.data() + i * model.dim;
    uint64_t expected = model.hash(vec).to_u64();

    bool ok_on = false;
    uint64_t got_on = hasher.hash64(vec, &ok_on);
    ctx.check(ok_on, "codegen loader hash64 succeeds with centroid index enabled");
    ctx.check(got_on == expected, "codegen loader hash64 matches model hash (index enabled)");

    vortex::CodegenHashCentroidResult with_centroid{};
    ctx.check(hasher.hash_with_centroid(vec, &with_centroid),
              "codegen loader hash_with_centroid succeeds");
    ctx.check(with_centroid.hash.words[0] == expected,
              "codegen loader hash_with_centroid hash matches model hash");

    uint32_t centroid = std::numeric_limits<uint32_t>::max();
    float dist2 = -1.0f;
    uint64_t cache_token = 0;
    ctx.check(hasher.nearest_active_centroid(vec, &centroid, &dist2, &cache_token),
              "codegen loader nearest_active_centroid succeeds");
    auto nearest = model.router.nearest(vec);
    ctx.check(centroid == nearest.centroid,
              "codegen loader nearest_active_centroid centroid matches model router");
    ctx.check(std::abs(static_cast<double>(dist2) - nearest.dist2) < 1e-4,
              "codegen loader nearest_active_centroid dist2 matches model router");
    ctx.check(cache_token != 0,
              "codegen loader nearest_active_centroid returns non-zero cache token");
  }

  std::vector<vortex::CodegenAnnNeighbor> ann;
  ctx.check(hasher.ann_query(queries.data(), 3, &ann),
            "codegen loader ann_query succeeds with centroid index enabled");
  if (!ann.empty()) {
    auto nearest = model.router.nearest(queries.data());
    ctx.check(ann.front().centroid == nearest.centroid,
              "codegen ann_query top-1 centroid matches model router");
    ctx.check(std::abs(static_cast<double>(ann.front().dist2) - nearest.dist2) < 1e-4,
              "codegen ann_query top-1 dist2 matches model router");
  }
  vortex::CodegenHashCentroidResult cached_seed{};
  ctx.check(hasher.hash_with_centroid(queries.data(), &cached_seed),
            "codegen loader hash_with_centroid provides cache token for ann_query");
  std::vector<vortex::CodegenAnnNeighbor> ann_cached;
  ctx.check(hasher.ann_query(queries.data(), 3, &ann_cached, cached_seed.cache_token),
            "codegen loader ann_query with cache token succeeds");
  ctx.check(ann_cached.size() == ann.size(),
            "codegen loader ann_query with cache token returns same result count");
  if (!ann.empty() && !ann_cached.empty()) {
    ctx.check(ann_cached.front().centroid == ann.front().centroid,
              "codegen loader ann_query cache top-1 centroid matches regular ann_query");
    ctx.check(std::abs(static_cast<double>(ann_cached.front().dist2) -
                       static_cast<double>(ann.front().dist2)) < 1e-4,
              "codegen loader ann_query cache top-1 dist2 matches regular ann_query");
  }

  hasher.unload();
  std::filesystem::remove_all(dir / "codegen" / "build");
  hasher.load(dir, load_on);
  ctx.check(hasher.library_status() == "cache_hit",
            "codegen loader restores generated library from build cache");
  bool ok_cached = false;
  ctx.check(hasher.hash64(queries.data(), &ok_cached) == model.hash(queries.data()).to_u64() &&
                ok_cached,
            "codegen loader cache-hit library preserves hash output");

  hasher.unload();
  vortex::CodegenLoadOptions load_off;
  load_off.enable_centroid_index = false;
  hasher.load(dir, load_off);
  for (std::size_t i = 0; i < queries.size() / model.dim; ++i) {
    const float* vec = queries.data() + i * model.dim;
    uint64_t expected = model.hash(vec).to_u64();
    bool ok_off = false;
    uint64_t got_off = hasher.hash64(vec, &ok_off);
    ctx.check(ok_off, "codegen loader hash64 succeeds with centroid index disabled");
    ctx.check(got_off == expected, "codegen loader hash64 matches model hash (index disabled)");
  }

  std::filesystem::remove_all(dir);
  std::filesystem::remove_all(build_cache_dir);
}

} // namespace

int main() {
  TestContext ctx;

  test_codegen_output(ctx);
  test_codegen_artifact_golden_stability(ctx);
  test_codegen_loader_hash_stability(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All vortex_v1 codegen tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " tests failed\n";
  return 1;
}
