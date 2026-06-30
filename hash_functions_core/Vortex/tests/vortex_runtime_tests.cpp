#include "vortex_test_common.h"

#include "vortex_v1/nsw_csr.h"
#include "vortex_v1/training.h"
#include "vortex_v1/types.h"
#include "vortex_v1/uint256.h"

#include "rm_model/centroid_router.h"
#include "rm_model/uint256.h"
#include "vector_io/vector_io.hpp"

#include <type_traits>

using namespace vortex_test;

namespace {

static_assert(std::is_same_v<vortex::UInt256, rm_model::UInt256>,
              "Vortex UInt256 must remain a thin adapter over the shared type");
static_assert(std::is_same_v<vortex::UInt320, rm_model::UInt320>,
              "Vortex UInt320 must remain a thin adapter over the shared type");

void test_csr_roundtrip(TestContext& ctx) {
  vortex::NswCsr csr;
  csr.dim = 4;
  csr.metric = vortex::Metric::L2;
  csr.layer = 2;
  csr.node_ids = {5, 7, 9};
  csr.offsets = {0, 2, 3, 3};
  csr.neighbors = {1, 2, 0};

  auto path = unique_temp_path("vortex_csr_test.bin");
  csr.write(path);
  auto loaded = vortex::NswCsr::read(path);

  ctx.check(loaded.dim == csr.dim, "CSR dim mismatch");
  ctx.check(loaded.layer == csr.layer, "CSR layer mismatch");
  ctx.check(loaded.node_ids == csr.node_ids, "CSR node ids mismatch");
  ctx.check(loaded.offsets == csr.offsets, "CSR offsets mismatch");
  ctx.check(loaded.neighbors == csr.neighbors, "CSR neighbors mismatch");

  std::filesystem::remove(path);
}

void test_metric_parse(TestContext& ctx) {
  ctx.check(vortex::parse_metric("L2") == vortex::Metric::L2, "parse_metric L2");
  ctx.check(vortex::parse_metric("l2") == vortex::Metric::L2, "parse_metric l2");
  bool threw = false;
  try {
    vortex::parse_metric("ip");
  } catch (const std::exception&) {
    threw = true;
  }
  ctx.check(threw, "parse_metric rejects unsupported metric");
}

void test_uint256_ops(TestContext& ctx) {
  auto a = vortex::UInt256::from_u64(5);
  auto b = vortex::UInt256::from_u64(7);
  auto sum = vortex::add(a, b);
  ctx.check(vortex::to_u64(sum, 64) == 12, "UInt256 add");

  auto diff = vortex::sub(b, a);
  ctx.check(vortex::to_u64(diff, 64) == 2, "UInt256 sub");

  auto shifted = vortex::UInt256::one_shifted(64);
  ctx.check(shifted.words[1] == 1 && shifted.words[0] == 0, "UInt256 one_shifted 64");
  auto shifted255 = vortex::UInt256::one_shifted(255);
  ctx.check(shifted255.words[3] == (1ULL << 63), "UInt256 one_shifted 255");
  ctx.check(vortex::UInt256::one_shifted(256).is_zero(),
            "UInt256 one_shifted 256 wraps to zero sentinel");
  bool threw = false;
  try {
    (void)vortex::UInt256::one_shifted(257);
  } catch (const std::exception&) {
    threw = true;
  }
  ctx.check(threw, "UInt256 one_shifted rejects >256");

  auto max4 = vortex::UInt256::max_for_bits(4);
  ctx.check(vortex::to_u64(max4, 64) == 0xF, "UInt256 max_for_bits");
  auto max65 = vortex::UInt256::max_for_bits(65);
  ctx.check(max65.words[0] == UINT64_MAX && max65.words[1] == 1,
            "UInt256 max_for_bits crosses word boundary");

  vortex::UInt256 masked = vortex::UInt256::max_for_bits(256);
  masked.mask_bits(4);
  ctx.check(vortex::to_u64(masked, 64) == 0xF, "UInt256 mask_bits");

  auto mul = vortex::mul_u64(a, 10, nullptr);
  ctx.check(vortex::to_u64(mul, 64) == 50, "UInt256 mul_u64");

  uint64_t rem = 0;
  auto div = vortex::div_u64(mul, 5, &rem);
  ctx.check(vortex::to_u64(div, 64) == 10 && rem == 0, "UInt256 div_u64");

  vortex::UInt256 carry_a{};
  carry_a.words = {UINT64_MAX, 0, 0, 0};
  auto carry_sum = vortex::add_u64(carry_a, 1);
  ctx.check(carry_sum.words[0] == 0 && carry_sum.words[1] == 1,
            "UInt256 add_u64 carries across words");

  auto borrowed = vortex::sub(carry_sum, vortex::UInt256::from_u64(1));
  ctx.check(borrowed.words[0] == UINT64_MAX && borrowed.words[1] == 0,
            "UInt256 sub borrows across words");
  ctx.check(vortex::compare(carry_sum, borrowed) > 0, "UInt256 compare high words");

  auto shifted320 = vortex::shl_u64_to_320(1, 128);
  ctx.check(shifted320.words[2] == 1, "UInt320 shl_u64_to_320 word shift");
  auto div320 = vortex::div_u64(shifted320, 2, &rem);
  ctx.check(div320.words[1] == (1ULL << 63) && rem == 0,
            "UInt320 div_u64 returns low 256-bit quotient");

  ctx.check(vortex::to_hex(max65, 65) == "1ffffffffffffffff",
            "UInt256 to_hex preserves requested bit width");
}

void test_vector_io_roundtrip(TestContext& ctx) {
  vector_io::VectorStorage<float> storage(3, 2);
  storage.values = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 4.0f};
  auto path = unique_temp_path("vortex_vecs.fvecs");
  vector_io::write_fvecs(path.string(), storage);
  auto loaded = vector_io::read_fvecs(path.string());
  ctx.check(loaded.dim == storage.dim, "vector_io dim");
  ctx.check(loaded.count == storage.count, "vector_io count");
  ctx.check(loaded.values == storage.values, "vector_io data");
  std::filesystem::remove(path);
}

void test_range_correctness(TestContext& ctx) {
  auto to_uint256 = [](const rm_model::CentroidRouter::Range256& range) {
    vortex::UInt256 out{};
    out.words = range.words;
    return out;
  };
  std::vector<uint64_t> mass = {10, 20, 30};
  std::vector<uint32_t> order = {0, 1, 2};
  auto ranges = vortex::allocate_ranges(mass, order, 32);

  vortex::UInt256 sum = vortex::UInt256::zero();
  for (std::size_t i = 0; i < mass.size(); ++i) {
    sum = vortex::add(sum, to_uint256(ranges.range_size[i]));
  }

  auto expected = vortex::UInt256::one_shifted(32);
  ctx.check(vortex::compare(sum, expected) == 0, "Range sum should equal 2^hash_bits");

  for (std::size_t i = 0; i < mass.size(); ++i) {
    uint64_t start = vortex::to_u64(to_uint256(ranges.range_start[i]), 32);
    uint64_t size = vortex::to_u64(to_uint256(ranges.range_size[i]), 32);
    ctx.check(start + size >= start, "Range overflow check");
    ctx.check(start + size <= (1ULL << 32), "Range end within H");
  }
}

void test_allocate_ranges_256(TestContext& ctx) {
  std::vector<uint64_t> mass = {100};
  std::vector<uint32_t> order = {0};
  auto ranges = vortex::allocate_ranges(mass, order, 256);
  ctx.check(ranges.range_full.size() == 1, "range_full size");
  ctx.check(ranges.range_full[0] == 1, "range_full set for full range");
}

void test_allocate_ranges_invalid_inputs(TestContext& ctx) {
  std::vector<uint64_t> mass = {10, 20, 30};
  expect_throw(
      ctx,
      [&]() {
        std::vector<uint32_t> bad_order = {0, 0, 2};
        (void)vortex::allocate_ranges(mass, bad_order, 32);
      },
      "allocate_ranges rejects duplicate order indices");

  expect_throw(
      ctx,
      [&]() {
        std::vector<uint32_t> bad_order = {0, 1, 3};
        (void)vortex::allocate_ranges(mass, bad_order, 32);
      },
      "allocate_ranges rejects out-of-range order indices");

  expect_throw(
      ctx,
      [&]() {
        std::vector<uint64_t> zero_mass = {0, 0, 0};
        std::vector<uint32_t> order = {0, 1, 2};
        (void)vortex::allocate_ranges(zero_mass, order, 32);
      },
      "allocate_ranges rejects zero total mass");

  expect_throw(
      ctx,
      [&]() {
        std::vector<uint64_t> overflow_mass = {std::numeric_limits<uint64_t>::max(), 1};
        std::vector<uint32_t> order = {0, 1};
        (void)vortex::allocate_ranges(overflow_mass, order, 32);
      },
      "allocate_ranges rejects total mass overflow");
}

void test_centroid_router(TestContext& ctx) {
  std::vector<float> centroids = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
  std::vector<rm_model::CentroidRouter::Range256> starts(3);
  std::vector<rm_model::CentroidRouter::Range256> sizes(3);
  std::vector<uint8_t> full(3, 0);
  sizes[0].words[0] = 1;
  sizes[1].words[0] = 1;
  sizes[2].words[0] = 1;

  rm_model::CentroidRouter router;
  router.set_centroids_view(2, centroids.data(), 3);
  router.set_ranges_view(64, starts.data(), sizes.data(), full.data(), 3);
  router.compute_active_from_ranges();
  ctx.check(router.active_centroids().size() == 3, "router active count");

  float query[2] = {0.9f, 0.1f};
  auto res = router.nearest(query);
  ctx.check(res.centroid == 1, "router nearest centroid");
  ctx.check(res.dist2 >= 0.0, "router dist2 non-negative");
}

void test_model_roundtrip(TestContext& ctx) {
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

  auto path = unique_temp_path("vortex_model.bin");
  model.write(path);
  auto loaded = vortex::VortexModel::read(path);

  float vec[2] = {0.5f, 0.0f};
  auto hash = loaded.hash(vec);
  ctx.check(hash.to_u64() < 100, "hash within range");
  ctx.check(loaded.hash_u64(vec) == hash.to_u64(),
            "hash_u64 matches generic hash output");
  auto nearest = loaded.router.nearest(vec);
  auto from_centroid = loaded.hash_from_centroid(nearest.centroid, nearest.dist2);
  ctx.check(hash.to_u64() == from_centroid.to_u64(),
            "hash_from_centroid matches direct hash");
  ctx.check(loaded.hash_from_centroid_u64(nearest.centroid, nearest.dist2) ==
                from_centroid.to_u64(),
            "hash_from_centroid_u64 matches generic centroid hash output");
  vortex::VortexModel full64 = loaded;
  full64.range_size[0].words = {0, 1, 0, 0};
  full64.compute_active();
  auto full64_hash = full64.hash(vec);
  ctx.check(full64.hash_u64(vec) == full64_hash.to_u64(),
            "hash_u64 matches generic hash for a full 64-bit range");
  auto full64_nearest = full64.router.nearest(vec);
  ctx.check(full64.hash_from_centroid_u64(full64_nearest.centroid,
                                          full64_nearest.dist2) ==
                full64.hash_from_centroid(full64_nearest.centroid,
                                          full64_nearest.dist2).to_u64(),
            "hash_from_centroid_u64 handles a full 64-bit range");
  expect_throw(
      ctx,
      [&]() {
        vortex::VortexModel inactive = loaded;
        inactive.range_size[0].words = {0, 0, 0, 0};
        inactive.range_full = {0};
        inactive.compute_active();
        (void)inactive.hash_from_centroid(0, 0.0);
      },
      "hash_from_centroid rejects model with no active centroids");

  vortex::VortexModel copy = loaded;
  auto hash2 = copy.hash(vec);
  ctx.check(hash.to_u64() == hash2.to_u64(), "copy hash matches");

  vortex::VortexModel moved = std::move(copy);
  auto hash3 = moved.hash(vec);
  ctx.check(hash.to_u64() == hash3.to_u64(), "move hash matches");

  std::filesystem::remove(path);
}

void test_model_hash_golden_outputs(TestContext& ctx) {
  auto model = make_golden_model();

  struct Case {
    std::array<float, 2> query;
    uint32_t centroid;
    double dist2;
    double pred;
    uint64_t hash;
    const char* hex;
  };

  const std::vector<Case> cases = {
      {{0.5f, 0.0f}, 0, 0.25, 0.25, 256, "0000000000000100"},
      {{2.5f, 0.0f}, 1, 0.25, 0.25, 1536, "0000000000000600"},
      {{0.0f, 2.75f}, 2, 0.5625, 0.5625, 6400, "0000000000001900"},
  };

  for (const auto& item : cases) {
    auto nearest = model.router.nearest(item.query.data());
    ctx.check(nearest.centroid == item.centroid, "golden nearest centroid is stable");
    ctx.check(std::abs(nearest.dist2 - item.dist2) < 1e-12,
              "golden nearest distance is stable");

    double pred = -1.0;
    auto hash = model.hash_with_pred(item.query.data(), &pred);
    ctx.check(std::abs(pred - item.pred) < 1e-12,
              "golden hash CDF prediction is stable");
    ctx.check(hash.to_u64() == item.hash, "golden hash64 output is stable");
    ctx.check(hash.to_hex() == item.hex, "golden hash hex output is stable");

    auto from_centroid = model.hash_from_centroid(item.centroid, item.dist2);
    ctx.check(from_centroid.to_u64() == item.hash,
              "golden hash_from_centroid output is stable");
    ctx.check(model.hash_u64(item.query.data()) == item.hash,
              "golden hash_u64 output is stable");
    ctx.check(model.hash_from_centroid_u64(item.centroid, item.dist2) == item.hash,
              "golden hash_from_centroid_u64 output is stable");
  }
}

void test_model_serialization_golden_stability(TestContext& ctx) {
  auto model = make_golden_model();
  auto path_a = unique_temp_path("vortex_model_golden_a.bin");
  auto path_b = unique_temp_path("vortex_model_golden_b.bin");

  model.write(path_a);
  model.write(path_b);

  check_same_bytes(ctx, path_a, path_b, "model serialization is byte-for-byte stable");
  ctx.check(std::filesystem::file_size(path_a) == 519,
            "model serialization byte size is stable");

  auto bytes = read_binary_file(path_a);
  ctx.check(bytes.size() >= 8, "model serialization has header");
  if (bytes.size() >= 8) {
    ctx.check(bytes[0] == 'V' && bytes[1] == 'T' && bytes[2] == 'X' &&
                  bytes[3] == 'M' && bytes[4] == '2',
              "model serialization magic is stable");
  }

  auto loaded = vortex::VortexModel::read(path_a);
  float query[2] = {0.0f, 2.75f};
  ctx.check(loaded.hash(query).to_u64() == 6400,
            "deserialized model preserves golden hash output");

  std::filesystem::remove(path_a);
  std::filesystem::remove(path_b);
}

void test_hash_bits_256(TestContext& ctx) {
  vortex::VortexModel model;
  model.dim = 2;
  model.metric = vortex::Metric::L2;
  model.hash_bits = 256;
  model.centroids = {0.0f, 0.0f};
  model.order = {0};
  model.mass = {10};
  model.range_start.resize(1);
  model.range_size.resize(1);
  model.range_full = {1};
  model.min_dist = {0.0};
  model.max_dist = {1.0};
  model.cdf_top_type = vortex::CdfModelType::Linear;
  model.cdf_leaf_type = vortex::CdfModelType::Linear;
  model.cdf_leaf_count = 2;
  model.cdf_top_param_count = vortex::cdf_param_count(model.cdf_top_type);
  model.cdf_leaf_param_count = vortex::cdf_param_count(model.cdf_leaf_type);
  model.cdf_rows = {2};
  model.cdf_top_params = {0.0, 0.0};
  model.cdf_leaf_params = {0.0, 0.0, 0.0, 0.0};
  model.compute_active();

  float vec[2] = {0.5f, 0.0f}; // dist2 = 0.25
  auto hash = model.hash(vec);

  long double value = 0.25L;
  long double scaled = std::ldexp(value, 64);
  uint64_t expected_q = static_cast<uint64_t>(scaled);
  if (expected_q == 0) {
    expected_q = 1;
  }
  if (hash.value.words[3] != expected_q) {
    std::cerr << "[INFO] hash_bits=256 expected_q=" << expected_q
              << " actual_q=" << hash.value.words[3] << "\n";
  }
  ctx.check(hash.value.words[3] == expected_q, "hash_bits=256 uses q64 in high word");
}

} // namespace

int main() {
  TestContext ctx;

  test_metric_parse(ctx);
  test_uint256_ops(ctx);
  test_csr_roundtrip(ctx);
  test_vector_io_roundtrip(ctx);
  test_range_correctness(ctx);
  test_allocate_ranges_256(ctx);
  test_allocate_ranges_invalid_inputs(ctx);
  test_centroid_router(ctx);
  test_model_roundtrip(ctx);
  test_model_hash_golden_outputs(ctx);
  test_model_serialization_golden_stability(ctx);
  test_hash_bits_256(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All vortex_v1 runtime tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " tests failed\n";
  return 1;
}
