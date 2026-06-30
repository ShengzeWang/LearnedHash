#include "learnedhash_test_common.h"

#include "lead/hasher.h"
#include "lead/mapping.h"
#include "lead/trainer.h"
#include "lead/uint256.h"

#include "rm_model/uint256.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace {

static_assert(std::is_same_v<lead::UInt256, rm_model::UInt256>,
              "LEAD UInt256 must remain a thin adapter over the shared type");
static_assert(std::is_same_v<lead::UInt320, rm_model::UInt320>,
              "LEAD UInt320 must remain a thin adapter over the shared type");

using learnedhash_test::TestContext;
using learnedhash_test::expect_throw;
using learnedhash_test::unique_temp_path;
using learnedhash_test::write_uint64_dataset;

void test_uint256_edge_ops(TestContext& ctx) {
  ctx.check(lead::UInt256::max_for_bits(0).is_zero(), "LEAD UInt256 max_for_bits(0)");

  auto max65 = lead::UInt256::max_for_bits(65);
  ctx.check(max65.words[0] == std::numeric_limits<uint64_t>::max() &&
                max65.words[1] == 1 && max65.words[2] == 0 && max65.words[3] == 0,
            "LEAD UInt256 max_for_bits crosses word boundary");

  lead::UInt256 masked = lead::UInt256::max_for_bits(256);
  masked.mask_bits(65);
  ctx.check(masked.words == max65.words, "LEAD UInt256 mask_bits crosses word boundary");

  uint64_t overflow = 0;
  auto doubled = lead::mul_u64(lead::UInt256::max_for_bits(256), 2, &overflow);
  ctx.check(doubled.words[0] == std::numeric_limits<uint64_t>::max() - 1 &&
                doubled.words[1] == std::numeric_limits<uint64_t>::max() &&
                doubled.words[2] == std::numeric_limits<uint64_t>::max() &&
                doubled.words[3] == std::numeric_limits<uint64_t>::max() &&
                overflow == 1,
            "LEAD UInt256 mul_u64 reports overflow");

  lead::UInt320 wide{};
  wide.words = {0, 1, 2, 3, 4};
  auto shifted320 = lead::shr_320(wide, 0);
  ctx.check(shifted320.words == std::array<uint64_t, 4>{0, 1, 2, 3},
            "LEAD UInt256 shr_320 zero shift");

  lead::UInt256 value{};
  value.words = {0, 1, 2, 3};
  auto shifted256 = lead::shr_256(value, 64);
  ctx.check(shifted256.words == std::array<uint64_t, 4>{1, 2, 3, 0},
            "LEAD UInt256 shr_256 word shift");

  uint64_t rem = 0;
  auto divided = lead::div_u64(lead::UInt256::max_for_bits(64), 3, &rem);
  ctx.check(divided.words[0] == 6148914691236517205ULL && rem == 0,
            "LEAD UInt256 div_u64 stable quotient");

  ctx.check(lead::msb_u64(value, 128) == 1, "LEAD UInt256 msb_u64 high window");
  ctx.check(lead::msb_u128(value, 256) == std::array<uint64_t, 2>{2, 3},
            "LEAD UInt256 msb_u128 high window");

  lead::UInt256 bytes_value{};
  bytes_value.words = {0x0102030405060708ULL, 0, 0, 0x1112131415161718ULL};
  auto bytes = lead::to_bytes_be(bytes_value);
  ctx.check(bytes.front() == 0x11 && bytes.back() == 0x08,
            "LEAD UInt256 to_bytes_be endian order");

  expect_throw(ctx,
               [] {
                 lead::UInt320 tmp{};
                 (void)lead::shr_320(tmp, 64);
               },
               "LEAD UInt256 shr_320 rejects invalid shift");
}

void test_mapping_edge_ops(TestContext& ctx) {
  std::string warning;
  auto cfg = lead::MappingConfig::compute(127, 128, 1.0, &warning);
  lead::IndexMapper mapper(cfg);
  auto low = mapper.map(0);
  auto high = mapper.map(127);
  ctx.check(low.bits == 128 && high.bits == 128, "LEAD mapper preserves configured bits");
  ctx.check(lead::msb_u128(high.value, high.bits)[1] != 0,
            "LEAD mapper reaches high 128-bit range");

  expect_throw(ctx,
               [] {
                 std::string unused;
                 (void)lead::MappingConfig::compute(1, 0, 1.0, &unused);
               },
               "LEAD mapper rejects zero space bits");
}

void test_generated_lead_load_smoke(TestContext& ctx) {
  auto dir = unique_temp_path("lead_runtime");
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  try {
    std::vector<uint64_t> values;
    values.reserve(128);
    for (uint64_t i = 0; i < 128; ++i) {
      values.push_back(i);
    }
    auto dataset = dir / "lead_runtime_uint64.bin";
    write_uint64_dataset(dataset, values);

    lead::TrainOptions opts;
    opts.input = dataset.string();
    opts.namespace_name = "lead_runtime_test";
    opts.model_spec = "linear,linear";
    opts.branching_factor = 8;
    opts.output_dir = dir / "out";
    opts.threads = 1;
    opts.selector_limit = 2;
    opts.space_bits = 64;
    opts.zero_build_time = true;

    auto result = lead::train_and_generate(opts);
    lead::Hasher hasher;
    ctx.check(hasher.load(result.lead_dir), "LEAD generated hasher loads generated rm_model");

    double pred = std::numeric_limits<double>::quiet_NaN();
    auto hash = hasher.hash_with_pred(64ULL, &pred);
    ctx.check(hash.bits == 64, "LEAD generated hash preserves configured bit width");
    ctx.check(std::isfinite(pred), "LEAD generated hash reports finite prediction");
    ctx.check(hash.to_u64_msb() == hasher.hash64(64ULL),
              "LEAD generated hash64 matches HashOutput conversion");
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] LEAD generated load smoke threw: " << e.what() << "\n";
    ctx.failures++;
  }

  std::filesystem::remove_all(dir);
}

} // namespace

int main() {
  TestContext ctx;
  test_uint256_edge_ops(ctx);
  test_mapping_edge_ops(ctx);
  test_generated_lead_load_smoke(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All LEAD runtime tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " LEAD runtime tests failed\n";
  return 1;
}
