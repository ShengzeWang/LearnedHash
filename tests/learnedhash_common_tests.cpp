#include "learnedhash_test_common.h"

#include "rm_model/benchmark.h"
#include "rm_model/detail/runtime_loader.h"
#include "rm_model/logging.h"
#include "rm_model/uint256.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>

namespace {

using learnedhash_test::ScopedEnvVar;
using learnedhash_test::TestContext;
using learnedhash_test::expect_throw;

void test_format_duration_ns(TestContext& ctx) {
  ctx.check(rm_model::format_duration_ns(999) == "999 ns",
            "nanosecond duration formatting");
  ctx.check(rm_model::format_duration_ns(1000) == "1.000 us",
            "microsecond duration formatting");
  ctx.check(rm_model::format_duration_ns(1000ULL * 1000ULL) == "1.000 ms",
            "millisecond duration formatting");
}

void test_timing_stats_single_sample(TestContext& ctx) {
  rm_model::TimingStats timing;
  ctx.check(timing.empty(), "new timing stats are empty");

  timing.add(1234);
  ctx.check(!timing.empty(), "timing stats become non-empty after add");
  ctx.check(timing.total_ns() == 1234, "timing total for one sample");
  ctx.check(timing.avg_ns() == 1234, "timing average for one sample");
  ctx.check(timing.min_ns() == 1234, "timing min for one sample");
  ctx.check(timing.max_ns() == 1234, "timing max for one sample");
  ctx.check(timing.p50_ns() == 1234, "timing p50 for one sample");
  ctx.check(timing.p95_ns() == 1234, "timing p95 for one sample");
  ctx.check(timing.p99_ns() == 1234, "timing p99 for one sample");
}

void test_runtime_loader_shell_quote_policy(TestContext& ctx) {
  using rm_model::detail::shell_quote_checked;

  ctx.check(shell_quote_checked("/tmp/model path", "path") == "\"/tmp/model path\"",
            "shell quote preserves safe spaces");
  expect_throw(ctx,
               [] { (void)shell_quote_checked("", "path"); },
               "shell quote rejects empty arguments");
  expect_throw(ctx,
               [] { (void)shell_quote_checked("/tmp/model;rm -rf", "path"); },
               "shell quote rejects shell metacharacters");
  expect_throw(ctx,
               [] { (void)shell_quote_checked("/tmp/model\npath", "path"); },
               "shell quote rejects control separators");
}

void test_logging_env_precedence(TestContext& ctx) {
  {
    ScopedEnvVar learnedhash_log("LEARNEDHASH_LOG", std::nullopt);
    ScopedEnvVar rust_log("RUST_LOG", std::nullopt);
    rm_model::set_log_level(rm_model::LogLevel::Error);
    rm_model::init_logging();
    ctx.check(rm_model::log_level() == rm_model::LogLevel::Error,
              "init_logging leaves explicit level unchanged when env is unset");
  }

  {
    ScopedEnvVar learnedhash_log("LEARNEDHASH_LOG", std::string("debug"));
    ScopedEnvVar rust_log("RUST_LOG", std::nullopt);
    rm_model::set_log_level(rm_model::LogLevel::Info);
    rm_model::init_logging();
    ctx.check(rm_model::log_level() == rm_model::LogLevel::Debug,
              "LEARNEDHASH_LOG controls log level");
  }

  {
    ScopedEnvVar learnedhash_log("LEARNEDHASH_LOG", std::nullopt);
    ScopedEnvVar rust_log("RUST_LOG", std::string("warn"));
    rm_model::set_log_level(rm_model::LogLevel::Info);
    rm_model::init_logging();
    ctx.check(rm_model::log_level() == rm_model::LogLevel::Warn,
              "RUST_LOG remains a compatibility fallback");
  }

  {
    ScopedEnvVar learnedhash_log("LEARNEDHASH_LOG", std::string("error"));
    ScopedEnvVar rust_log("RUST_LOG", std::string("trace"));
    rm_model::set_log_level(rm_model::LogLevel::Info);
    rm_model::init_logging();
    ctx.check(rm_model::log_level() == rm_model::LogLevel::Error,
              "LEARNEDHASH_LOG takes precedence over RUST_LOG");
  }
  rm_model::set_log_level(rm_model::LogLevel::Info);
}

void test_uint256_shared_superset(TestContext& ctx) {
  auto a = rm_model::UInt256::from_u64(5);
  auto b = rm_model::UInt256::from_u64(7);
  auto sum = rm_model::add(a, b);
  ctx.check(rm_model::to_u64(sum, 64) == 12, "shared UInt256 add");

  auto high = rm_model::UInt256::one_shifted(128);
  ctx.check(high.words[2] == 1 && high.words[0] == 0,
            "shared UInt256 one_shifted high word");
  ctx.check(rm_model::UInt256::one_shifted(256).is_zero(),
            "shared UInt256 one_shifted 256 sentinel");

  rm_model::UInt320 wide = rm_model::shl_u64_to_320(1, 128);
  uint64_t rem = 0;
  auto div = rm_model::div_u64(wide, 2, &rem);
  ctx.check(div.words[1] == (1ULL << 63) && rem == 0,
            "shared UInt320 division");

  auto max65 = rm_model::UInt256::max_for_bits(65);
  ctx.check(rm_model::to_hex(max65, 65) == "1ffffffffffffffff",
            "shared UInt256 hex formatting");
  auto bytes = rm_model::to_bytes_be(max65);
  ctx.check(bytes.back() == 0xFF, "shared UInt256 big-endian bytes");
}

} // namespace

int main() {
  TestContext ctx;
  test_format_duration_ns(ctx);
  test_timing_stats_single_sample(ctx);
  test_runtime_loader_shell_quote_policy(ctx);
  test_logging_env_precedence(ctx);
  test_uint256_shared_superset(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All LearnedHash common tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " common tests failed\n";
  return 1;
}
