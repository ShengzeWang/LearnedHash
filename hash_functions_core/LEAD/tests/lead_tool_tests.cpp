#include "learnedhash_test_common.h"

#include "rm_model/detail/runtime_loader.h"

#include <filesystem>
#include <iostream>
#include <string>

#ifndef LEAD_HASHGEN_PATH
#define LEAD_HASHGEN_PATH ""
#endif

#ifndef LEAD_HASHER_PATH
#define LEAD_HASHER_PATH ""
#endif

namespace {

using learnedhash_test::TestContext;
using learnedhash_test::expect_command_fails;
using learnedhash_test::quote;
using learnedhash_test::unique_temp_path;

void test_malformed_numeric_flags(TestContext& ctx) {
  std::string hashgen = quote(LEAD_HASHGEN_PATH, "lead_hashgen");
  std::string hasher = quote(LEAD_HASHER_PATH, "lead_hasher");
  auto missing_dataset = unique_temp_path("missing_lead_dataset.bin");
  auto missing_model = unique_temp_path("missing_lead_model");

  expect_command_fails(
      ctx,
      hashgen + " " + quote(missing_dataset, "dataset") +
          " lead_tool_test linear,linear 8 --selector-limit 1junk",
      "lead_hashgen_malformed_selector_limit",
      "Invalid value for --selector-limit");
  expect_command_fails(
      ctx,
      hashgen + " " + quote(missing_dataset, "dataset") +
          " lead_tool_test linear,linear 8 --space-bits -1",
      "lead_hashgen_negative_space_bits",
      "Invalid value for --space-bits");
  expect_command_fails(
      ctx,
      hashgen + " " + quote(missing_dataset, "dataset") +
          " lead_tool_test linear,linear 8 --scale-factor nan",
      "lead_hashgen_nonfinite_scale_factor",
      "Invalid value for --scale-factor");
  expect_command_fails(
      ctx,
      hasher + " --model-dir " + quote(missing_model, "model_dir") +
          " --key 1 --limit 1junk",
      "lead_hasher_malformed_limit",
      "Invalid value for --limit");
  expect_command_fails(
      ctx,
      hasher + " --model-dir " + quote(missing_model, "model_dir") +
          " --key 1 --limit -1",
      "lead_hasher_negative_limit",
      "Invalid value for --limit");
}

} // namespace

int main() {
  TestContext ctx;
  test_malformed_numeric_flags(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All LEAD tool tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " LEAD tool tests failed\n";
  return 1;
}
