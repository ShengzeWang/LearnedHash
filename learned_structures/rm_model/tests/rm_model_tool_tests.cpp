#include "learnedhash_test_common.h"

#include "rm_model/detail/runtime_loader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef RM_MODEL_LEARNER_PATH
#define RM_MODEL_LEARNER_PATH ""
#endif

#ifndef RM_MODEL_INFERENCER_PATH
#define RM_MODEL_INFERENCER_PATH ""
#endif

namespace {

using learnedhash_test::TestContext;
using learnedhash_test::expect_command_fails;
using learnedhash_test::quote;
using learnedhash_test::unique_temp_path;
using learnedhash_test::write_uint64_dataset;

void test_malformed_numeric_flags(TestContext& ctx) {
  std::string learner = quote(RM_MODEL_LEARNER_PATH, "rm_model_learner");
  std::string inferencer = quote(RM_MODEL_INFERENCER_PATH, "rm_model_inferencer");
  auto missing_dataset = std::filesystem::temp_directory_path() /
                         "learnedhash_missing_rm_model_tool.bin";
  auto missing_model = std::filesystem::temp_directory_path() /
                       "learnedhash_missing_rm_model_tool_model";

  expect_command_fails(
      ctx,
      learner + " " + quote(missing_dataset, "dataset") +
          " rm_model_tool_test linear,linear 8 --threads 1junk",
      "rm_model_learner_malformed_threads",
      "Invalid value for --threads");
  expect_command_fails(
      ctx,
      learner + " " + quote(missing_dataset, "dataset") +
          " rm_model_tool_test linear,linear 8 --bounded -1",
      "rm_model_learner_negative_bounded",
      "Invalid value for --bounded");
  expect_command_fails(
      ctx,
      inferencer + " --model-dir " + quote(missing_model, "model_dir") +
          " --dataset " + quote(missing_dataset, "dataset") +
          " --limit 1junk",
      "rm_model_inferencer_malformed_limit",
      "Invalid value for --limit");
  expect_command_fails(
      ctx,
      inferencer + " --model-dir " + quote(missing_model, "model_dir") +
          " --dataset " + quote(missing_dataset, "dataset") +
          " --limit -1",
      "rm_model_inferencer_negative_limit",
      "Invalid value for --limit");
}

void test_generated_tool_load_smoke(TestContext& ctx) {
  auto dir = unique_temp_path("rm_model_tools");
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  try {
    std::vector<uint64_t> values;
    values.reserve(128);
    for (uint64_t i = 0; i < 128; ++i) {
      values.push_back(i);
    }
    auto dataset = dir / "rm_model_tool_uint64.bin";
    auto model_dir = dir / "model";
    auto out_file = dir / "infer.csv";
    write_uint64_dataset(dataset, values);

    std::string learner = quote(RM_MODEL_LEARNER_PATH, "rm_model_learner");
    std::string inferencer = quote(RM_MODEL_INFERENCER_PATH, "rm_model_inferencer");

    std::string learn_cmd =
        learner + " " + quote(dataset, "dataset") +
        " rm_model_tool_test linear,linear 8 --output-dir " + quote(model_dir, "model_dir") +
        " --threads 1 --zero-build-time";
    rm_model::detail::run_command(learn_cmd, "rm_model_learner failed: ");

    std::string infer_cmd =
        inferencer + " --model-dir " + quote(model_dir, "model_dir") +
        " --key 64 --key-type u64 --out " + quote(out_file, "output");
    rm_model::detail::run_command(infer_cmd, "rm_model_inferencer failed: ");

    ctx.check(std::filesystem::exists(model_dir / "model.json"),
              "rm_model learner wrote model manifest");
    ctx.check(std::filesystem::exists(out_file), "rm_model inferencer wrote output");

    std::ifstream in(out_file);
    std::string header;
    std::string row;
    std::getline(in, header);
    std::getline(in, row);
    ctx.check(header == "key,pred,err", "rm_model inferencer output header");
    ctx.check(!row.empty() && row.find("64,") == 0, "rm_model inferencer output row");
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] rm_model tool smoke threw: " << e.what() << "\n";
    ctx.failures++;
  }

  std::filesystem::remove_all(dir);
}

} // namespace

int main() {
  TestContext ctx;
  test_malformed_numeric_flags(ctx);
  test_generated_tool_load_smoke(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All rm_model tool tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " rm_model tool tests failed\n";
  return 1;
}
