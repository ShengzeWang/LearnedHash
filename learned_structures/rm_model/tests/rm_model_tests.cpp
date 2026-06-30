#include "learnedhash_test_common.h"

#include "rm_model/data_loader.h"
#include "rm_model/parallel.h"
#include "rm_model/train.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using learnedhash_test::TestContext;
using learnedhash_test::expect_no_throw;
using learnedhash_test::expect_throw;
using learnedhash_test::unique_temp_path;
using learnedhash_test::write_u64_le;

template <typename T>
rm_model::TrainingData<T> make_training_data(std::vector<std::pair<T, std::size_t>> rows) {
  using Provider = typename rm_model::TrainingData<T>::VectorProvider;
  return rm_model::TrainingData<T>(std::make_shared<Provider>(std::move(rows)));
}

template <typename T>
rm_model::TrainingData<T> make_linear_rows(std::size_t count) {
  std::vector<std::pair<T, std::size_t>> rows;
  rows.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    rows.emplace_back(static_cast<T>(i), i);
  }
  return make_training_data<T>(std::move(rows));
}

void test_data_loader_reads_uint64_dataset(TestContext& ctx) {
  auto path = unique_temp_path("rm_model_data_loader", ".bin");
  try {
    {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out) {
        throw std::runtime_error("Unable to write loader test dataset");
      }
      write_u64_le(out, 3);
      write_u64_le(out, 10);
      write_u64_le(out, 20);
      write_u64_le(out, 30);
    }

    auto loaded = rm_model::load_data(path.string(), rm_model::DataType::UINT64);
    ctx.check(loaded.first == 3, "data loader row count");
    auto data = loaded.second.into_u64();
    ctx.check(data.has_value(), "data loader returns u64 data");
    if (data.has_value()) {
      ctx.check(data->len() == 3, "data loader u64 length");
      ctx.check(data->get(0) == std::pair<uint64_t, std::size_t>{10, 0},
                "data loader first row");
      ctx.check(data->get(2) == std::pair<uint64_t, std::size_t>{30, 2},
                "data loader last row");
    }
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] data loader smoke threw: " << e.what() << "\n";
    ctx.failures++;
  }
  std::filesystem::remove(path);
}

void test_two_layer_training_smoke(TestContext& ctx) {
  std::vector<std::pair<uint64_t, std::size_t>> rows;
  rows.reserve(1024);
  for (std::size_t i = 0; i < 1024; ++i) {
    rows.emplace_back(static_cast<uint64_t>(i), i);
  }
  auto data = make_training_data<uint64_t>(std::move(rows));
  auto trained = rm_model::train<uint64_t>(data, "linear,linear", 8);

  ctx.check(trained.model_layers.size() == 2, "two-layer model count");
  ctx.check(trained.model_layers[0].size() == 1, "top layer size");
  ctx.check(trained.model_layers[1].size() == 8, "leaf layer size");
  ctx.check(trained.num_data_rows == 1024, "row count propagated");
}

void test_duplicate_key_training_stability(TestContext& ctx) {
  std::vector<std::pair<double, std::size_t>> rows;
  rows.reserve(2048);
  for (std::size_t i = 0; i < 2048; ++i) {
    // 8 identical keys per value keeps ordering valid while stressing split behavior.
    rows.emplace_back(static_cast<double>(i / 8), i);
  }
  auto data = make_training_data<double>(std::move(rows));

  bool threw = false;
  try {
    auto trained = rm_model::train<double>(data, "linear,linear", 32);
    ctx.check(trained.model_layers.size() == 2, "duplicate-key model has two layers");
    ctx.check(trained.model_layers[1].size() == 32, "duplicate-key leaf count");
  } catch (const std::exception&) {
    threw = true;
  }
  ctx.check(!threw, "training with duplicate keys remains stable");
}

void test_invalid_model_spec_rejected(TestContext& ctx) {
  expect_throw(
      ctx,
      [&]() {
        auto data = make_linear_rows<uint32_t>(16);
        (void)rm_model::train<uint32_t>(data, "", 4);
      },
      "empty model spec is rejected");
  expect_throw(
      ctx,
      [&]() {
        auto data = make_linear_rows<uint32_t>(16);
        (void)rm_model::train<uint32_t>(data, "linear,,linear", 4);
      },
      "model spec empty layer is rejected");
  expect_throw(
      ctx,
      [&]() {
        auto data = make_linear_rows<uint32_t>(16);
        (void)rm_model::train<uint32_t>(data, "linear,not_a_model", 4);
      },
      "unknown model type is rejected");
  expect_throw(
      ctx,
      [&]() {
        auto data = make_linear_rows<uint32_t>(16);
        (void)rm_model::train<uint32_t>(data, "linear,radix", 4);
      },
      "root-only model below root is rejected");
  expect_no_throw(
      ctx,
      [&]() {
        auto data = make_linear_rows<uint32_t>(64);
        auto trained = rm_model::train<uint32_t>(data, " linear , linear ", 4);
        ctx.check(trained.model_spec == "linear,linear",
                  "trimmed model spec is normalized in trained model");
      },
      "model spec whitespace is accepted");
  expect_no_throw(
      ctx,
      [&]() {
        auto data = make_linear_rows<double>(64);
        auto trained = rm_model::train<double>(data, "linear,cdf_piecewise", 4);
        ctx.check(trained.model_layers.size() == 2,
                  "cdf_piecewise training keeps two layers");
      },
      "cdf_piecewise model listed in README is trainable");
}

void test_parallel_lifetime_stress(TestContext& ctx) {
  rm_model::set_thread_count(4);
  for (std::size_t round = 0; round < 200; ++round) {
    std::atomic<std::size_t> task_count{0};
    {
      rm_model::TaskGroup group(8);
      for (std::size_t i = 0; i < 64; ++i) {
        group.schedule([&]() {
          task_count.fetch_add(1, std::memory_order_relaxed);
        });
      }
    }
    ctx.check(task_count.load(std::memory_order_relaxed) == 64,
              "TaskGroup completes all scheduled tasks before destruction");

    std::atomic<std::size_t> loop_count{0};
    rm_model::parallel_for(128, [&](std::size_t) {
      loop_count.fetch_add(1, std::memory_order_relaxed);
    });
    ctx.check(loop_count.load(std::memory_order_relaxed) == 128,
              "parallel_for completes all scheduled tasks before returning");
  }
  rm_model::set_thread_count(1);
}

} // namespace

int main() {
  TestContext ctx;
  test_data_loader_reads_uint64_dataset(ctx);
  test_two_layer_training_smoke(ctx);
  test_duplicate_key_training_stability(ctx);
  test_invalid_model_spec_rejected(ctx);
  test_parallel_lifetime_stress(ctx);

  if (ctx.failures == 0) {
    std::cerr << "All rm_model tests passed\n";
    return 0;
  }
  std::cerr << ctx.failures << " rm_model tests failed\n";
  return 1;
}
