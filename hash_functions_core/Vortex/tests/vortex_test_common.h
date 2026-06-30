#ifndef VORTEX_V1_TEST_COMMON_H
#define VORTEX_V1_TEST_COMMON_H

#include "vortex_v1/model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vortex_test {

struct TestContext {
  int failures = 0;

  void check(bool cond, const std::string& msg) {
    if (!cond) {
      std::cerr << "[FAIL] " << msg << "\n";
      failures++;
    }
  }
};

template <typename Fn>
void expect_throw(TestContext& ctx, Fn&& fn, const std::string& msg) {
  bool threw = false;
  try {
    std::forward<Fn>(fn)();
  } catch (const std::exception&) {
    threw = true;
  }
  ctx.check(threw, msg);
}

inline std::filesystem::path project_root() {
#ifdef VORTEX_V1_PROJECT_ROOT
  return std::filesystem::path(VORTEX_V1_PROJECT_ROOT);
#else
  return std::filesystem::current_path();
#endif
}

inline std::filesystem::path unique_temp_path(const std::string& prefix) {
  auto temp_dir = std::filesystem::temp_directory_path();
  std::filesystem::path hint(prefix);
  std::string stem = hint.stem().string();
  if (stem.empty()) {
    stem = hint.string();
  }
  std::string ext = hint.extension().string();
  std::random_device rd;
  for (int attempt = 0; attempt < 64; ++attempt) {
    uint64_t suffix = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    auto candidate = temp_dir / (stem + "_" + std::to_string(suffix) + ext);
    if (!std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  return temp_dir / (stem + "_fallback" + ext);
}

inline std::filesystem::path model_selector_tool_path() {
#ifdef VORTEX_MODEL_SELECTOR_PATH
  return std::filesystem::path(VORTEX_MODEL_SELECTOR_PATH);
#else
  return std::filesystem::current_path() / "vortex_model_selector";
#endif
}

inline bool model_selector_tool_is_required() {
#ifdef VORTEX_MODEL_SELECTOR_PATH
  return true;
#else
  return false;
#endif
}

inline std::filesystem::path vortex_cli_tool_path() {
#ifdef VORTEX_V1_CLI_PATH
  return std::filesystem::path(VORTEX_V1_CLI_PATH);
#else
  return std::filesystem::current_path() / "vortex_v1_cli";
#endif
}

inline bool vortex_cli_tool_is_required() {
#ifdef VORTEX_V1_CLI_PATH
  return true;
#else
  return false;
#endif
}

inline std::string shell_quote(const std::string& value) {
#if defined(_WIN32)
  std::string quoted = "\"";
  for (char ch : value) {
    if (ch == '"') {
      quoted += "\\\"";
    } else {
      quoted.push_back(ch);
    }
  }
  quoted.push_back('"');
  return quoted;
#else
  std::string quoted = "'";
  for (char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(ch);
    }
  }
  quoted.push_back('\'');
  return quoted;
#endif
}

inline std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

inline std::vector<unsigned char> read_binary_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open file: " + path.string());
  }
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(input),
                                    std::istreambuf_iterator<char>());
}

inline void check_same_bytes(TestContext& ctx,
                             const std::filesystem::path& lhs,
                             const std::filesystem::path& rhs,
                             const std::string& msg) {
  auto lhs_bytes = read_binary_file(lhs);
  auto rhs_bytes = read_binary_file(rhs);
  ctx.check(lhs_bytes == rhs_bytes, msg);
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(std::string key, std::string value) : key_(std::move(key)) {
    if (const char* prior = std::getenv(key_.c_str())) {
      had_prior_ = true;
      prior_value_ = prior;
    }
#if defined(_WIN32)
    _putenv_s(key_.c_str(), value.c_str());
#else
    setenv(key_.c_str(), value.c_str(), 1);
#endif
  }

  ~ScopedEnvVar() {
#if defined(_WIN32)
    if (had_prior_) {
      _putenv_s(key_.c_str(), prior_value_.c_str());
    } else {
      _putenv_s(key_.c_str(), "");
    }
#else
    if (had_prior_) {
      setenv(key_.c_str(), prior_value_.c_str(), 1);
    } else {
      unsetenv(key_.c_str());
    }
#endif
  }

  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

 private:
  std::string key_;
  std::string prior_value_;
  bool had_prior_ = false;
};

inline vortex::VortexModel make_golden_model() {
  vortex::VortexModel model;
  model.dim = 2;
  model.metric = vortex::Metric::L2;
  model.hash_bits = 64;
  model.centroids = {
      0.0f, 0.0f,
      2.0f, 0.0f,
      0.0f, 2.0f,
  };
  model.order = {0, 1, 2};
  model.mass = {1, 2, 4};
  model.range_start.resize(3);
  model.range_size.resize(3);
  model.range_start[0].words = {0, 0, 0, 0};
  model.range_size[0].words = {1024, 0, 0, 0};
  model.range_start[1].words = {1024, 0, 0, 0};
  model.range_size[1].words = {2048, 0, 0, 0};
  model.range_start[2].words = {4096, 0, 0, 0};
  model.range_size[2].words = {4096, 0, 0, 0};
  model.range_full = {0, 0, 0};
  model.min_dist = {0.0, 0.0, 0.0};
  model.max_dist = {1.0, 1.0, 1.0};
  model.cdf_top_type = vortex::CdfModelType::Linear;
  model.cdf_leaf_type = vortex::CdfModelType::Linear;
  model.cdf_leaf_count = 2;
  model.cdf_top_param_count = vortex::cdf_param_count(model.cdf_top_type);
  model.cdf_leaf_param_count = vortex::cdf_param_count(model.cdf_leaf_type);
  model.cdf_rows = {2, 2, 2};
  model.cdf_top_params = {
      0.0, 1.0,
      0.0, 1.0,
      0.0, 1.0,
  };
  model.cdf_leaf_params = {
      0.0, 1.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 1.0,
  };
  model.compute_active();
  return model;
}

} // namespace vortex_test

#endif // VORTEX_V1_TEST_COMMON_H
