#ifndef LEARNEDHASH_TEST_COMMON_H
#define LEARNEDHASH_TEST_COMMON_H

#include "rm_model/detail/runtime_loader.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace learnedhash_test {

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

template <typename Fn>
void expect_no_throw(TestContext& ctx, Fn&& fn, const std::string& msg) {
  try {
    std::forward<Fn>(fn)();
    ctx.check(true, msg);
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << msg << ": " << e.what() << "\n";
    ctx.failures++;
  }
}

inline std::filesystem::path unique_temp_path(const std::string& tag,
                                              const std::string& suffix = "") {
  static std::atomic<uint64_t> counter{0};
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("learnedhash_" + tag + "_" + std::to_string(now) + "_" +
          std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) + suffix);
}

inline std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Unable to read text file: " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

inline std::string quote(const std::filesystem::path& path, const char* label) {
  return rm_model::detail::shell_quote_checked(path.string(), label);
}

inline void write_u64_le(std::ofstream& out, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out.put(static_cast<char>((value >> (8 * i)) & 0xFFU));
  }
}

inline void write_uint64_dataset(const std::filesystem::path& path,
                                 const std::vector<uint64_t>& values) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Unable to write test dataset: " + path.string());
  }
  write_u64_le(out, static_cast<uint64_t>(values.size()));
  for (uint64_t value : values) {
    write_u64_le(out, value);
  }
}

inline void expect_command_fails(TestContext& ctx,
                                 const std::string& command,
                                 const std::string& tag,
                                 const std::string& expected_substr) {
  auto output_path = unique_temp_path(tag + "_output", ".txt");
  std::string wrapped = command + " > " + quote(output_path, "output") + " 2>&1";
  int status = std::system(wrapped.c_str());
  ctx.check(status != 0, tag + " exits non-zero");
  std::string output = read_text_file(output_path);
  ctx.check(output.find(expected_substr) != std::string::npos,
            tag + " reports expected error");
  std::filesystem::remove(output_path);
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(std::string key, std::optional<std::string> value)
      : key_(std::move(key)) {
    if (const char* prior = std::getenv(key_.c_str())) {
      prior_value_ = std::string(prior);
    }
    apply(value);
  }

  ~ScopedEnvVar() {
    apply(prior_value_);
  }

  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

 private:
  void apply(const std::optional<std::string>& value) {
#if defined(_WIN32)
    _putenv_s(key_.c_str(), value ? value->c_str() : "");
#else
    if (value) {
      setenv(key_.c_str(), value->c_str(), 1);
    } else {
      unsetenv(key_.c_str());
    }
#endif
  }

  std::string key_;
  std::optional<std::string> prior_value_;
};

} // namespace learnedhash_test

#endif // LEARNEDHASH_TEST_COMMON_H
