#ifndef RM_MODEL_DETAIL_CLI_ARGS_H
#define RM_MODEL_DETAIL_CLI_ARGS_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace rm_model::detail {

inline bool is_flag(const std::string& arg) {
  return arg.rfind("-", 0) == 0;
}

inline bool has_help_flag(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      return true;
    }
  }
  return false;
}

inline std::string require_value(int& index, int argc, char** argv, const std::string& name) {
  if (index + 1 >= argc) {
    throw std::runtime_error("Missing value for " + name);
  }
  return argv[++index];
}

inline uint64_t parse_u64_arg(const std::string& value, const std::string& name) {
  if (!value.empty() && value.front() == '-') {
    throw std::runtime_error("Invalid value for " + name + ": " + value);
  }
  try {
    std::size_t pos = 0;
    unsigned long long parsed = std::stoull(value, &pos);
    if (pos != value.size()) {
      throw std::runtime_error("Trailing characters in " + name);
    }
    return static_cast<uint64_t>(parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid value for " + name + ": " + value);
  }
}

inline uint32_t parse_u32_arg(const std::string& value, const std::string& name) {
  uint64_t parsed = parse_u64_arg(value, name);
  if (parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Value too large for " + name + ": " + value);
  }
  return static_cast<uint32_t>(parsed);
}

inline std::size_t parse_size_arg(const std::string& value, const std::string& name) {
  uint64_t parsed = parse_u64_arg(value, name);
  if (parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("Value too large for " + name + ": " + value);
  }
  return static_cast<std::size_t>(parsed);
}

inline double parse_double_arg(const std::string& value, const std::string& name) {
  try {
    std::size_t pos = 0;
    double parsed = std::stod(value, &pos);
    if (pos != value.size()) {
      throw std::runtime_error("Trailing characters in " + name);
    }
    return parsed;
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid value for " + name + ": " + value);
  }
}

inline double parse_finite_double_arg(const std::string& value, const std::string& name) {
  double parsed = parse_double_arg(value, name);
  if (!std::isfinite(parsed)) {
    throw std::runtime_error("Invalid value for " + name + ": " + value);
  }
  return parsed;
}

} // namespace rm_model::detail

#endif // RM_MODEL_DETAIL_CLI_ARGS_H
