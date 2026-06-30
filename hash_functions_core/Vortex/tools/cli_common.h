#ifndef VORTEX_V1_TOOLS_CLI_COMMON_H
#define VORTEX_V1_TOOLS_CLI_COMMON_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace vortex::tools {

struct ArgList {
  std::vector<std::string> args;

  explicit ArgList(int argc, char** argv) {
    args.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
      args.emplace_back(argv[i]);
    }
  }

  bool has(const std::string& flag) const {
    for (const auto& arg : args) {
      if (arg == flag) return true;
    }
    return false;
  }

  std::optional<std::string> value(const std::string& flag) const {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
      if (args[i] == flag) {
        if (args[i + 1].rfind("--", 0) == 0) {
          return std::nullopt;
        }
        return args[i + 1];
      }
    }
    return std::nullopt;
  }

  std::string require(const std::string& flag) const {
    auto val = value(flag);
    if (!val.has_value()) {
      throw std::runtime_error("Missing required flag: " + flag);
    }
    return *val;
  }
};

inline void validate_known_flags(const ArgList& args,
                                 std::initializer_list<std::string> allowed_flags,
                                 std::size_t start_index = 1) {
  std::unordered_set<std::string> allowed(allowed_flags.begin(), allowed_flags.end());
  for (std::size_t i = start_index; i < args.args.size(); ++i) {
    const std::string& arg = args.args[i];
    if (arg.rfind("--", 0) != 0) {
      continue;
    }
    if (!allowed.count(arg)) {
      throw std::runtime_error("Unknown flag: " + arg);
    }
  }
}

inline void validate_required_flag_values(
    const ArgList& args,
    std::initializer_list<std::string> flags_requiring_values,
    std::size_t start_index = 1) {
  std::unordered_set<std::string> needs_value(flags_requiring_values.begin(),
                                              flags_requiring_values.end());
  for (std::size_t i = start_index; i < args.args.size(); ++i) {
    const std::string& arg = args.args[i];
    if (!needs_value.count(arg)) {
      continue;
    }
    if (i + 1 >= args.args.size() || args.args[i + 1].rfind("--", 0) == 0) {
      throw std::runtime_error("Flag requires a value: " + arg);
    }
  }
}

inline uint32_t parse_u32(const std::string& value, const std::string& name) {
  if (!value.empty() && value.front() == '-') {
    throw std::runtime_error("Invalid value for " + name + ": " + value);
  }
  try {
    std::size_t pos = 0;
    unsigned long parsed = std::stoul(value, &pos);
    if (pos != value.size() || parsed > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("Value too large for " + name);
    }
    return static_cast<uint32_t>(parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid value for " + name + ": " + value);
  }
}

inline uint64_t parse_u64(const std::string& value, const std::string& name) {
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

inline int parse_i32(const std::string& value, const std::string& name) {
  try {
    std::size_t pos = 0;
    long parsed = std::stol(value, &pos);
    if (pos != value.size() || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
      throw std::runtime_error("Value out of range for " + name);
    }
    return static_cast<int>(parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid value for " + name + ": " + value);
  }
}

inline double parse_double(const std::string& value, const std::string& name) {
  try {
    std::size_t pos = 0;
    double parsed = std::stod(value, &pos);
    if (pos != value.size() || !std::isfinite(parsed)) {
      throw std::runtime_error("Invalid numeric value for " + name);
    }
    return parsed;
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid value for " + name + ": " + value);
  }
}

inline std::string trim_copy(const std::string& value) {
  std::size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

inline std::vector<std::string> split_list(const std::string& value, char delimiter) {
  std::vector<std::string> items;
  std::string token;
  for (char ch : value) {
    if (ch == delimiter) {
      std::string trimmed = trim_copy(token);
      if (!trimmed.empty()) {
        items.push_back(trimmed);
      }
      token.clear();
    } else {
      token.push_back(ch);
    }
  }
  std::string trimmed = trim_copy(token);
  if (!trimmed.empty()) {
    items.push_back(trimmed);
  }
  return items;
}

inline std::vector<uint32_t> parse_u32_list(const std::string& value,
                                            const std::string& name) {
  std::vector<uint32_t> out;
  for (const auto& token : split_list(value, ',')) {
    out.push_back(parse_u32(token, name));
  }
  if (out.empty()) {
    throw std::runtime_error(name + " must contain at least one value");
  }
  return out;
}

inline std::vector<uint64_t> parse_u64_list(const std::string& value,
                                            const std::string& name) {
  std::vector<uint64_t> out;
  for (const auto& token : split_list(value, ',')) {
    out.push_back(parse_u64(token, name));
  }
  if (out.empty()) {
    throw std::runtime_error(name + " must contain at least one value");
  }
  return out;
}

inline std::vector<double> parse_double_list(const std::string& value,
                                             const std::string& name) {
  std::vector<double> out;
  for (const auto& token : split_list(value, ',')) {
    out.push_back(parse_double(token, name));
  }
  if (out.empty()) {
    throw std::runtime_error(name + " must contain at least one value");
  }
  return out;
}

} // namespace vortex::tools

#endif // VORTEX_V1_TOOLS_CLI_COMMON_H
