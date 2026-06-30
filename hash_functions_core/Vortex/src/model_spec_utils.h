#ifndef VORTEX_V1_MODEL_SPEC_UTILS_H
#define VORTEX_V1_MODEL_SPEC_UTILS_H

#include <string>
#include <vector>

namespace vortex::detail {

inline std::string trim_copy(const std::string& value) {
  std::size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

inline std::vector<std::string> split_model_spec(const std::string& spec) {
  std::vector<std::string> parts;
  std::string token;
  for (char ch : spec) {
    if (ch == ',') {
      auto trimmed = trim_copy(token);
      if (!trimmed.empty()) {
        parts.push_back(trimmed);
      }
      token.clear();
    } else {
      token.push_back(ch);
    }
  }
  auto trimmed = trim_copy(token);
  if (!trimmed.empty()) {
    parts.push_back(trimmed);
  }
  return parts;
}

} // namespace vortex::detail

#endif // VORTEX_V1_MODEL_SPEC_UTILS_H
