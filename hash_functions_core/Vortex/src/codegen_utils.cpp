#include "codegen_internal.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vortex::codegen_internal {

std::string sanitize_identifier(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (unsigned char ch : value) {
    if (std::isalnum(ch) || ch == '_') {
      out.push_back(static_cast<char>(ch));
    } else {
      out.push_back('_');
    }
  }
  if (out.empty() || (!std::isalpha(static_cast<unsigned char>(out[0])) && out[0] != '_')) {
    out.insert(out.begin(), '_');
    out.insert(out.begin(), 'v');
    out.insert(out.begin(), 't');
    out.insert(out.begin(), 'x');
  }
  return out;
}

std::string to_upper(std::string value) {
  for (auto& ch : value) {
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  }
  return value;
}

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Unable to write file: " + path.string());
  }
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!out) {
    throw std::runtime_error("Unable to write file: " + path.string());
  }
}

std::string header_guard_for(const std::string& ident) {
  return "VORTEX_V1_CODEGEN_" + to_upper(ident) + "_H";
}

std::string escape_c_string(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    if (ch == '\\' || ch == '\"') {
      out.push_back('\\');
    }
    out.push_back(ch);
  }
  return out;
}

} // namespace vortex::codegen_internal
