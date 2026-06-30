#ifndef VORTEX_V1_CODEGEN_INTERNAL_H
#define VORTEX_V1_CODEGEN_INTERNAL_H

#include "vortex_v1/codegen.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace vortex::codegen_internal {

inline constexpr uint32_t kDefaultCentroidIndexPivots = 16;
inline constexpr const char* kCentroidIndexFile = "centroid_index.bin";

std::string sanitize_identifier(const std::string& value);
std::string to_upper(std::string value);
void write_text(const std::filesystem::path& path, const std::string& text);
std::string header_guard_for(const std::string& ident);
std::string escape_c_string(const std::string& value);

void write_centroid_index_file(const VortexModel& model,
                               const std::filesystem::path& path,
                               uint32_t pivot_count);

std::string generate_header(const std::string& ident);
std::string generate_source(const std::string& ident, const std::string& user_name);
std::string generate_cmake(const std::string& ident);
std::string generate_readme(const std::string& ident, const std::string& user_name);

void copy_runtime_sources(const std::filesystem::path& runtime_include,
                          const std::filesystem::path& runtime_src);

void write_manifest(const VortexModel& model,
                    const CodegenOptions& options,
                    const std::string& ident,
                    const std::filesystem::path& manifest_path);

} // namespace vortex::codegen_internal

#endif // VORTEX_V1_CODEGEN_INTERNAL_H
