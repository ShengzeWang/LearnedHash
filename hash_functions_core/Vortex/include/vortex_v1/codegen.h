#ifndef VORTEX_V1_CODEGEN_H
#define VORTEX_V1_CODEGEN_H

#include "vortex_v1/model.h"

#include <filesystem>
#include <string>

namespace vortex {

struct CodegenOptions {
  std::filesystem::path output_dir;
  std::string name;
  bool emit_cmake = true;
};

struct CodegenResult {
  std::filesystem::path output_dir;
  std::filesystem::path model_path;
  std::filesystem::path centroid_index_path;
  std::filesystem::path manifest_path;
  std::filesystem::path include_path;
  std::filesystem::path src_path;
  std::filesystem::path cmake_path;
};

CodegenResult emit_codegen(const VortexModel& model, const CodegenOptions& options);
CodegenResult emit_codegen_from_file(const std::filesystem::path& model_path,
                                     const CodegenOptions& options);

} // namespace vortex

#endif // VORTEX_V1_CODEGEN_H
