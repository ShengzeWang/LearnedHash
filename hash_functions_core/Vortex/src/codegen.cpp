#include "vortex_v1/codegen.h"

#include "codegen_internal.h"

#include <filesystem>
#include <stdexcept>

namespace vortex {

CodegenResult emit_codegen(const VortexModel& model, const CodegenOptions& options) {
  if (options.output_dir.empty()) {
    throw std::runtime_error("Codegen output_dir is required");
  }
  if (options.name.empty()) {
    throw std::runtime_error("Codegen name is required");
  }

  CodegenResult result;
  result.output_dir = options.output_dir;

  std::string ident = codegen_internal::sanitize_identifier(options.name);
  std::filesystem::path codegen_dir = options.output_dir / "codegen";
  std::filesystem::path include_dir = codegen_dir / "include";
  std::filesystem::path src_dir = codegen_dir / "src";
  std::filesystem::path runtime_dir = codegen_dir / "runtime";
  std::filesystem::path runtime_include = runtime_dir / "include";
  std::filesystem::path runtime_src = runtime_dir / "src";

  std::filesystem::create_directories(include_dir);
  std::filesystem::create_directories(src_dir);
  std::filesystem::create_directories(runtime_include / "vortex_v1" / "detail");
  std::filesystem::create_directories(runtime_include / "rm_model");
  std::filesystem::create_directories(runtime_src);

  result.model_path = options.output_dir / "model.bin";
  model.write(result.model_path);
  result.centroid_index_path = options.output_dir / codegen_internal::kCentroidIndexFile;
  codegen_internal::write_centroid_index_file(
      model, result.centroid_index_path, codegen_internal::kDefaultCentroidIndexPivots);

  result.include_path = include_dir / (ident + ".h");
  result.src_path = src_dir / (ident + ".cpp");
  codegen_internal::write_text(result.include_path, codegen_internal::generate_header(ident));
  codegen_internal::write_text(result.src_path,
                               codegen_internal::generate_source(ident, options.name));

  if (options.emit_cmake) {
    result.cmake_path = codegen_dir / "CMakeLists.txt";
    codegen_internal::write_text(result.cmake_path, codegen_internal::generate_cmake(ident));
  }
  codegen_internal::write_text(codegen_dir / "README.md",
                               codegen_internal::generate_readme(ident, options.name));
  codegen_internal::copy_runtime_sources(runtime_include, runtime_src);

  result.manifest_path = options.output_dir / "vortex.json";
  codegen_internal::write_manifest(model, options, ident, result.manifest_path);
  return result;
}

CodegenResult emit_codegen_from_file(const std::filesystem::path& model_path,
                                     const CodegenOptions& options) {
  VortexModel model = VortexModel::read(model_path);
  return emit_codegen(model, options);
}

} // namespace vortex
