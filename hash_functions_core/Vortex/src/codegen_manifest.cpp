#include "codegen_internal.h"

#include "rm_model/json.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace vortex::codegen_internal {

void write_manifest(const VortexModel& model,
                    const CodegenOptions& options,
                    const std::string& ident,
                    const std::filesystem::path& manifest_path) {
  rm_model::json::Value::Object root;
  root.emplace_back("schema_version", rm_model::json::Value(uint64_t{1}));
  root.emplace_back("name", rm_model::json::Value(options.name));
  root.emplace_back("codegen_name", rm_model::json::Value(ident));
  root.emplace_back("dim", rm_model::json::Value(static_cast<uint64_t>(model.dim)));
  root.emplace_back("hash_bits", rm_model::json::Value(static_cast<uint64_t>(model.hash_bits)));
  root.emplace_back("metric", rm_model::json::Value(metric_name(model.metric)));

  rm_model::json::Value::Object paths;
  paths.emplace_back("model", rm_model::json::Value(std::string("model.bin")));
  paths.emplace_back("centroid_index", rm_model::json::Value(std::string(kCentroidIndexFile)));
  paths.emplace_back("codegen_dir", rm_model::json::Value(std::string("codegen")));
  paths.emplace_back("include", rm_model::json::Value((std::filesystem::path("codegen") / "include" / (ident + ".h")).string()));
  paths.emplace_back("src", rm_model::json::Value((std::filesystem::path("codegen") / "src" / (ident + ".cpp")).string()));
  paths.emplace_back("runtime_dir", rm_model::json::Value((std::filesystem::path("codegen") / "runtime").string()));
  if (options.emit_cmake) {
    paths.emplace_back("cmake", rm_model::json::Value((std::filesystem::path("codegen") / "CMakeLists.txt").string()));
  }
  root.emplace_back("paths", rm_model::json::Value(std::move(paths)));

  std::ofstream manifest_out(manifest_path, std::ios::binary | std::ios::trunc);
  if (!manifest_out) {
    throw std::runtime_error("Unable to write manifest: " + manifest_path.string());
  }
  rm_model::json::write(manifest_out, rm_model::json::Value(std::move(root)));
}

} // namespace vortex::codegen_internal
