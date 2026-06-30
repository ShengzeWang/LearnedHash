#include "codegen_internal.h"

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vortex::codegen_internal {

std::string generate_cmake(const std::string& ident) {
  std::ostringstream out;
  out << "cmake_minimum_required(VERSION 3.16)\n";
  out << "project(" << ident << "_vortex_codegen LANGUAGES CXX)\n\n";
  out << "set(CMAKE_CXX_STANDARD 17)\n";
  out << "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n\n";
  out << "add_library(" << ident << "_vortex_codegen SHARED\n";
  out << "  src/" << ident << ".cpp\n";
  out << "  runtime/src/model.cpp\n";
  out << "  runtime/src/uint256.cpp\n";
  out << "  runtime/src/centroid_router.cpp\n";
  out << ")\n";
  out << "target_include_directories(" << ident << "_vortex_codegen PRIVATE include runtime/include)\n";
  return out.str();
}

std::string generate_readme(const std::string& ident, const std::string& user_name) {
  std::ostringstream out;
  out << "# Vortex Generated Hash\n\n";
  out << "This directory is a portable, inference-only Vortex hash artifact generated from\n";
  out << "model `" << user_name << "`.\n\n";
  out << "## Contents\n\n";
  out << "- `../model.bin`: serialized trained model parameters.\n";
  out << "- `../centroid_index.bin`: prebuilt centroid pivot index (default pivots=16).\n";
  out << "- `../vortex.json`: manifest (model metadata + file paths).\n";
  out << "- `include/" << ident << ".h`: public C++ and C ABI header.\n";
  out << "- `src/" << ident << ".cpp`: generated wrapper implementation.\n";
  out << "- `runtime/`: embedded inference runtime (`model`, `centroid_router`, shared `uint256`).\n";
  out << "- `CMakeLists.txt`: standalone build script for this generated library.\n\n";
  out << "## Build\n\n";
  out << "Run from this `codegen/` directory:\n\n";
  out << "```bash\n";
  out << "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n";
  out << "cmake --build build --config Release\n";
  out << "```\n\n";
  out << "Produced shared library:\n";
  out << "- macOS: `build/lib" << ident << "_vortex_codegen.dylib`\n";
  out << "- Linux: `build/lib" << ident << "_vortex_codegen.so`\n";
  out << "- Windows: `build/" << ident << "_vortex_codegen.dll`\n\n";
  out << "## C++ Usage\n\n";
  out << "```cpp\n";
  out << "#include \"" << ident << ".h\"\n\n";
  out << "#include <iostream>\n";
  out << "#include <stdexcept>\n";
  out << "#include <vector>\n\n";
  out << "int main() {\n";
  out << "  try {\n";
  out << "    vortex_codegen::" << ident << "::Hasher hasher;\n";
  out << "    vortex_codegen::" << ident << "::LoadOptions load_opts;\n";
  out << "    load_opts.enable_centroid_index = true; // default\n";
  out << "    load_opts.centroid_index_pivots = 16;   // default\n";
  out << "    // Directory that contains model.bin (artifact root, not codegen/).\n";
  out << "    hasher.load_from_dir(\"../\", load_opts);\n\n";
  out << "    const uint32_t dim = hasher.dim();\n";
  out << "    std::vector<float> vector(dim, 0.0f);\n\n";
  out << "    std::vector<vortex_codegen::" << ident << "::AnnNeighbor> neighbors;\n";
  out << "    auto hc = hasher.hash_with_centroid(vector.data());\n";
  out << "    hasher.ann_query(vector.data(), 8, &neighbors, hc.cache_token);\n\n";
  out << "    if (hasher.hash_bits() <= 64) {\n";
  out << "      uint64_t h = hasher.hash64(vector.data());\n";
  out << "      std::cout << h << \"\\n\";\n";
  out << "    } else {\n";
  out << "      auto h = hasher.hash(vector.data());\n";
  out << "      std::cout << h.words[0] << \"\\n\"; // least-significant 64 bits\n";
  out << "    }\n";
  out << "  } catch (const std::exception& e) {\n";
  out << "    std::cerr << e.what() << \"\\n\";\n";
  out << "    return 1;\n";
  out << "  }\n";
  out << "  return 0;\n";
  out << "}\n";
  out << "```\n\n";
  out << "## C ABI Usage\n\n";
  out << "```c\n";
  out << "#include \"" << ident << ".h\"\n\n";
  out << "#include <stdlib.h>\n\n";
  out << "int main(void) {\n";
  out << "  // Directory containing model.bin.\n";
  out << "  if (!" << ident << "_vortex_infer_load_ex(\"../\", 1, 16)) {\n";
  out << "    return 1;\n";
  out << "  }\n\n";
  out << "  uint32_t dim = " << ident << "_vortex_infer_dim();\n";
  out << "  float* vector = (float*)calloc(dim, sizeof(float));\n";
  out << "  if (!vector) {\n";
  out << "    " << ident << "_vortex_infer_cleanup();\n";
  out << "    return 1;\n";
  out << "  }\n";
  out << "  uint64_t words[4] = {0, 0, 0, 0};\n";
  out << "  uint32_t nearest = 0;\n";
  out << "  float nearest_d2 = 0.0f;\n";
  out << "  uint64_t cache_token = 0;\n";
  out << "  if (!" << ident
      << "_vortex_infer_hash_with_centroid(vector, words, &nearest, &nearest_d2, &cache_token)) {\n";
  out << "    free(vector);\n";
  out << "    " << ident << "_vortex_infer_cleanup();\n";
  out << "    return 1;\n";
  out << "  }\n\n";
  out << "  uint32_t ids[8] = {0};\n";
  out << "  float dist2[8] = {0};\n";
  out << "  uint32_t count = 0;\n";
  out << "  " << ident
      << "_vortex_infer_ann_query_cached(vector, 8, ids, dist2, 8, &count, cache_token);\n\n";
  out << "  free(vector);\n";
  out << "  " << ident << "_vortex_infer_cleanup();\n";
  out << "  return 0;\n";
  out << "}\n";
  out << "```\n\n";
  out << "## API Notes\n\n";
  out << "- Input vector length must be exactly `dim()` floats.\n";
  out << "- Centroid index loading is enabled by default; disable with `load(..., {false, ...})`\n";
  out << "  or `" << ident << "_vortex_infer_load_ex(..., 0, ...)`.\n";
  out << "- Query cache tokens are enabled by default and returned by `hash_with_centroid()` /\n";
  out << "  `" << ident << "_vortex_infer_hash_with_centroid(...)`. Reuse token in\n";
  out << "  `ann_query(..., cache_token)` / `" << ident << "_vortex_infer_ann_query_cached(...)`\n";
  out << "  to skip repeated query-to-pivot distance computation. Tokens are valid only for\n";
  out << "  the same loaded hasher state and the same query vector contents.\n";
  out << "- The generated hasher reuses a prebuilt `centroid_index.bin` when available and\n";
  out << "  falls back to rebuilding in-memory index if missing or incompatible.\n";
  out << "- Active/pivot centroid vectors are packed into contiguous aligned buffers at load\n";
  out << "  time to improve SIMD/cache behavior.\n";
  out << "- `hash64()` is valid only when `hash_bits <= 64`.\n";
  out << "- `ann_query(..., cache_token)` / `" << ident
      << "_vortex_infer_ann_query_cached(...)` returns nearest centroids\n";
  out << "  with exact L2 re-ranking and pivot-bound pruning.\n";
  out << "- `hash()` / `" << ident << "_vortex_infer_hash()` always return 256-bit words, where\n";
  out << "  `words[0]` is least significant.\n";
  out << "- C ABI uses one global hasher instance inside the library. Treat load/cleanup\n";
  out << "  as process-global operations.\n\n";
  out << "## Troubleshooting\n\n";
  out << "- `load(\".\")` fails from `codegen/`: use `\"../\"` or an absolute artifact root path.\n";
  out << "- If your model has `hash_bits > 64`, do not call `hash64()`.\n";
  out << "- If `dim()` is not what you expect, verify you loaded the intended `model.bin`.\n";
  return out.str();
}


void copy_runtime_sources(const std::filesystem::path& runtime_include,
                          const std::filesystem::path& runtime_src) {
  std::filesystem::path repo_root;
  if (const char* env_root = std::getenv("VORTEX_ROOT"); env_root && *env_root) {
    repo_root = std::filesystem::path(env_root);
  } else {
    repo_root = std::filesystem::current_path();
  }
  auto has_repo_marker = [&](const std::filesystem::path& root) {
    return std::filesystem::exists(
               root / "hash_functions_core" / "Vortex" / "include" / "vortex_v1" / "model.h") &&
           std::filesystem::exists(
               root / "hash_functions_core" / "Vortex" / "src" / "model.cpp") &&
           std::filesystem::exists(
               root / "hash_functions_core" / "Vortex" / "include" / "vortex_v1" /
               "detail" / "binary_io.h") &&
           std::filesystem::exists(
               root / "learned_structures" / "rm_model" / "include" / "rm_model" /
               "uint256.h") &&
           std::filesystem::exists(
               root / "learned_structures" / "rm_model" / "src" / "rm_model" /
               "uint256.cpp") &&
           std::filesystem::exists(
               root / "learned_structures" / "rm_model" / "src" / "rm_model" /
               "centroid_router.cpp");
  };

  if (!has_repo_marker(repo_root)) {
    std::filesystem::path probe = repo_root;
    bool found = false;
    for (int depth = 0; depth < 6; ++depth) {
      if (has_repo_marker(probe)) {
        repo_root = probe;
        found = true;
        break;
      }
      if (!probe.has_parent_path()) {
        break;
      }
      probe = probe.parent_path();
    }
    if (!found) {
      throw std::runtime_error("Unable to locate repo root for runtime embedding");
    }
  }

  auto copy_file = [](const std::filesystem::path& from, const std::filesystem::path& to) {
    if (!std::filesystem::exists(from)) {
      throw std::runtime_error("Missing runtime source: " + from.string());
    }
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing);
  };

  copy_file(repo_root / "hash_functions_core" / "Vortex" / "include" / "vortex_v1" / "model.h",
            runtime_include / "vortex_v1" / "model.h");
  copy_file(repo_root / "hash_functions_core" / "Vortex" / "include" / "vortex_v1" / "types.h",
            runtime_include / "vortex_v1" / "types.h");
  copy_file(repo_root / "hash_functions_core" / "Vortex" / "include" / "vortex_v1" / "uint256.h",
            runtime_include / "vortex_v1" / "uint256.h");
  copy_file(repo_root / "hash_functions_core" / "Vortex" / "include" / "vortex_v1" /
                "detail" / "binary_io.h",
            runtime_include / "vortex_v1" / "detail" / "binary_io.h");
  copy_file(repo_root / "learned_structures" / "rm_model" / "include" / "rm_model" / "centroid_router.h",
            runtime_include / "rm_model" / "centroid_router.h");
  copy_file(repo_root / "learned_structures" / "rm_model" / "include" / "rm_model" / "uint256.h",
            runtime_include / "rm_model" / "uint256.h");
  copy_file(repo_root / "hash_functions_core" / "Vortex" / "src" / "model.cpp",
            runtime_src / "model.cpp");
  copy_file(repo_root / "learned_structures" / "rm_model" / "src" / "rm_model" / "uint256.cpp",
            runtime_src / "uint256.cpp");
  copy_file(repo_root / "learned_structures" / "rm_model" / "src" / "rm_model" / "centroid_router.cpp",
            runtime_src / "centroid_router.cpp");
}

} // namespace vortex::codegen_internal
