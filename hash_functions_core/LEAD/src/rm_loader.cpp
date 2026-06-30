#include "lead/rm_loader.h"

#include "rm_model/detail/runtime_loader.h"
#include "rm_model/json.h"

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lead {

namespace {

struct ManifestPaths {
  std::optional<std::filesystem::path> src;
  std::optional<std::filesystem::path> library;
  std::optional<std::filesystem::path> data_dir;
  std::optional<std::filesystem::path> include_dir;
};

struct ManifestInfo {
  std::string name;
  std::optional<RmKeyType> key_type;
  ManifestPaths paths;
};

std::optional<RmKeyType> parse_key_type_optional(const std::string& value) {
  if (value == "u64") return RmKeyType::U64;
  if (value == "u32") return RmKeyType::U32;
  if (value == "f64") return RmKeyType::F64;
  return std::nullopt;
}

std::optional<std::filesystem::path> resolve_manifest_path(
    const std::filesystem::path& root,
    const rm_model::json::Value* value) {
  if (!value || !value->is_string()) return std::nullopt;
  std::filesystem::path path(value->as_string());
  if (path.is_relative()) {
    path = root / path;
  }
  return path;
}

std::optional<ManifestInfo> load_manifest(const std::filesystem::path& model_dir) {
  std::filesystem::path manifest_path = model_dir / "model.json";
  if (!std::filesystem::exists(manifest_path)) {
    return std::nullopt;
  }

  auto root = rm_model::json::parse_file(manifest_path.string());
  if (!root.is_object()) {
    throw std::runtime_error("model.json must contain an object");
  }

  ManifestInfo info;
  if (const auto* name_val = root.find("name"); name_val && name_val->is_string()) {
    info.name = name_val->as_string();
  }
  if (const auto* key_val = root.find("key_type"); key_val && key_val->is_string()) {
    info.key_type = parse_key_type_optional(key_val->as_string());
  }

  if (const auto* paths_val = root.find("paths"); paths_val && paths_val->is_object()) {
    if (const auto* src_val = paths_val->find("src")) {
      info.paths.src = resolve_manifest_path(model_dir, src_val);
    }
    if (const auto* lib_val = paths_val->find("library")) {
      info.paths.library = resolve_manifest_path(model_dir, lib_val);
    }
    if (const auto* data_val = paths_val->find("data_dir")) {
      info.paths.data_dir = resolve_manifest_path(model_dir, data_val);
    }
    if (const auto* include_dir_val = paths_val->find("include_dir")) {
      info.paths.include_dir = resolve_manifest_path(model_dir, include_dir_val);
    } else if (const auto* includes_val = paths_val->find("include");
               includes_val && includes_val->is_array() && !includes_val->as_array().empty()) {
      const auto& first = includes_val->as_array().front();
      if (first.is_string()) {
        auto path = resolve_manifest_path(model_dir, &first);
        if (path.has_value()) {
          info.paths.include_dir = path->parent_path();
        }
      }
    }
  }

  return info;
}

std::string detect_namespace(const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> search_dirs;
  auto include_dir = dir / "include";
  if (std::filesystem::exists(include_dir) && std::filesystem::is_directory(include_dir)) {
    search_dirs.push_back(include_dir);
  }
  search_dirs.push_back(dir);

  std::string detected;
  for (const auto& scan_dir : search_dirs) {
    for (const auto& entry : std::filesystem::directory_iterator(scan_dir)) {
      if (!entry.is_regular_file()) continue;
      auto name = entry.path().filename().string();
      const std::string suffix = "_data.h";
      if (name.size() <= suffix.size()) continue;
      if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
      std::string ns = name.substr(0, name.size() - suffix.size());
      if (!detected.empty() && detected != ns) {
        throw std::runtime_error("Multiple model headers found in directory");
      }
      detected = ns;
    }
    if (!detected.empty()) break;
  }
  return detected;
}

std::vector<std::filesystem::path> library_search_dirs(const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> dirs;
  auto lib_dir = dir / "lib";
  if (std::filesystem::exists(lib_dir) && std::filesystem::is_directory(lib_dir)) {
    dirs.push_back(lib_dir);
  }
  dirs.push_back(dir);
  return dirs;
}

std::vector<std::filesystem::path> library_candidates(const std::filesystem::path& dir,
                                                      const std::string& ns) {
  std::vector<std::string> bases;
  if (!ns.empty()) bases.push_back(ns);
  bases.push_back("model");

#if defined(_WIN32)
  std::vector<std::string> exts = {".dll"};
#elif defined(__APPLE__)
  std::vector<std::string> exts = {".dylib"};
#else
  std::vector<std::string> exts = {".so"};
#endif

  std::vector<std::filesystem::path> candidates;
  for (const auto& search_dir : library_search_dirs(dir)) {
    for (const auto& base : bases) {
      for (const auto& ext : exts) {
        candidates.push_back(search_dir / (base + ext));
        candidates.push_back(search_dir / ("lib" + base + ext));
      }
    }
  }
  return candidates;
}

std::filesystem::path find_source_path(const std::filesystem::path& dir, const std::string& ns) {
  if (ns.empty()) return {};
  std::filesystem::path candidate = dir / (ns + ".cpp");
  if (std::filesystem::exists(candidate)) return candidate;
  candidate = dir / "src" / (ns + ".cpp");
  if (std::filesystem::exists(candidate)) return candidate;
  return {};
}

std::string shared_library_name(const std::string& ns) {
#if defined(_WIN32)
  return ns + ".dll";
#elif defined(__APPLE__)
  return "lib" + ns + ".dylib";
#else
  return "lib" + ns + ".so";
#endif
}

std::string default_compiler() {
  if (const char* env = std::getenv("RM_MODEL_CXX")) {
    return env;
  }
  if (const char* env = std::getenv("CXX")) {
    return env;
  }
#if defined(_WIN32)
  return "cl";
#else
  return "c++";
#endif
}

std::filesystem::path build_shared_library(const std::filesystem::path& model_dir,
                                           const std::string& ns,
                                           const ManifestInfo* manifest) {
  if (ns.empty()) {
    throw std::runtime_error("Cannot build model library without a namespace");
  }

  std::filesystem::path cpp_path = manifest && manifest->paths.src.has_value()
      ? *manifest->paths.src
      : find_source_path(model_dir, ns);
  if (cpp_path.empty()) {
    throw std::runtime_error("Missing generated model source for namespace " + ns);
  }

  std::filesystem::path lib_path = manifest && manifest->paths.library.has_value()
      ? *manifest->paths.library
      : (model_dir / "lib" / shared_library_name(ns));
  std::filesystem::create_directories(lib_path.parent_path());
  std::string compiler = default_compiler();
  std::filesystem::path include_dir = manifest && manifest->paths.include_dir.has_value()
      ? *manifest->paths.include_dir
      : (model_dir / "include");
  if (!std::filesystem::exists(include_dir)) {
    include_dir = model_dir;
  }

  std::string compiler_arg = rm_model::detail::shell_quote_checked(compiler, "compiler");
  std::string include_dir_arg = rm_model::detail::shell_quote_checked(
      include_dir.string(), "include_dir");
  std::string model_dir_arg = rm_model::detail::shell_quote_checked(model_dir.string(), "model_dir");
  std::string cpp_path_arg = rm_model::detail::shell_quote_checked(cpp_path.string(), "source_path");
  std::string lib_path_arg = rm_model::detail::shell_quote_checked(lib_path.string(), "library_path");

  std::ostringstream cmd;
#if defined(_WIN32)
  cmd << compiler_arg
      << " /std:c++17 /O2 /LD /EHsc /I " << include_dir_arg
      << " /I " << model_dir_arg << " "
      << cpp_path_arg
      << " /link /OUT:" << lib_path_arg;
#elif defined(__APPLE__)
  cmd << compiler_arg
      << " -std=c++17 -O3 -dynamiclib -I " << include_dir_arg
      << " -I " << model_dir_arg << " "
      << cpp_path_arg << " -o "
      << lib_path_arg;
#else
  cmd << compiler_arg
      << " -std=c++17 -O3 -shared -fPIC -I " << include_dir_arg
      << " -I " << model_dir_arg << " "
      << cpp_path_arg << " -o "
      << lib_path_arg;
#endif

  rm_model::detail::run_command(cmd.str(), "Failed to build model library. Command: ");

  return lib_path;
}

} // namespace

RmModel::RmModel() = default;

RmModel::~RmModel() {
  cleanup();
}

bool RmModel::load(const std::filesystem::path& model_dir,
                   const std::optional<std::filesystem::path>& model_lib,
                   const std::optional<std::filesystem::path>& data_dir) {
  cleanup();

  std::filesystem::path resolved_model_dir = model_dir;
  if (resolved_model_dir.empty()) {
    throw std::runtime_error("model_dir must not be empty");
  }

  auto manifest = load_manifest(resolved_model_dir);
  std::string namespace_name;
  if (manifest.has_value() && !manifest->name.empty()) {
    namespace_name = manifest->name;
  } else {
    namespace_name = detect_namespace(resolved_model_dir);
  }

  std::filesystem::path lib_path;
  if (model_lib.has_value()) {
    lib_path = *model_lib;
  } else if (manifest.has_value() && manifest->paths.library.has_value()) {
    lib_path = *manifest->paths.library;
  } else {
    for (const auto& candidate : library_candidates(resolved_model_dir, namespace_name)) {
      if (std::filesystem::exists(candidate)) {
        lib_path = candidate;
        break;
      }
    }
  }

  if (lib_path.empty() || !std::filesystem::exists(lib_path)) {
    lib_path = build_shared_library(resolved_model_dir, namespace_name,
                                    manifest.has_value() ? &*manifest : nullptr);
  }

  std::filesystem::path data_path;
  if (data_dir.has_value()) {
    data_path = *data_dir;
  } else if (manifest.has_value() && manifest->paths.data_dir.has_value()) {
    data_path = *manifest->paths.data_dir;
  } else {
    data_path = resolved_model_dir / "data";
  }

  rm_model::detail::SharedLibrary lib;
  lib.open(lib_path,
           rm_model::detail::SharedLibraryLoadMode::Lazy,
           "Failed to load model library");

  auto sym = [&](const char* name) -> void* {
    void* out = lib.symbol(name);
    if (!out) {
      throw std::runtime_error(std::string("Missing symbol: ") + name);
    }
    return out;
  };

  (void)sym("rm_model_infer_name");
  InferLoadFn infer_load = reinterpret_cast<InferLoadFn>(sym("rm_model_infer_load"));
  InferCleanupFn infer_cleanup = reinterpret_cast<InferCleanupFn>(sym("rm_model_infer_cleanup"));
  InferKeyTypeFn infer_key_type = reinterpret_cast<InferKeyTypeFn>(sym("rm_model_infer_key_type"));
  InferPredictU64Fn predict_u64 = reinterpret_cast<InferPredictU64Fn>(sym("rm_model_infer_predict_u64"));
  InferPredictU32Fn predict_u32 = reinterpret_cast<InferPredictU32Fn>(sym("rm_model_infer_predict_u32"));
  InferPredictF64Fn predict_f64 = reinterpret_cast<InferPredictF64Fn>(sym("rm_model_infer_predict_f64"));
  InferPredictRawU64Fn predict_raw_u64 = reinterpret_cast<InferPredictRawU64Fn>(
      lib.symbol("rm_model_infer_predict_raw_u64"));
  InferPredictRawU32Fn predict_raw_u32 = reinterpret_cast<InferPredictRawU32Fn>(
      lib.symbol("rm_model_infer_predict_raw_u32"));
  InferPredictRawF64Fn predict_raw_f64 = reinterpret_cast<InferPredictRawF64Fn>(
      lib.symbol("rm_model_infer_predict_raw_f64"));

  if (!infer_load(data_path.string().c_str())) {
    if (infer_cleanup) {
      infer_cleanup();
    }
    throw std::runtime_error("Failed to load model data");
  }

  int kt = infer_key_type ? infer_key_type() : 0;
  if (kt == 1) {
    key_type_ = RmKeyType::U32;
  } else if (kt == 2) {
    key_type_ = RmKeyType::F64;
  } else {
    key_type_ = RmKeyType::U64;
  }

  infer_cleanup_ = infer_cleanup;
  infer_key_type_ = infer_key_type;
  predict_u64_ = predict_u64;
  predict_u32_ = predict_u32;
  predict_f64_ = predict_f64;
  predict_raw_u64_ = predict_raw_u64;
  predict_raw_u32_ = predict_raw_u32;
  predict_raw_f64_ = predict_raw_f64;
  handle_ = lib.release();
  loaded_ = true;
  return true;
}

double RmModel::predict_u64(uint64_t key, size_t* err) const {
  if (!loaded_) {
    throw std::runtime_error("Model is not loaded");
  }
  if (!predict_u64_) {
    throw std::runtime_error("Model does not support predict() for u64");
  }
  if (err) {
    return predict_u64_(key, err);
  }
  size_t scratch = 0;
  return predict_u64_(key, &scratch);
}

double RmModel::predict_u32(uint32_t key, size_t* err) const {
  if (!loaded_) {
    throw std::runtime_error("Model is not loaded");
  }
  if (!predict_u32_) {
    throw std::runtime_error("Model does not support predict() for u32");
  }
  if (err) {
    return predict_u32_(key, err);
  }
  size_t scratch = 0;
  return predict_u32_(key, &scratch);
}

double RmModel::predict_f64(double key, size_t* err) const {
  if (!loaded_) {
    throw std::runtime_error("Model is not loaded");
  }
  if (!predict_f64_) {
    throw std::runtime_error("Model does not support predict() for f64");
  }
  if (err) {
    return predict_f64_(key, err);
  }
  size_t scratch = 0;
  return predict_f64_(key, &scratch);
}

double RmModel::predict_raw_u64(uint64_t key, size_t* err) const {
  if (!loaded_) {
    throw std::runtime_error("Model is not loaded");
  }
  if (!predict_raw_u64_) {
    throw std::runtime_error("Model does not support predict_raw() for u64");
  }
  if (err) {
    return predict_raw_u64_(key, err);
  }
  size_t scratch = 0;
  return predict_raw_u64_(key, &scratch);
}

double RmModel::predict_raw_u32(uint32_t key, size_t* err) const {
  if (!loaded_) {
    throw std::runtime_error("Model is not loaded");
  }
  if (!predict_raw_u32_) {
    throw std::runtime_error("Model does not support predict_raw() for u32");
  }
  if (err) {
    return predict_raw_u32_(key, err);
  }
  size_t scratch = 0;
  return predict_raw_u32_(key, &scratch);
}

double RmModel::predict_raw_f64(double key, size_t* err) const {
  if (!loaded_) {
    throw std::runtime_error("Model is not loaded");
  }
  if (!predict_raw_f64_) {
    throw std::runtime_error("Model does not support predict_raw() for f64");
  }
  if (err) {
    return predict_raw_f64_(key, err);
  }
  size_t scratch = 0;
  return predict_raw_f64_(key, &scratch);
}

bool RmModel::has_raw_predict(RmKeyType key_type) const {
  if (!loaded_) {
    return false;
  }
  if (key_type == RmKeyType::U64) {
    return predict_raw_u64_ != nullptr;
  }
  if (key_type == RmKeyType::U32) {
    return predict_raw_u32_ != nullptr;
  }
  if (key_type == RmKeyType::F64) {
    return predict_raw_f64_ != nullptr;
  }
  return false;
}

void RmModel::cleanup() {
  if (loaded_ && infer_cleanup_) {
    infer_cleanup_();
  }
  loaded_ = false;
  infer_cleanup_ = nullptr;
  infer_key_type_ = nullptr;
  predict_u64_ = nullptr;
  predict_u32_ = nullptr;
  predict_f64_ = nullptr;
  predict_raw_u64_ = nullptr;
  predict_raw_u32_ = nullptr;
  predict_raw_f64_ = nullptr;
  key_type_ = RmKeyType::U64;
  rm_model::detail::close_shared_library(handle_);
  handle_ = nullptr;
}

} // namespace lead
