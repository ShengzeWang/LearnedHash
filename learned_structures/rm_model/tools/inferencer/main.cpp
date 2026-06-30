#include "rm_model/data_loader.h"

#include "rm_model/detail/cli_args.h"
#include "rm_model/json.h"
#include "rm_model/logging.h"

#include "rm_model/benchmark.h"
#include "rm_model/detail/runtime_loader.h"

#include <cmath>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using rm_model::DataType;
using rm_model::MappedDataset;
using rm_model::load_data;
using rm_model::detail::has_help_flag;
using rm_model::detail::is_flag;
using rm_model::detail::parse_double_arg;
using rm_model::detail::parse_size_arg;
using rm_model::detail::parse_u32_arg;
using rm_model::detail::parse_u64_arg;
using rm_model::detail::require_value;

namespace {

enum class ModelKeyType {
  U64 = 0,
  U32 = 1,
  F64 = 2,
};

enum class PredPrecision {
  Int,
  Double,
};

struct Options {
  std::optional<std::filesystem::path> model_dir;
  std::optional<std::filesystem::path> model_lib;
  std::optional<std::filesystem::path> data_dir;
  std::optional<std::filesystem::path> dataset;
  std::optional<std::string> key_value;
  std::optional<ModelKeyType> key_type;
  PredPrecision pred_precision = PredPrecision::Int;
  std::optional<std::filesystem::path> output;
  std::size_t limit = 0;
};

using SteadyClock = std::chrono::steady_clock;
using rm_model::TimingStats;
using rm_model::format_duration_ns;

struct ManifestPaths {
  std::optional<std::filesystem::path> src;
  std::optional<std::filesystem::path> library;
  std::optional<std::filesystem::path> data_dir;
  std::optional<std::filesystem::path> include_dir;
};

struct ManifestInfo {
  std::optional<std::string> name;
  std::optional<ModelKeyType> key_type;
  ManifestPaths paths;
};

void print_usage(std::ostream& out) {
  out << "Usage: rm_model_inferencer --model-dir <dir> [--model-lib <path>] "
      << "[--data-dir <dir>] [--dataset <file> | --key <value>] "
      << "[--key-type <u64|u32|f64>] [--pred-precision <int|double>] "
      << "[--out <file>] [--limit <n>]\n";
}

ModelKeyType parse_key_type(const std::string& value) {
  std::string lowered;
  lowered.reserve(value.size());
  for (unsigned char ch : value) {
    lowered.push_back(static_cast<char>(std::tolower(ch)));
  }
  if (lowered == "u64" || lowered == "uint64") return ModelKeyType::U64;
  if (lowered == "u32" || lowered == "uint32") return ModelKeyType::U32;
  if (lowered == "f64" || lowered == "double") return ModelKeyType::F64;
  throw std::runtime_error("Unknown key type: " + value);
}

PredPrecision parse_pred_precision(const std::string& value) {
  std::string lowered;
  lowered.reserve(value.size());
  for (unsigned char ch : value) {
    lowered.push_back(static_cast<char>(std::tolower(ch)));
  }
  if (lowered == "int" || lowered == "integer") return PredPrecision::Int;
  if (lowered == "double" || lowered == "float" || lowered == "fp64") return PredPrecision::Double;
  throw std::runtime_error("Unknown prediction precision: " + value);
}

std::optional<ModelKeyType> parse_key_type_optional(const std::string& value) {
  if (value.empty()) return std::nullopt;
  return parse_key_type(value);
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (!is_flag(arg)) {
      throw std::runtime_error("Unexpected positional arg: " + arg);
    }

    if (arg == "--model-dir") {
      opts.model_dir = require_value(i, argc, argv, arg);
    } else if (arg == "--model-lib") {
      opts.model_lib = require_value(i, argc, argv, arg);
    } else if (arg == "--data-dir") {
      opts.data_dir = require_value(i, argc, argv, arg);
    } else if (arg == "--dataset") {
      opts.dataset = require_value(i, argc, argv, arg);
    } else if (arg == "--key") {
      opts.key_value = require_value(i, argc, argv, arg);
    } else if (arg == "--key-type") {
      opts.key_type = parse_key_type(require_value(i, argc, argv, arg));
    } else if (arg == "--pred-precision") {
      opts.pred_precision = parse_pred_precision(require_value(i, argc, argv, arg));
    } else if (arg == "--out") {
      opts.output = require_value(i, argc, argv, arg);
    } else if (arg == "--limit") {
      opts.limit = parse_size_arg(require_value(i, argc, argv, arg), arg);
    } else if (arg == "--help" || arg == "-h") {
      std::ostringstream oss;
      oss << "Usage: rm_model_inferencer --model-dir <dir> [--model-lib <path>] "
          << "[--data-dir <dir>] [--dataset <file> | --key <value>] "
          << "[--key-type <u64|u32|f64>] [--pred-precision <int|double>] "
          << "[--out <file>] [--limit <n>]\n";
      throw std::runtime_error(oss.str());
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  if (!opts.model_dir && !opts.model_lib) {
    throw std::runtime_error("Must provide --model-dir or --model-lib");
  }
  if (!opts.model_dir && !opts.data_dir) {
    throw std::runtime_error("Provide --data-dir when using --model-lib without --model-dir");
  }
  if (!opts.dataset && !opts.key_value) {
    throw std::runtime_error("Must provide --dataset or --key");
  }
  if (opts.dataset && opts.key_value) {
    throw std::runtime_error("Specify only one of --dataset or --key");
  }

  return opts;
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

std::vector<std::filesystem::path> source_candidates(const std::filesystem::path& dir,
                                                     const std::string& ns) {
  std::vector<std::filesystem::path> candidates;
  if (ns.empty()) return candidates;
  candidates.push_back(dir / (ns + ".cpp"));
  candidates.push_back(dir / "src" / (ns + ".cpp"));
  return candidates;
}

std::filesystem::path find_source_path(const std::filesystem::path& dir, const std::string& ns) {
  for (const auto& candidate : source_candidates(dir, ns)) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
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
#else
  cmd << compiler_arg << " -std=c++17 -O3 -shared -fPIC -I "
      << include_dir_arg << " -I "
      << model_dir_arg << " "
      << cpp_path_arg << " -o "
      << lib_path_arg;
#endif

  RM_MODEL_LOG_INFO("Building model library: " << lib_path.string());
  rm_model::detail::run_command(cmd.str(), "Failed to build model library. Command: ");
  if (!std::filesystem::exists(lib_path)) {
    throw std::runtime_error("Failed to build model library. Command: " + cmd.str());
  }
  return lib_path;
}

struct ModelApi {
  using LoadFn = bool (*)(const char*);
  using CleanupFn = void (*)();
  using NameFn = const char* (*)();
  using KeyTypeFn = int (*)();
  using LookupU64Fn = uint64_t (*)(uint64_t, size_t*);
  using LookupU32Fn = uint64_t (*)(uint32_t, size_t*);
  using LookupF64Fn = uint64_t (*)(double, size_t*);
  using PredictU64Fn = double (*)(uint64_t, size_t*);
  using PredictU32Fn = double (*)(uint32_t, size_t*);
  using PredictF64Fn = double (*)(double, size_t*);

  LoadFn load = nullptr;
  CleanupFn cleanup = nullptr;
  NameFn name = nullptr;
  KeyTypeFn key_type = nullptr;
  LookupU64Fn lookup_u64 = nullptr;
  LookupU32Fn lookup_u32 = nullptr;
  LookupF64Fn lookup_f64 = nullptr;
  PredictU64Fn predict_u64 = nullptr;
  PredictU32Fn predict_u32 = nullptr;
  PredictF64Fn predict_f64 = nullptr;
};

ModelApi load_model_api(const rm_model::detail::SharedLibrary& lib) {
  ModelApi api;
  api.load = lib.required_symbol<ModelApi::LoadFn>("rm_model_infer_load");
  api.cleanup = lib.required_symbol<ModelApi::CleanupFn>("rm_model_infer_cleanup");
  api.name = lib.required_symbol<ModelApi::NameFn>("rm_model_infer_name");
  api.key_type = lib.required_symbol<ModelApi::KeyTypeFn>("rm_model_infer_key_type");
  api.lookup_u64 = lib.required_symbol<ModelApi::LookupU64Fn>("rm_model_infer_lookup_u64");
  api.lookup_u32 = lib.required_symbol<ModelApi::LookupU32Fn>("rm_model_infer_lookup_u32");
  api.lookup_f64 = lib.required_symbol<ModelApi::LookupF64Fn>("rm_model_infer_lookup_f64");
  api.predict_u64 = lib.optional_symbol<ModelApi::PredictU64Fn>("rm_model_infer_predict_u64");
  api.predict_u32 = lib.optional_symbol<ModelApi::PredictU32Fn>("rm_model_infer_predict_u32");
  api.predict_f64 = lib.optional_symbol<ModelApi::PredictF64Fn>("rm_model_infer_predict_f64");
  return api;
}

ModelKeyType model_key_type(const ModelApi& api) {
  int type = api.key_type();
  if (type == 0) return ModelKeyType::U64;
  if (type == 1) return ModelKeyType::U32;
  if (type == 2) return ModelKeyType::F64;
  throw std::runtime_error("Unknown key type returned by model library");
}

std::string key_type_name(ModelKeyType type) {
  switch (type) {
    case ModelKeyType::U64:
      return "u64";
    case ModelKeyType::U32:
      return "u32";
    case ModelKeyType::F64:
      return "f64";
  }
  return "unknown";
}

const char* pred_precision_name(PredPrecision precision) {
  return precision == PredPrecision::Double ? "double" : "int";
}

template <typename T>
ModelKeyType dataset_key_type() {
  if constexpr (std::is_same_v<T, uint64_t>) return ModelKeyType::U64;
  if constexpr (std::is_same_v<T, uint32_t>) return ModelKeyType::U32;
  return ModelKeyType::F64;
}

} // namespace

int main(int argc, char** argv) {
  if (has_help_flag(argc, argv)) {
    print_usage(std::cout);
    return 0;
  }

  try {
    rm_model::init_logging();

    Options opts = parse_args(argc, argv);

    std::filesystem::path model_dir = opts.model_dir.value_or(std::filesystem::path());
    std::optional<ManifestInfo> manifest;
    std::string namespace_name;
    if (opts.model_dir.has_value()) {
      manifest = load_manifest(model_dir);
      if (manifest && manifest->name.has_value()) {
        namespace_name = *manifest->name;
      } else {
        namespace_name = detect_namespace(model_dir);
      }
    }

    std::filesystem::path lib_path;
    if (opts.model_lib.has_value()) {
      lib_path = *opts.model_lib;
    } else {
      if (manifest && manifest->paths.library.has_value() &&
          std::filesystem::exists(*manifest->paths.library)) {
        lib_path = *manifest->paths.library;
      }
      for (const auto& candidate : library_candidates(model_dir, namespace_name)) {
        if (std::filesystem::exists(candidate)) {
          lib_path = candidate;
          break;
        }
      }
      if (lib_path.empty()) {
        lib_path = build_shared_library(model_dir, namespace_name,
                                        manifest ? &*manifest : nullptr);
      }
    }

    rm_model::detail::SharedLibrary library(
        lib_path,
        rm_model::detail::SharedLibraryLoadMode::Now,
        "Unable to load model library: " + lib_path.string(),
        true);
    ModelApi api = load_model_api(library);
    ModelKeyType model_type = model_key_type(api);
    RM_MODEL_LOG_INFO("Loaded model library: " << lib_path.string());
    RM_MODEL_LOG_INFO("Model name: " << api.name());
    RM_MODEL_LOG_INFO("Model key type: " << key_type_name(model_type));

    bool use_double_pred = opts.pred_precision == PredPrecision::Double;
    RM_MODEL_LOG_INFO("Prediction precision: " << pred_precision_name(opts.pred_precision));
    if (use_double_pred) {
      bool missing = false;
      if (model_type == ModelKeyType::U64 && !api.predict_u64) missing = true;
      if (model_type == ModelKeyType::U32 && !api.predict_u32) missing = true;
      if (model_type == ModelKeyType::F64 && !api.predict_f64) missing = true;
      if (missing) {
        throw std::runtime_error("Model library does not expose predict() for requested precision");
      }
    }

    std::filesystem::path data_dir;
    if (opts.data_dir.has_value()) {
      data_dir = *opts.data_dir;
    } else if (manifest && manifest->paths.data_dir.has_value()) {
      data_dir = *manifest->paths.data_dir;
    } else {
      data_dir = model_dir / "data";
    }
    if (!std::filesystem::exists(data_dir)) {
      throw std::runtime_error("Data directory does not exist: " + data_dir.string());
    }
    if (!api.load(data_dir.string().c_str())) {
      throw std::runtime_error("Model load() returned false");
    }

    struct CleanupGuard {
      ModelApi* api;
      ~CleanupGuard() { api->cleanup(); }
    } cleanup_guard{&api};

    TimingStats timing;

    const char* pred_column = use_double_pred ? "pred_f64" : "pred";

    if (opts.key_value.has_value()) {
      ModelKeyType key_type = opts.key_type.value_or(model_type);
      if (key_type != model_type) {
        throw std::runtime_error("Key type does not match model key type (" +
                                 key_type_name(model_type) + ")");
      }
      size_t err = 0;

      std::ostream* out = &std::cout;
      std::ofstream out_file;
      if (opts.output.has_value()) {
        out_file.open(*opts.output);
        if (!out_file) {
          throw std::runtime_error("Unable to open output file");
        }
        out = &out_file;
      }
      *out << "key," << pred_column << ",err\n";

      if (use_double_pred) {
        double pred = 0.0;
        if (key_type == ModelKeyType::U64) {
          uint64_t key = parse_u64_arg(*opts.key_value, "--key");
          auto start = SteadyClock::now();
          pred = api.predict_u64(key, &err);
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
        } else if (key_type == ModelKeyType::U32) {
          uint32_t key = parse_u32_arg(*opts.key_value, "--key");
          auto start = SteadyClock::now();
          pred = api.predict_u32(key, &err);
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
        } else {
          double key = parse_double_arg(*opts.key_value, "--key");
          auto start = SteadyClock::now();
          pred = api.predict_f64(key, &err);
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
        }
        *out << std::setprecision(17) << *opts.key_value << "," << pred << "," << err << "\n";
      } else {
        uint64_t pred = 0;
        if (key_type == ModelKeyType::U64) {
          uint64_t key = parse_u64_arg(*opts.key_value, "--key");
          auto start = SteadyClock::now();
          pred = api.lookup_u64(key, &err);
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
        } else if (key_type == ModelKeyType::U32) {
          uint32_t key = parse_u32_arg(*opts.key_value, "--key");
          auto start = SteadyClock::now();
          pred = api.lookup_u32(key, &err);
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
        } else {
          double key = parse_double_arg(*opts.key_value, "--key");
          auto start = SteadyClock::now();
          pred = api.lookup_f64(key, &err);
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
        }
        *out << *opts.key_value << "," << pred << "," << err << "\n";
      }
      if (!timing.empty()) {
        RM_MODEL_LOG_INFO("Inference time (model only): total "
                          << format_duration_ns(timing.total_ns())
                          << ", avg " << format_duration_ns(timing.avg_ns())
                          << " per key, min " << format_duration_ns(timing.min_ns())
                          << ", p50 " << format_duration_ns(timing.p50_ns())
                          << ", p95 " << format_duration_ns(timing.p95_ns())
                          << ", p99 " << format_duration_ns(timing.p99_ns())
                          << ", max " << format_duration_ns(timing.max_ns()));
      }
      return 0;
    }

    std::string dataset_path = opts.dataset->string();
    ModelKeyType desired_type = opts.key_type.value_or(model_type);
    DataType dt = DataType::UINT64;
    if (desired_type == ModelKeyType::U32) {
      dt = DataType::UINT32;
    } else if (desired_type == ModelKeyType::F64) {
      dt = DataType::FLOAT64;
    }

    auto load_result = load_data(dataset_path, dt);
    std::size_t num_rows = load_result.first;
    MappedDataset dataset = std::move(load_result.second);
    RM_MODEL_LOG_INFO("Loaded dataset (" << num_rows << " rows)");

    std::ostream* out = &std::cout;
    std::ofstream out_file;
    if (opts.output.has_value()) {
      out_file.open(*opts.output);
      if (!out_file) {
        throw std::runtime_error("Unable to open output file");
      }
      out = &out_file;
    }

    std::size_t limit = opts.limit;
    bool limit_enabled = limit > 0;

    *out << "key," << pred_column << ",err\n";

    dataset.visit([&](auto& typed_data) {
      using DataT = std::decay_t<decltype(typed_data)>;
      using KeyT = typename DataT::value_type::first_type;
      ModelKeyType data_type = dataset_key_type<KeyT>();
      if (data_type != model_type) {
        throw std::runtime_error("Dataset key type (" + key_type_name(data_type) +
                                 ") does not match model key type (" +
                                 key_type_name(model_type) + ")");
      }

      std::size_t count = 0;
      if (use_double_pred) {
        for (const auto& [key, _offset] : typed_data.iter()) {
          size_t err = 0;
          double pred = 0.0;
          auto start = SteadyClock::now();
          if constexpr (std::is_same_v<KeyT, uint64_t>) {
            pred = api.predict_u64(key, &err);
          } else if constexpr (std::is_same_v<KeyT, uint32_t>) {
            pred = api.predict_u32(key, &err);
          } else {
            pred = api.predict_f64(key, &err);
          }
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));

          if constexpr (std::is_same_v<KeyT, double>) {
            *out << std::setprecision(17) << key << "," << pred << "," << err << "\n";
          } else {
            *out << key << "," << std::setprecision(17) << pred << "," << err << "\n";
          }

          count += 1;
          if (limit_enabled && count >= limit) {
            break;
          }
        }
      } else {
        for (const auto& [key, _offset] : typed_data.iter()) {
          size_t err = 0;
          uint64_t pred = 0;
          auto start = SteadyClock::now();
          if constexpr (std::is_same_v<KeyT, uint64_t>) {
            pred = api.lookup_u64(key, &err);
          } else if constexpr (std::is_same_v<KeyT, uint32_t>) {
            pred = api.lookup_u32(key, &err);
          } else {
            pred = api.lookup_f64(key, &err);
          }
          auto end = SteadyClock::now();
          timing.add(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));

          if constexpr (std::is_same_v<KeyT, double>) {
            *out << std::setprecision(17) << key << "," << pred << "," << err << "\n";
          } else {
            *out << key << "," << pred << "," << err << "\n";
          }

          count += 1;
          if (limit_enabled && count >= limit) {
            break;
          }
        }
      }
    });

    if (!timing.empty()) {
      RM_MODEL_LOG_INFO("Inference time (model only): total "
                        << format_duration_ns(timing.total_ns())
                        << ", avg " << format_duration_ns(timing.avg_ns())
                        << " per key, min " << format_duration_ns(timing.min_ns())
                        << ", p50 " << format_duration_ns(timing.p50_ns())
                        << ", p95 " << format_duration_ns(timing.p95_ns())
                        << ", p99 " << format_duration_ns(timing.p99_ns())
                        << ", max " << format_duration_ns(timing.max_ns()));
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
