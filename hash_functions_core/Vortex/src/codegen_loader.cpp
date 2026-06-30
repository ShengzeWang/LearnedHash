#include "vortex_v1/codegen_loader.h"

#include "rm_model/detail/runtime_loader.h"
#include "rm_model/json.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace vortex {

namespace {

struct AnnQueryScratch {
  std::vector<uint32_t> centroids;
  std::vector<float> dist2;
};

AnnQueryScratch& ann_query_scratch() {
  static thread_local AnnQueryScratch scratch;
  return scratch;
}

std::filesystem::path resolve_path(const std::filesystem::path& root, const rm_model::json::Value* val) {
  if (!val || !val->is_string()) {
    return {};
  }
  std::filesystem::path path(val->as_string());
  if (path.is_relative()) {
    path = root / path;
  }
  return path;
}

std::string shared_library_name(const std::string& ns) {
#if defined(_WIN32)
  return ns + "_vortex_codegen.dll";
#elif defined(__APPLE__)
  return "lib" + ns + "_vortex_codegen.dylib";
#else
  return "lib" + ns + "_vortex_codegen.so";
#endif
}

std::filesystem::path default_build_dir(const std::filesystem::path& codegen_dir) {
  return codegen_dir / "build";
}

std::filesystem::path build_fingerprint_path(const std::filesystem::path& codegen_dir) {
  return default_build_dir(codegen_dir) / ".vortex_codegen_build_fingerprint";
}

std::filesystem::path default_library_path(const std::filesystem::path& codegen_dir,
                                           const std::string& ident) {
  return default_build_dir(codegen_dir) / shared_library_name(ident);
}

struct FingerprintHash {
  uint64_t value = 1469598103934665603ULL;

  void add_bytes(const char* data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      value ^= static_cast<unsigned char>(data[i]);
      value *= 1099511628211ULL;
    }
  }

  void add_string(const std::string& text) {
    add_bytes(text.data(), text.size());
    add_bytes("\0", 1);
  }
};

std::string hex_u64(uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

void hash_file(FingerprintHash* hash, const std::filesystem::path& root,
               const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::path rel = std::filesystem::relative(path, root, ec);
  hash->add_string(ec ? path.generic_string() : rel.generic_string());
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Unable to read codegen input: " + path.string());
  }
  char buffer[64 * 1024];
  while (in) {
    in.read(buffer, sizeof(buffer));
    std::streamsize count = in.gcount();
    if (count > 0) {
      hash->add_bytes(buffer, static_cast<std::size_t>(count));
    }
  }
}

std::vector<std::filesystem::path> codegen_build_inputs(const std::filesystem::path& codegen_dir) {
  std::vector<std::filesystem::path> files;
  auto add_file = [&](const std::filesystem::path& path) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(path, ec)) {
      files.push_back(path);
    }
  };
  auto add_tree = [&](const std::filesystem::path& root) {
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
      return;
    }
    for (std::filesystem::recursive_directory_iterator it(root, ec), end;
         it != end;
         it.increment(ec)) {
      if (ec) {
        throw std::runtime_error("Unable to scan codegen inputs under: " + root.string());
      }
      if (it->is_regular_file()) {
        files.push_back(it->path());
      }
    }
  };

  add_file(codegen_dir / "CMakeLists.txt");
  add_tree(codegen_dir / "include");
  add_tree(codegen_dir / "src");
  add_tree(codegen_dir / "runtime" / "include");
  add_tree(codegen_dir / "runtime" / "src");
  std::sort(files.begin(), files.end());
  return files;
}

std::string codegen_build_fingerprint(const CodegenManifest& manifest) {
  FingerprintHash hash;
  hash.add_string("vortex_codegen_build_cache_v1");
  hash.add_string(manifest.codegen_name);
  hash.add_string(shared_library_name(manifest.codegen_name));
#if defined(_WIN32)
  hash.add_string("windows");
#elif defined(__APPLE__)
  hash.add_string("apple");
#else
  hash.add_string("unix");
#endif
  for (const char* env_name : {"CMAKE_GENERATOR", "CC", "CXX", "SDKROOT",
                               "MACOSX_DEPLOYMENT_TARGET"}) {
    const char* value = std::getenv(env_name);
    hash.add_string(std::string(env_name) + "=" + (value ? value : ""));
  }
  for (const auto& file : codegen_build_inputs(manifest.codegen_dir)) {
    hash_file(&hash, manifest.codegen_dir, file);
  }
  return hex_u64(hash.value);
}

std::optional<std::filesystem::path> codegen_build_cache_root() {
  const char* value = std::getenv("VORTEX_CODEGEN_BUILD_CACHE");
  if (!value || !*value) {
    return std::nullopt;
  }
  std::filesystem::path root(value);
  if (root.empty()) {
    return std::nullopt;
  }
  return root;
}

std::optional<std::string> read_text_trimmed(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    return std::nullopt;
  }
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r' ||
                           text.back() == ' ' || text.back() == '\t')) {
    text.pop_back();
  }
  return text;
}

void write_text_file(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Unable to write: " + path.string());
  }
  out << text << "\n";
}

bool inputs_newer_than_library(const std::vector<std::filesystem::path>& inputs,
                               const std::filesystem::path& library_path) {
  std::error_code ec;
  auto library_time = std::filesystem::last_write_time(library_path, ec);
  if (ec) {
    return true;
  }
  for (const auto& input : inputs) {
    auto input_time = std::filesystem::last_write_time(input, ec);
    if (!ec && input_time > library_time) {
      return true;
    }
  }
  return false;
}

bool existing_library_is_fresh(const CodegenManifest& manifest,
                               const std::filesystem::path& library_path,
                               const std::string& fingerprint) {
  if (!std::filesystem::exists(library_path)) {
    return false;
  }
  std::filesystem::path marker = build_fingerprint_path(manifest.codegen_dir);
  if (auto recorded = read_text_trimmed(marker); recorded && *recorded == fingerprint) {
    return true;
  }
  if (inputs_newer_than_library(codegen_build_inputs(manifest.codegen_dir), library_path)) {
    return false;
  }
  write_text_file(marker, fingerprint);
  return true;
}

std::filesystem::path cached_library_path(const std::filesystem::path& cache_root,
                                          const std::string& fingerprint,
                                          const std::string& codegen_name) {
  return cache_root / fingerprint / shared_library_name(codegen_name);
}

std::filesystem::path temp_copy_path(const std::filesystem::path& path) {
  auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  return path.string() + ".tmp." + std::to_string(ticks) + "." + std::to_string(tid);
}

void copy_file_atomic(const std::filesystem::path& from, const std::filesystem::path& to) {
  std::filesystem::create_directories(to.parent_path());
  std::filesystem::path tmp = temp_copy_path(to);
  std::filesystem::copy_file(from, tmp, std::filesystem::copy_options::overwrite_existing);
  std::error_code ec;
  std::filesystem::rename(tmp, to, ec);
  if (ec) {
    std::filesystem::remove(to, ec);
    ec.clear();
    std::filesystem::rename(tmp, to, ec);
  }
  if (ec) {
    std::filesystem::remove(tmp, ec);
    throw std::runtime_error("Unable to install cached codegen library: " + to.string());
  }
}

bool restore_cached_library(const CodegenManifest& manifest,
                            const std::filesystem::path& library_path,
                            const std::string& fingerprint) {
  auto root = codegen_build_cache_root();
  if (!root.has_value()) {
    return false;
  }
  std::filesystem::path cached = cached_library_path(*root, fingerprint, manifest.codegen_name);
  if (!std::filesystem::exists(cached)) {
    return false;
  }
  copy_file_atomic(cached, library_path);
  write_text_file(build_fingerprint_path(manifest.codegen_dir), fingerprint);
  return true;
}

void store_cached_library(const CodegenManifest& manifest,
                          const std::filesystem::path& library_path,
                          const std::string& fingerprint) {
  auto root = codegen_build_cache_root();
  if (!root.has_value() || !std::filesystem::exists(library_path)) {
    return;
  }
  std::filesystem::path cached = cached_library_path(*root, fingerprint, manifest.codegen_name);
  copy_file_atomic(library_path, cached);
  write_text_file(cached.parent_path() / "fingerprint.txt", fingerprint);
}

void build_codegen_library(const std::filesystem::path& codegen_dir,
                           const std::string& codegen_name) {
  std::filesystem::path build_dir = default_build_dir(codegen_dir);
  std::filesystem::create_directories(build_dir);
  std::string codegen_dir_arg = rm_model::detail::shell_quote_checked(
      codegen_dir.string(), "codegen_dir");
  std::string build_dir_arg = rm_model::detail::shell_quote_checked(build_dir.string(), "build_dir");
  std::string target_arg = rm_model::detail::shell_quote_checked(
      codegen_name + "_vortex_codegen", "codegen_target");
  std::ostringstream configure;
  configure << "cmake -S " << codegen_dir_arg
            << " -B " << build_dir_arg
            << " -DCMAKE_BUILD_TYPE=Release";
  rm_model::detail::run_command(configure.str(), "Command failed: ");

  std::ostringstream build;
  build << "cmake --build " << build_dir_arg
        << " --config Release"
        << " --target " << target_arg
        << " --parallel";
  rm_model::detail::run_command(build.str(), "Command failed: ");
}

struct PreparedLibrary {
  std::filesystem::path path;
  std::string status;
  std::string fingerprint;
  bool restored_from_cache = false;
};

PreparedLibrary build_and_store_library(const CodegenManifest& manifest,
                                        const std::filesystem::path& lib_path,
                                        const std::string& fingerprint,
                                        const std::string& status) {
  build_codegen_library(manifest.codegen_dir, manifest.codegen_name);
  if (!std::filesystem::exists(lib_path)) {
    throw std::runtime_error("Codegen library not found after build: " + lib_path.string());
  }
  write_text_file(build_fingerprint_path(manifest.codegen_dir), fingerprint);
  store_cached_library(manifest, lib_path, fingerprint);
  return PreparedLibrary{lib_path, status, fingerprint, false};
}

PreparedLibrary prepare_codegen_library(const CodegenManifest& manifest) {
  std::filesystem::path lib_path =
      default_library_path(manifest.codegen_dir, manifest.codegen_name);
  std::string fingerprint = codegen_build_fingerprint(manifest);
  if (existing_library_is_fresh(manifest, lib_path, fingerprint)) {
    return PreparedLibrary{lib_path, "existing", fingerprint, false};
  }
  bool had_library = std::filesystem::exists(lib_path);
  if (restore_cached_library(manifest, lib_path, fingerprint)) {
    return PreparedLibrary{lib_path, "cache_hit", fingerprint, true};
  }
  return build_and_store_library(
      manifest,
      lib_path,
      fingerprint,
      codegen_build_cache_root().has_value()
          ? (had_library ? "cache_miss_rebuilt" : "cache_miss_built")
          : (had_library ? "rebuilt_stale" : "built"));
}

} // namespace

CodegenManifest load_codegen_manifest(const std::filesystem::path& manifest_path) {
  if (!std::filesystem::exists(manifest_path)) {
    throw std::runtime_error("Missing vortex.json: " + manifest_path.string());
  }
  auto root = rm_model::json::parse_file(manifest_path.string());
  if (!root.is_object()) {
    throw std::runtime_error("vortex.json must be an object");
  }
  CodegenManifest manifest;
  manifest.root_dir = manifest_path.parent_path();

  if (const auto* name_val = root.find("name"); name_val && name_val->is_string()) {
    manifest.name = name_val->as_string();
  }
  if (const auto* codegen_val = root.find("codegen_name"); codegen_val && codegen_val->is_string()) {
    manifest.codegen_name = codegen_val->as_string();
  }
  if (manifest.codegen_name.empty()) {
    throw std::runtime_error("vortex.json missing codegen_name");
  }

  const auto* paths = root.find("paths");
  if (!paths || !paths->is_object()) {
    throw std::runtime_error("vortex.json missing paths");
  }

  manifest.model_path = resolve_path(manifest.root_dir, paths->find("model"));
  manifest.centroid_index_path = resolve_path(manifest.root_dir, paths->find("centroid_index"));
  manifest.codegen_dir = resolve_path(manifest.root_dir, paths->find("codegen_dir"));
  manifest.include_path = resolve_path(manifest.root_dir, paths->find("include"));
  manifest.src_path = resolve_path(manifest.root_dir, paths->find("src"));

  if (manifest.model_path.empty() || manifest.codegen_dir.empty()) {
    throw std::runtime_error("vortex.json missing model or codegen_dir path");
  }

  return manifest;
}

CodegenHasher::~CodegenHasher() {
  unload();
}

bool CodegenHasher::load(const std::filesystem::path& codegen_dir,
                         const CodegenLoadOptions& options) {
  unload();
  std::filesystem::path manifest_path = codegen_dir / "vortex.json";
  CodegenManifest manifest = load_codegen_manifest(manifest_path);

  PreparedLibrary prepared = prepare_codegen_library(manifest);

  rm_model::detail::SharedLibrary lib;
  try {
    lib.open(prepared.path,
             rm_model::detail::SharedLibraryLoadMode::Lazy,
             "Failed to load codegen library");
  } catch (const std::exception&) {
    if (!prepared.restored_from_cache) {
      throw;
    }
    prepared = build_and_store_library(manifest,
                                       prepared.path,
                                       prepared.fingerprint,
                                       "cache_hit_rebuilt_after_load_error");
    lib.open(prepared.path,
             rm_model::detail::SharedLibraryLoadMode::Lazy,
             "Failed to load codegen library");
  }

  auto sym = [&](const std::string& suffix) -> void* {
    std::string name = manifest.codegen_name + "_vortex_" + suffix;
    void* out = lib.symbol(name.c_str());
    if (!out) {
      throw std::runtime_error("Missing symbol: " + name);
    }
    return out;
  };
  auto sym_optional = [&](const std::string& suffix) -> void* {
    std::string name = manifest.codegen_name + "_vortex_" + suffix;
    return lib.symbol(name.c_str());
  };

  (void)sym("infer_name");
  InferLoadFn infer_load = reinterpret_cast<InferLoadFn>(sym("infer_load"));
  InferLoadExFn infer_load_ex =
      reinterpret_cast<InferLoadExFn>(sym_optional("infer_load_ex"));
  InferLoadEx2Fn infer_load_ex2 =
      reinterpret_cast<InferLoadEx2Fn>(sym_optional("infer_load_ex2"));
  InferCleanupFn infer_cleanup = reinterpret_cast<InferCleanupFn>(sym("infer_cleanup"));
  InferDimFn infer_dim = reinterpret_cast<InferDimFn>(sym("infer_dim"));
  InferHashBitsFn infer_hash_bits =
      reinterpret_cast<InferHashBitsFn>(sym("infer_hash_bits"));
  InferHashFn infer_hash = reinterpret_cast<InferHashFn>(sym("infer_hash"));
  InferHash64Fn infer_hash64 = reinterpret_cast<InferHash64Fn>(sym("infer_hash64"));
  InferNearestActiveCentroidFn infer_nearest_active_centroid =
      reinterpret_cast<InferNearestActiveCentroidFn>(sym_optional("infer_nearest_active_centroid"));
  InferHashWithCentroidFn infer_hash_with_centroid =
      reinterpret_cast<InferHashWithCentroidFn>(sym_optional("infer_hash_with_centroid"));
  InferAnnQueryFn infer_ann_query =
      reinterpret_cast<InferAnnQueryFn>(sym_optional("infer_ann_query"));
  InferAnnQueryCachedFn infer_ann_query_cached =
      reinterpret_cast<InferAnnQueryCachedFn>(sym_optional("infer_ann_query_cached"));

  std::filesystem::path model_path = manifest.model_path;
  bool loaded = false;
  if (infer_load_ex2) {
    loaded = infer_load_ex2(
        model_path.string().c_str(),
        options.enable_centroid_index ? 1 : 0,
        options.centroid_index_pivots,
        options.enable_query_cache ? 1 : 0);
  } else if (infer_load_ex) {
    loaded = infer_load_ex(
        model_path.string().c_str(),
        options.enable_centroid_index ? 1 : 0,
        options.centroid_index_pivots);
  } else {
    loaded = infer_load(model_path.string().c_str());
  }
  if (!loaded) {
    if (infer_cleanup) {
      infer_cleanup();
    }
    throw std::runtime_error("Failed to load Vortex model");
  }

  infer_cleanup_ = infer_cleanup;
  infer_dim_ = infer_dim;
  infer_hash_bits_ = infer_hash_bits;
  infer_hash_ = infer_hash;
  infer_hash64_ = infer_hash64;
  infer_nearest_active_centroid_ = infer_nearest_active_centroid;
  infer_hash_with_centroid_ = infer_hash_with_centroid;
  infer_ann_query_ = infer_ann_query;
  infer_ann_query_cached_ = infer_ann_query_cached;

  dim_ = infer_dim_ ? infer_dim_() : 0;
  hash_bits_ = infer_hash_bits_ ? infer_hash_bits_() : 0;
  handle_ = lib.release();
  library_status_ = prepared.status;
  library_path_ = prepared.path;
  loaded_ = true;
  return true;
}

void CodegenHasher::unload() {
  if (loaded_ && infer_cleanup_) {
    infer_cleanup_();
  }
  rm_model::detail::close_shared_library(handle_);
  handle_ = nullptr;
  infer_cleanup_ = nullptr;
  infer_dim_ = nullptr;
  infer_hash_bits_ = nullptr;
  infer_hash_ = nullptr;
  infer_hash64_ = nullptr;
  infer_nearest_active_centroid_ = nullptr;
  infer_hash_with_centroid_ = nullptr;
  infer_ann_query_ = nullptr;
  infer_ann_query_cached_ = nullptr;
  loaded_ = false;
  dim_ = 0;
  hash_bits_ = 0;
  library_status_.clear();
  library_path_.clear();
}

bool CodegenHasher::hash(const float* vector, UInt256* out) const {
  if (!loaded_ || !infer_hash_) {
    return false;
  }
  uint64_t words[4] = {0, 0, 0, 0};
  bool ok = infer_hash_(vector, words);
  if (!ok) {
    return false;
  }
  if (out) {
    out->words = {words[0], words[1], words[2], words[3]};
  }
  return true;
}

uint64_t CodegenHasher::hash64(const float* vector, bool* ok) const {
  if (ok) *ok = false;
  if (!loaded_ || !infer_hash64_) {
    return 0;
  }
  int flag = 0;
  uint64_t value = infer_hash64_(vector, &flag);
  if (ok) *ok = (flag != 0);
  return value;
}

bool CodegenHasher::nearest_active_centroid(const float* vector,
                                            uint32_t* centroid,
                                            float* dist2,
                                            uint64_t* cache_token) const {
  if (cache_token) {
    *cache_token = 0;
  }
  if (!loaded_ || !infer_nearest_active_centroid_ || !vector || !centroid || !dist2) {
    return false;
  }
  return infer_nearest_active_centroid_(vector, centroid, dist2, cache_token);
}

bool CodegenHasher::hash_with_centroid(const float* vector, CodegenHashCentroidResult* out) const {
  if (!loaded_ || !vector || !out) {
    return false;
  }
  *out = CodegenHashCentroidResult{};
  if (infer_hash_with_centroid_) {
    uint64_t words[4] = {0, 0, 0, 0};
    if (!infer_hash_with_centroid_(
            vector,
            words,
            &out->centroid,
            &out->dist2,
            &out->cache_token)) {
      return false;
    }
    out->hash.words = {words[0], words[1], words[2], words[3]};
    return true;
  }

  if (!hash(vector, &out->hash)) {
    return false;
  }
  if (!nearest_active_centroid(vector, &out->centroid, &out->dist2, &out->cache_token)) {
    out->cache_token = 0;
  }
  return true;
}

bool CodegenHasher::ann_query(const float* vector,
                              uint32_t k,
                              std::vector<CodegenAnnNeighbor>* out,
                              uint64_t cache_token) const {
  if (out) {
    out->clear();
  }
  if (!loaded_ || (!infer_ann_query_ && !infer_ann_query_cached_) || !vector || !out || k == 0) {
    return false;
  }
  auto& scratch = ann_query_scratch();
  scratch.centroids.resize(k);
  scratch.dist2.resize(k);
  uint32_t count = 0;
  if (cache_token != 0 && infer_ann_query_cached_) {
    if (!infer_ann_query_cached_(
            vector,
            k,
            scratch.centroids.data(),
            scratch.dist2.data(),
            k,
            &count,
            cache_token)) {
      return false;
    }
  } else if (infer_ann_query_) {
    if (!infer_ann_query_(
            vector,
            k,
            scratch.centroids.data(),
            scratch.dist2.data(),
            k,
            &count)) {
      return false;
    }
  } else {
    return false;
  }
  if (count > k) {
    count = k;
  }
  out->reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    out->push_back(CodegenAnnNeighbor{scratch.centroids[i], scratch.dist2[i]});
  }
  return !out->empty();
}

} // namespace vortex
