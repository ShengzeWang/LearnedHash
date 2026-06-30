#ifndef VORTEX_V1_CODEGEN_LOADER_H
#define VORTEX_V1_CODEGEN_LOADER_H

#include "vortex_v1/model.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vortex {

struct CodegenManifest {
  std::string name;
  std::string codegen_name;
  std::filesystem::path root_dir;
  std::filesystem::path model_path;
  std::filesystem::path centroid_index_path;
  std::filesystem::path codegen_dir;
  std::filesystem::path include_path;
  std::filesystem::path src_path;
};

CodegenManifest load_codegen_manifest(const std::filesystem::path& manifest_path);

struct CodegenLoadOptions {
  bool enable_centroid_index = true;
  uint32_t centroid_index_pivots = 16;
  bool enable_query_cache = true;
};

struct CodegenAnnNeighbor {
  uint32_t centroid = 0;
  float dist2 = 0.0f;
};

struct CodegenHashCentroidResult {
  UInt256 hash{};
  uint32_t centroid = 0;
  float dist2 = 0.0f;
  uint64_t cache_token = 0;
};

class CodegenHasher {
 public:
  CodegenHasher() = default;
  ~CodegenHasher();

  bool load(const std::filesystem::path& codegen_dir,
            const CodegenLoadOptions& options = CodegenLoadOptions{});
  void unload();

  uint32_t dim() const { return dim_; }
  uint32_t hash_bits() const { return hash_bits_; }
  const std::string& library_status() const { return library_status_; }
  const std::filesystem::path& library_path() const { return library_path_; }
  bool hash(const float* vector, UInt256* out) const;
  uint64_t hash64(const float* vector, bool* ok) const;
  bool nearest_active_centroid(const float* vector,
                               uint32_t* centroid,
                               float* dist2,
                               uint64_t* cache_token = nullptr) const;
  bool hash_with_centroid(const float* vector, CodegenHashCentroidResult* out) const;
  bool ann_query(const float* vector,
                 uint32_t k,
                 std::vector<CodegenAnnNeighbor>* out,
                 uint64_t cache_token = 0) const;

 private:
  using InferLoadFn = bool (*)(const char*);
  using InferLoadExFn = bool (*)(const char*, int, uint32_t);
  using InferLoadEx2Fn = bool (*)(const char*, int, uint32_t, int);
  using InferCleanupFn = void (*)();
  using InferDimFn = uint32_t (*)();
  using InferHashBitsFn = uint32_t (*)();
  using InferHashFn = bool (*)(const float*, uint64_t out_words[4]);
  using InferHash64Fn = uint64_t (*)(const float*, int* ok);
  using InferNearestActiveCentroidFn = bool (*)(const float*, uint32_t*, float*, uint64_t*);
  using InferHashWithCentroidFn = bool (*)(const float*, uint64_t[4], uint32_t*, float*, uint64_t*);
  using InferAnnQueryFn =
      bool (*)(const float*, uint32_t, uint32_t*, float*, uint32_t, uint32_t*);
  using InferAnnQueryCachedFn =
      bool (*)(const float*, uint32_t, uint32_t*, float*, uint32_t, uint32_t*, uint64_t);

  void* handle_ = nullptr;
  InferCleanupFn infer_cleanup_ = nullptr;
  InferDimFn infer_dim_ = nullptr;
  InferHashBitsFn infer_hash_bits_ = nullptr;
  InferHashFn infer_hash_ = nullptr;
  InferHash64Fn infer_hash64_ = nullptr;
  InferNearestActiveCentroidFn infer_nearest_active_centroid_ = nullptr;
  InferHashWithCentroidFn infer_hash_with_centroid_ = nullptr;
  InferAnnQueryFn infer_ann_query_ = nullptr;
  InferAnnQueryCachedFn infer_ann_query_cached_ = nullptr;
  uint32_t dim_ = 0;
  uint32_t hash_bits_ = 0;
  std::string library_status_;
  std::filesystem::path library_path_;
  bool loaded_ = false;
};

} // namespace vortex

#endif // VORTEX_V1_CODEGEN_LOADER_H
