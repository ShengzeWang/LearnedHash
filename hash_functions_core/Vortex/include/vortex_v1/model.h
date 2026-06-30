#ifndef VORTEX_V1_MODEL_H
#define VORTEX_V1_MODEL_H

#include "vortex_v1/types.h"
#include "vortex_v1/uint256.h"

#include "rm_model/centroid_router.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vortex {

struct HashOutput {
  UInt256 value;
  uint32_t bits = 0;

  uint64_t to_u64() const { return vortex::to_u64(value, bits); }
  std::string to_hex() const { return vortex::to_hex(value, bits); }
};

enum class CdfModelType : uint32_t {
  Linear = 1,
  LogLinear = 2,
  Cubic = 3,
  Normal = 4,
  LogNormal = 5,
};

const char* cdf_model_name(CdfModelType type);
CdfModelType parse_cdf_model(const std::string& name);
uint32_t cdf_param_count(CdfModelType type);

struct VortexModelHeader {
  char magic[8];
  uint32_t version = 2;
  uint32_t dim = 0;
  uint32_t metric = 0;
  uint32_t hash_bits = 0;
  uint32_t centroid_count = 0;
  uint32_t cdf_top_type = 0;
  uint32_t cdf_leaf_type = 0;
  uint32_t cdf_leaf_count = 0;
  uint32_t cdf_top_param_count = 0;
  uint32_t cdf_leaf_param_count = 0;
};

struct VortexModel {
  VortexModel() = default;
  VortexModel(const VortexModel& other);
  VortexModel& operator=(const VortexModel& other);
  VortexModel(VortexModel&& other) noexcept;
  VortexModel& operator=(VortexModel&& other) noexcept;
  ~VortexModel() = default;

  uint32_t dim = 0;
  Metric metric = Metric::L2;
  uint32_t hash_bits = 0;
  std::vector<float> centroids; // K * dim
  std::vector<uint32_t> order;  // size K
  std::vector<uint64_t> mass;   // size K
  std::vector<rm_model::CentroidRouter::Range256> range_start; // size K
  std::vector<rm_model::CentroidRouter::Range256> range_size;  // size K
  std::vector<uint8_t> range_full;  // size K, for bits=256 overflow case
  std::vector<double> min_dist;     // size K
  std::vector<double> max_dist;     // size K
  CdfModelType cdf_top_type = CdfModelType::Linear;
  CdfModelType cdf_leaf_type = CdfModelType::Linear;
  uint32_t cdf_leaf_count = 0;
  uint32_t cdf_top_param_count = 0;
  uint32_t cdf_leaf_param_count = 0;
  std::vector<uint64_t> cdf_rows;    // size K
  std::vector<double> cdf_top_params;   // size K * cdf_top_param_count
  std::vector<double> cdf_leaf_params;  // size K * cdf_leaf_count * cdf_leaf_param_count
  rm_model::CentroidRouter router;

  std::size_t centroid_count() const { return centroids.empty() ? 0 : centroids.size() / dim; }
  void compute_active();

  void write(const std::filesystem::path& path) const;
  static VortexModel read(const std::filesystem::path& path);

  HashOutput hash(const float* vector) const;
  uint64_t hash_u64(const float* vector, double* pred = nullptr) const;
  HashOutput hash_with_pred(const float* vector, double* pred) const;
  HashOutput hash_from_centroid(uint32_t centroid, double dist2, double* pred = nullptr) const;
  uint64_t hash_from_centroid_u64(uint32_t centroid, double dist2, double* pred = nullptr) const;
};

} // namespace vortex

#endif // VORTEX_V1_MODEL_H
