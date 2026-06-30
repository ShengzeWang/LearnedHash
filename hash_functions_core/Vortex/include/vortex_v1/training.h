#ifndef VORTEX_V1_TRAINING_H
#define VORTEX_V1_TRAINING_H

#include "vortex_v1/model.h"
#include "vortex_v1/nsw_csr.h"
#include "vortex_v1/types.h"
#include "vortex_v1/uint256.h"

#include "rm_model/centroid_router.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vortex {

struct BuildHnswOptions {
  std::filesystem::path dataset_path;
  uint32_t M = 32;
  uint32_t ef_construction = 200;
  Metric metric = Metric::L2;
};

struct ExtractNswOptions {
  std::optional<std::filesystem::path> index_path;
  std::optional<std::filesystem::path> dataset_path;
  uint32_t M = 32;
  uint32_t ef_construction = 200;
  Metric metric = Metric::L2;
  uint64_t target_skeleton = 0;
  std::optional<int> layer;
};

struct TrainOptions {
  std::filesystem::path dataset_path;
  std::filesystem::path nsw_path;
  uint32_t K = 0;
  uint32_t hash_bits = 64;
  std::string cdf_model_spec = "linear,linear";
  uint64_t cdf_branching_factor = 32;
  uint32_t centroid_knn = 32;
  bool enable_2opt = true;
  uint32_t two_opt_iterations = 8;
  uint64_t seed = 42;
  uint32_t threads = 1;
  bool use_full_dataset_for_assignments = true;
  uint64_t assignment_sample_limit = 0;
  bool enable_graph_centroid_order = true;
};

struct RangeAllocResult {
  std::vector<rm_model::CentroidRouter::Range256> range_start;
  std::vector<rm_model::CentroidRouter::Range256> range_size;
  std::vector<uint8_t> range_full;
};

std::filesystem::path build_hnsw_index(const BuildHnswOptions& options,
                                      const std::filesystem::path& output_path);

NswCsr extract_nsw(const ExtractNswOptions& options,
                   const std::filesystem::path& output_path);

VortexModel train_vortex(const TrainOptions& options);

RangeAllocResult allocate_ranges(const std::vector<uint64_t>& mass,
                                 const std::vector<uint32_t>& order,
                                 uint32_t hash_bits);

} // namespace vortex

#endif // VORTEX_V1_TRAINING_H
