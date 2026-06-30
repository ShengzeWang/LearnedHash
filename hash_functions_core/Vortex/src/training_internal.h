#ifndef VORTEX_V1_TRAINING_INTERNAL_H
#define VORTEX_V1_TRAINING_INTERNAL_H

#include "vortex_v1/nsw_csr.h"
#include "vortex_v1/training.h"

#include "vector_io/vector_io.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace vortex::training_internal {

struct VortexTrainingProfile {
  bool base_cache_hit = false;
  bool order_cache_hit = false;
  bool cdf_cache_hit = false;
  double base_cache_wait_ms = 0.0;
  double order_cache_wait_ms = 0.0;
  double cdf_cache_wait_ms = 0.0;
  double base_build_ms = 0.0;
  double order_build_ms = 0.0;
  double cdf_cache_build_ms = 0.0;
  double gather_skeleton_ms = 0.0;
  double cluster_assign_ms = 0.0;
  double assign_centroids_ms = 0.0;
  double centroid_order_ms = 0.0;
  double range_alloc_ms = 0.0;
  double cdf_fit_ms = 0.0;
  double model_assembly_ms = 0.0;
};

struct VortexTrainingBase {
  uint32_t dim = 0;
  uint32_t K = 0;
  uint64_t seed = 0;
  uint64_t skeleton_node_count = 0;
  std::vector<float> centroids;
  std::vector<uint64_t> mass;
  std::vector<std::vector<float>> distances;
  std::vector<uint64_t> centroid_graph_offsets;
  std::vector<uint32_t> centroid_graph_neighbors;
  std::vector<float> centroid_graph_weights;

  uint64_t memory_bytes() const;
};

struct VortexCdfFit {
  std::vector<double> min_dist;
  std::vector<double> max_dist;
  std::vector<uint64_t> cdf_rows;
  std::vector<double> top_params;
  std::vector<double> leaf_params;
  CdfModelType top_type = CdfModelType::Linear;
  CdfModelType leaf_type = CdfModelType::Linear;
  uint32_t top_param_count = 0;
  uint32_t leaf_param_count = 0;
  uint32_t leaf_count = 0;

  uint64_t memory_bytes() const;
};

std::shared_ptr<const VortexTrainingBase> build_vortex_training_base(
    const TrainOptions& options,
    const vector_io::VectorStorage<float>& dataset,
    const NswCsr& csr,
    VortexTrainingProfile* profile = nullptr);

std::vector<uint32_t> build_vortex_centroid_order(const TrainOptions& options,
                                                  const VortexTrainingBase& base,
                                                  VortexTrainingProfile* profile = nullptr);

std::shared_ptr<const VortexCdfFit> fit_vortex_cdf_models(
    const TrainOptions& options,
    const VortexTrainingBase& base,
    VortexTrainingProfile* profile = nullptr);

VortexModel train_vortex_from_training_base(const TrainOptions& options,
                                            const VortexTrainingBase& base,
                                            const std::vector<uint32_t>* precomputed_order = nullptr,
                                            const VortexCdfFit* precomputed_cdf = nullptr,
                                            VortexTrainingProfile* profile = nullptr);

VortexModel train_vortex_from_loaded(const TrainOptions& options,
                                     const vector_io::VectorStorage<float>& dataset,
                                     const NswCsr& csr,
                                     VortexTrainingProfile* profile = nullptr);

} // namespace vortex::training_internal

#endif // VORTEX_V1_TRAINING_INTERNAL_H
