#include "vortex_v1/training.h"

#include "training_common_internal.h"
#include "training_internal.h"

#include "rm_model/parallel.h"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vortex {
namespace training_internal {

VortexModel train_vortex_from_training_base(const TrainOptions& options,
                                            const VortexTrainingBase& base,
                                            const std::vector<uint32_t>* precomputed_order,
                                            const VortexCdfFit* precomputed_cdf,
                                            VortexTrainingProfile* profile) {
  set_threads(options.threads);
  rm_model::set_thread_count(options.threads);

  validate_train_options(options);
  validate_training_base(options, base);

  std::vector<uint32_t> order = precomputed_order != nullptr
      ? *precomputed_order
      : build_vortex_centroid_order(options, base, profile);
  if (order.size() != options.K) {
    throw std::runtime_error("Centroid ordering failed");
  }

  auto range_start = TrainingClock::now();
  auto ranges = allocate_ranges(base.mass, order, options.hash_bits);
  if (profile != nullptr) {
    profile->range_alloc_ms += elapsed_ms(range_start);
  }

  std::shared_ptr<const VortexCdfFit> owned_cdf;
  const VortexCdfFit* cdf_fit = precomputed_cdf;
  if (cdf_fit == nullptr) {
    owned_cdf = fit_vortex_cdf_models(options, base, profile);
    cdf_fit = owned_cdf.get();
  }

  auto assembly_start = TrainingClock::now();
  VortexModel model;
  model.dim = base.dim;
  model.metric = Metric::L2;
  model.hash_bits = options.hash_bits;
  model.centroids = base.centroids;
  model.order = std::move(order);
  model.mass = base.mass;
  model.range_start = std::move(ranges.range_start);
  model.range_size = std::move(ranges.range_size);
  model.range_full = std::move(ranges.range_full);
  model.min_dist = cdf_fit->min_dist;
  model.max_dist = cdf_fit->max_dist;
  model.cdf_top_type = cdf_fit->top_type;
  model.cdf_leaf_type = cdf_fit->leaf_type;
  model.cdf_leaf_count = cdf_fit->leaf_count;
  model.cdf_top_param_count = cdf_fit->top_param_count;
  model.cdf_leaf_param_count = cdf_fit->leaf_param_count;
  model.cdf_rows = cdf_fit->cdf_rows;
  model.cdf_top_params = cdf_fit->top_params;
  model.cdf_leaf_params = cdf_fit->leaf_params;
  model.compute_active();
  if (profile != nullptr) {
    profile->model_assembly_ms += elapsed_ms(assembly_start);
  }

  return model;
}

VortexModel train_vortex_from_loaded(const TrainOptions& options,
                                     const vector_io::VectorStorage<float>& dataset,
                                     const NswCsr& csr,
                                     VortexTrainingProfile* profile) {
  auto base = build_vortex_training_base(options, dataset, csr, profile);
  return train_vortex_from_training_base(options, *base, nullptr, nullptr, profile);
}

} // namespace training_internal

VortexModel train_vortex(const TrainOptions& options) {
  training_internal::set_threads(options.threads);
  rm_model::set_thread_count(options.threads);

  training_internal::validate_train_options(options);

  auto dataset = training_internal::load_vectors_or_throw(options.dataset_path);
  NswCsr csr = NswCsr::read(options.nsw_path);
  return training_internal::train_vortex_from_loaded(options, dataset, csr);
}

} // namespace vortex
