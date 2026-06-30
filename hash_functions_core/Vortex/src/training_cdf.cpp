#include "training_common_internal.h"
#include "training_internal.h"

#include "model_spec_utils.h"
#include "vortex_v1/model.h"

#include "rm_model/train.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace vortex {
namespace {

using detail::split_model_spec;

struct CdfTrainData {
  rm_model::TrainingData<double> data;
  std::size_t rows = 0;
};

std::vector<double> extract_params(const rm_model::Model& model,
                                   CdfModelType expected,
                                   uint32_t expected_params) {
  CdfModelType actual = parse_cdf_model(model.function_name());
  if (actual != expected) {
    throw std::runtime_error("CDF model type mismatch (expected " +
                             std::string(cdf_model_name(expected)) + ", got " +
                             model.function_name() + ")");
  }
  auto params = model.params();
  if (params.size() != expected_params) {
    throw std::runtime_error("CDF model param count mismatch");
  }
  std::vector<double> out;
  out.reserve(params.size());
  for (const auto& param : params) {
    if (param.is_array()) {
      throw std::runtime_error("CDF model parameters must be scalar");
    }
    out.push_back(param.as_float());
  }
  return out;
}

CdfTrainData build_cdf_train_data(const std::vector<float>& sorted_distances) {
  std::vector<std::pair<double, std::size_t>> pairs;
  pairs.reserve(sorted_distances.size());
  double last_x = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < sorted_distances.size(); ++i) {
    double x = static_cast<double>(sorted_distances[i]);
    if (!std::isfinite(x)) {
      x = std::isfinite(last_x) ? last_x : 0.0;
    }
    // rm_model training assumes monotonic keys and is more stable when keys are strictly increasing.
    if (!(x > last_x)) {
      x = std::nextafter(last_x, std::numeric_limits<double>::infinity());
    }
    pairs.emplace_back(x, i);
    last_x = x;
  }
  CdfTrainData out{
      rm_model::TrainingData<double>(
          std::make_shared<rm_model::TrainingData<double>::VectorProvider>(std::move(pairs))),
      sorted_distances.size()};
  return out;
}

uint32_t compute_leaf_count(const std::vector<std::vector<float>>& per_centroid,
                            uint64_t requested) {
  if (requested < 2) {
    throw std::runtime_error("CDF branching factor must be >= 2");
  }
  uint64_t min_count = std::numeric_limits<uint64_t>::max();
  for (const auto& dists : per_centroid) {
    if (dists.size() >= 2) {
      min_count = std::min<uint64_t>(min_count, dists.size());
    }
  }
  if (min_count == std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error("Not enough centroid assignments to train CDF models");
  }
  return static_cast<uint32_t>(std::min<uint64_t>(requested, min_count));
}

training_internal::VortexCdfFit fit_cdf_models(std::vector<std::vector<float>>& per_centroid,
                                               const std::string& model_spec,
                                               uint64_t branching_factor) {
  auto cdf_layers = split_model_spec(model_spec);
  if (cdf_layers.size() != 2) {
    throw std::runtime_error("cdf_model_spec must contain exactly 2 layers");
  }
  CdfModelType top_type = parse_cdf_model(cdf_layers[0]);
  CdfModelType leaf_type = parse_cdf_model(cdf_layers[1]);
  uint32_t top_param_count = cdf_param_count(top_type);
  uint32_t leaf_param_count = cdf_param_count(leaf_type);
  uint32_t leaf_count = compute_leaf_count(per_centroid, branching_factor);

  std::size_t k = per_centroid.size();
  training_internal::VortexCdfFit result;
  result.min_dist.assign(k, 0.0);
  result.max_dist.assign(k, 0.0);
  result.cdf_rows.assign(k, 0);
  result.top_params.assign(k * top_param_count, 0.0);
  result.leaf_params.assign(k * leaf_count * leaf_param_count, 0.0);
  result.top_type = top_type;
  result.leaf_type = leaf_type;
  result.top_param_count = top_param_count;
  result.leaf_param_count = leaf_param_count;
  result.leaf_count = leaf_count;

  for (std::size_t cid = 0; cid < k; ++cid) {
    auto& dists = per_centroid[cid];
    if (dists.empty()) {
      continue;
    }
    std::sort(dists.begin(), dists.end());
    result.min_dist[cid] = static_cast<double>(dists.front());
    result.max_dist[cid] = static_cast<double>(dists.back());
    result.cdf_rows[cid] = static_cast<uint64_t>(dists.size());

    if (dists.size() < 2) {
      continue;
    }

    CdfTrainData train_data = build_cdf_train_data(dists);
    rm_model::TrainedModel trained;
    try {
      trained = rm_model::train<double>(train_data.data, model_spec, leaf_count);
    } catch (const std::exception& ex) {
      throw std::runtime_error("CDF training failed for centroid " + std::to_string(cid) +
                               " with " + std::to_string(train_data.rows) + " samples: " +
                               ex.what());
    }
    if (trained.model_layers.size() != 2) {
      throw std::runtime_error("CDF training produced unexpected layer count");
    }
    if (trained.branching_factor != leaf_count) {
      throw std::runtime_error("CDF training branching factor mismatch");
    }

    auto top_params = extract_params(*trained.model_layers[0][0], top_type, top_param_count);
    std::size_t top_offset = cid * top_param_count;
    std::copy(top_params.begin(), top_params.end(), result.top_params.begin() + top_offset);

    const auto& leaf_layer = trained.model_layers[1];
    if (leaf_layer.size() != leaf_count) {
      throw std::runtime_error("CDF leaf layer size mismatch");
    }
    for (std::size_t leaf_idx = 0; leaf_idx < leaf_layer.size(); ++leaf_idx) {
      auto leaf_params = extract_params(*leaf_layer[leaf_idx], leaf_type, leaf_param_count);
      std::size_t leaf_offset = (cid * leaf_count + leaf_idx) * leaf_param_count;
      std::copy(leaf_params.begin(), leaf_params.end(),
                result.leaf_params.begin() + leaf_offset);
    }
  }

  return result;
}

} // namespace

namespace training_internal {

uint64_t VortexCdfFit::memory_bytes() const {
  uint64_t total = byte_count(min_dist.size(), sizeof(double));
  total = saturating_add(total, byte_count(max_dist.size(), sizeof(double)));
  total = saturating_add(total, byte_count(cdf_rows.size(), sizeof(uint64_t)));
  total = saturating_add(total, byte_count(top_params.size(), sizeof(double)));
  total = saturating_add(total, byte_count(leaf_params.size(), sizeof(double)));
  return total;
}

std::shared_ptr<const VortexCdfFit> fit_vortex_cdf_models(
    const TrainOptions& options,
    const VortexTrainingBase& base,
    VortexTrainingProfile* profile) {
  validate_training_base(options, base);

  auto cdf_start = TrainingClock::now();
  auto per_centroid_distances = base.distances;
  auto cdf_fit = std::make_shared<VortexCdfFit>(
      fit_cdf_models(per_centroid_distances,
                     options.cdf_model_spec,
                     options.cdf_branching_factor));
  if (profile != nullptr) {
    profile->cdf_fit_ms += elapsed_ms(cdf_start);
  }
  return cdf_fit;
}

} // namespace training_internal
} // namespace vortex
