#ifndef RM_MODEL_MODEL_SPEC_INTERNAL_H
#define RM_MODEL_MODEL_SPEC_INTERNAL_H

#include "rm_model/models/balanced_radix.h"
#include "rm_model/models/cdf_piecewise.h"
#include "rm_model/models/cubic_spline.h"
#include "rm_model/models/histogram.h"
#include "rm_model/models/linear.h"
#include "rm_model/models/linear_spline.h"
#include "rm_model/models/normal.h"
#include "rm_model/models/radix.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rm_model::detail {

inline std::string trim_model_spec_layer(const std::string& value) {
  const auto start = value.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return "";
  const auto end = value.find_last_not_of(" \t\n\r");
  return value.substr(start, end - start + 1);
}

inline std::vector<std::string> parse_model_spec_layers(const std::string& model_spec) {
  std::vector<std::string> layers;
  std::stringstream ss(model_spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    std::string trimmed = trim_model_spec_layer(item);
    if (trimmed.empty()) {
      throw std::runtime_error("Model spec contains an empty layer");
    }
    layers.push_back(trimmed);
  }
  if (layers.empty()) {
    throw std::runtime_error("Model spec must not be empty");
  }
  return layers;
}

template <typename T>
std::unique_ptr<Model> make_model(const std::string& model_type, const TrainingData<T>& data) {
  if (model_type == "linear") return std::make_unique<LinearModel>(data);
  if (model_type == "robust_linear") return std::make_unique<RobustLinearModel>(data);
  if (model_type == "linear_spline") return std::make_unique<LinearSplineModel>(data);
  if (model_type == "cubic") return std::make_unique<CubicSplineModel>(data);
  if (model_type == "loglinear") return std::make_unique<LogLinearModel>(data);
  if (model_type == "normal") return std::make_unique<NormalModel>(data);
  if (model_type == "lognormal") return std::make_unique<LogNormalModel>(data);
  if (model_type == "radix") return std::make_unique<RadixModel>(data);
  if (model_type == "radix8") return std::make_unique<RadixTable>(data, 8);
  if (model_type == "radix18") return std::make_unique<RadixTable>(data, 18);
  if (model_type == "radix22") return std::make_unique<RadixTable>(data, 22);
  if (model_type == "radix26") return std::make_unique<RadixTable>(data, 26);
  if (model_type == "radix28") return std::make_unique<RadixTable>(data, 28);
  if (model_type == "bradix") return std::make_unique<BalancedRadixModel>(data);
  if (model_type == "histogram") return std::make_unique<EquidepthHistogramModel>(data);
  if (model_type == "cdf_piecewise") return std::make_unique<CdfPiecewiseModel>(data);
  throw std::runtime_error("Unknown model type: " + model_type);
}

template <typename T>
void validate_model_spec(const std::vector<std::string>& model_spec) {
  std::size_t num_layers = model_spec.size();
  auto empty = TrainingData<T>::empty();

  for (std::size_t idx = 0; idx < model_spec.size(); ++idx) {
    auto model = make_model(model_spec[idx], empty);
    switch (model->restriction()) {
      case ModelRestriction::None:
        break;
      case ModelRestriction::MustBeTop:
        if (idx != 0) {
          throw std::runtime_error("Model type must be the root model: " + model_spec[idx]);
        }
        break;
      case ModelRestriction::MustBeBottom:
        if (idx != num_layers - 1) {
          throw std::runtime_error("Model type must be the bottom model: " + model_spec[idx]);
        }
        break;
    }
  }
}

} // namespace rm_model::detail

#endif // RM_MODEL_MODEL_SPEC_INTERNAL_H
