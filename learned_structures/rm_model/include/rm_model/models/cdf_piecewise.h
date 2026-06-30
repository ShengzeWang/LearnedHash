#ifndef RM_MODEL_MODELS_CDF_PIECEWISE_H
#define RM_MODEL_MODELS_CDF_PIECEWISE_H

#include "rm_model/models/model.h"
#include "rm_model/training_data.h"

#include <cstdint>
#include <vector>

namespace rm_model {

class CdfPiecewiseModel : public Model {
 public:
  explicit CdfPiecewiseModel(const TrainingData<uint64_t>& data);
  explicit CdfPiecewiseModel(const TrainingData<uint32_t>& data);
  explicit CdfPiecewiseModel(const TrainingData<double>& data);
  explicit CdfPiecewiseModel(std::vector<double> knots);

  static CdfPiecewiseModel fit_from_distances(std::vector<double> dists, uint32_t max_knots);
  static std::vector<double> build_knots(std::vector<double> dists, uint32_t max_knots);
  static std::vector<double> build_knots_from_sorted(const std::vector<double>& dists,
                                                     uint32_t max_knots);

  double predict_to_float(const ModelInput& inp) const override;
  ModelDataType input_type() const override { return ModelDataType::Float; }
  ModelDataType output_type() const override { return ModelDataType::Float; }
  std::vector<ModelParam> params() const override;
  std::string code() const override;
  std::string function_name() const override { return "cdf_piecewise"; }
  bool needs_bounds_check() const override { return false; }

  const std::vector<double>& knots() const { return knots_; }
  uint64_t knot_count() const { return static_cast<uint64_t>(knots_.size()); }

 private:
  std::vector<double> knots_;
};

double cdf_piecewise_eval(const double* knots, uint64_t count, double x);

} // namespace rm_model

#endif // RM_MODEL_MODELS_CDF_PIECEWISE_H
