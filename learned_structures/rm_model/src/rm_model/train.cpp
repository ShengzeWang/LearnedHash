#include "rm_model/train.h"

#include "model_spec_internal.h"

#include "rm_model/cache_fix.h"
#include "rm_model/logging.h"
#include "rm_model/learned_model_selector.h"

#include <algorithm>
#include <chrono>

namespace rm_model {

template <typename T>
TrainedModel train_two_layer(TrainingData<T>& data,
                           const std::string& layer1_model,
                           const std::string& layer2_model,
                           uint64_t num_leaf_models);

template <typename T>
TrainedModel train_multi_layer(TrainingData<T>& data,
                             const std::vector<std::string>& model_list,
                             const std::string& last_model,
                             uint64_t branch_factor);

template <typename T>
TrainedModel train(TrainingData<T>& data, const std::string& model_spec, uint64_t branch_factor) {
  auto start = std::chrono::steady_clock::now();
  if (branch_factor <= 1) {
    throw std::runtime_error("Branching factor must be >= 2");
  }

  std::vector<std::string> models = detail::parse_model_spec_layers(model_spec);
  detail::validate_model_spec<T>(models);
  std::string last_model = models.back();
  models.pop_back();

  if (models.size() == 1) {
    auto result = train_two_layer<T>(data, models[0], last_model, branch_factor);
    auto end = std::chrono::steady_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    result.build_time_ns = static_cast<uint64_t>(nanos < 0 ? 0 : nanos);
    return result;
  }
  if (models.size() > 1) {
    auto result = train_multi_layer<T>(data, models, last_model, branch_factor);
    auto end = std::chrono::steady_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    result.build_time_ns = static_cast<uint64_t>(nanos < 0 ? 0 : nanos);
    return result;
  }

  throw std::runtime_error("Invalid model specification");
}

template <typename T>
TrainedModel train_for_size(TrainingData<T>& data, std::size_t max_size) {
  auto start = std::chrono::steady_clock::now();
  auto pareto = select_pareto_configs(data, 1000);

  auto it = std::find_if(pareto.begin(), pareto.end(), [&](const ModelSelectionStats& stats) {
    return stats.size < max_size;
  });
  if (it == pareto.end()) {
    throw std::runtime_error("Could not find any configurations smaller than " +
                             std::to_string(max_size));
  }

  RM_MODEL_LOG_INFO("Found model config " << it->model_spec << " " << it->branching_factor
                                          << " with size " << it->size << " and average log2 "
                                          << it->average_log2_error);

  auto result = train<T>(data, it->model_spec, it->branching_factor);
  auto end = std::chrono::steady_clock::now();
  auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  result.build_time_ns = static_cast<uint64_t>(nanos < 0 ? 0 : nanos);
  return result;
}

TrainedModel train_bounded(TrainingData<uint64_t>& data,
                         const std::string& model_spec,
                         uint64_t branch_factor,
                         std::size_t line_size) {
  auto start = std::chrono::steady_clock::now();
  if (line_size == 0) {
    throw std::runtime_error("Line size must be >= 1");
  }

  auto spline = cache_fix(data, line_size);

  std::vector<std::pair<uint64_t, std::size_t>> reindexed;
  reindexed.reserve(spline.size());
  for (std::size_t idx = 0; idx < spline.size(); ++idx) {
    reindexed.emplace_back(spline[idx].first, idx);
  }

  TrainingData<uint64_t> new_data(
      std::make_shared<TrainingData<uint64_t>::VectorProvider>(reindexed));

  auto result = train<uint64_t>(new_data, model_spec, branch_factor);
  result.cache_fix = std::make_pair(line_size, spline);
  result.num_data_rows = data.len();

  auto end = std::chrono::steady_clock::now();
  auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  result.build_time_ns = static_cast<uint64_t>(nanos < 0 ? 0 : nanos);
  return result;
}

template TrainedModel train<uint64_t>(TrainingData<uint64_t>&, const std::string&, uint64_t);
template TrainedModel train<uint32_t>(TrainingData<uint32_t>&, const std::string&, uint64_t);
template TrainedModel train<double>(TrainingData<double>&, const std::string&, uint64_t);

template TrainedModel train_for_size<uint64_t>(TrainingData<uint64_t>&, std::size_t);
template TrainedModel train_for_size<uint32_t>(TrainingData<uint32_t>&, std::size_t);
template TrainedModel train_for_size<double>(TrainingData<double>&, std::size_t);

} // namespace rm_model
