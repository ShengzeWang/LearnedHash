#include "rm_model/models/cdf_piecewise.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rm_model {

namespace {

constexpr uint32_t kDefaultKnots = 32;

template <typename T>
std::vector<double> collect_distances(const TrainingData<T>& data) {
  std::vector<double> values;
  values.reserve(data.len());
  for (const auto& [x, _y] : data.iter_model_input()) {
    values.push_back(x.as_float());
  }
  return values;
}

double clamp_unit(double value) {
  if (value < 0.0) return 0.0;
  if (value >= 1.0) return std::nextafter(1.0, 0.0);
  return value;
}

} // namespace

double cdf_piecewise_eval(const double* knots, uint64_t count, double x) {
  if (!knots || count < 2) {
    return 0.0;
  }

  if (x <= knots[0]) {
    return 0.0;
  }
  double last = knots[count - 1];
  if (x >= last) {
    return std::nextafter(1.0, 0.0);
  }

  uint64_t lo = 0;
  uint64_t hi = count;
  while (lo < hi) {
    uint64_t mid = lo + (hi - lo) / 2;
    if (x >= knots[mid]) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  uint64_t idx = lo;
  if (idx == 0) {
    return 0.0;
  }
  uint64_t left = idx - 1;
  double d0 = knots[left];
  double d1 = knots[idx];
  double t = 0.0;
  if (d1 > d0) {
    t = (x - d0) / (d1 - d0);
  }
  double pos = (static_cast<double>(left) + t) / static_cast<double>(count - 1);
  return clamp_unit(pos);
}

CdfPiecewiseModel::CdfPiecewiseModel(const TrainingData<uint64_t>& data)
    : knots_(build_knots(collect_distances(data), kDefaultKnots)) {}

CdfPiecewiseModel::CdfPiecewiseModel(const TrainingData<uint32_t>& data)
    : knots_(build_knots(collect_distances(data), kDefaultKnots)) {}

CdfPiecewiseModel::CdfPiecewiseModel(const TrainingData<double>& data)
    : knots_(build_knots(collect_distances(data), kDefaultKnots)) {}

CdfPiecewiseModel::CdfPiecewiseModel(std::vector<double> knots)
    : knots_(std::move(knots)) {}

CdfPiecewiseModel CdfPiecewiseModel::fit_from_distances(std::vector<double> dists,
                                                        uint32_t max_knots) {
  return CdfPiecewiseModel(build_knots(std::move(dists), max_knots));
}

std::vector<double> CdfPiecewiseModel::build_knots(std::vector<double> dists,
                                                   uint32_t max_knots) {
  if (max_knots == 0) {
    throw std::runtime_error("cdf_piecewise max_knots must be > 0");
  }
  if (dists.empty()) {
    return {};
  }
  std::sort(dists.begin(), dists.end());
  return build_knots_from_sorted(dists, max_knots);
}

std::vector<double> CdfPiecewiseModel::build_knots_from_sorted(const std::vector<double>& dists,
                                                               uint32_t max_knots) {
  if (max_knots == 0) {
    throw std::runtime_error("cdf_piecewise max_knots must be > 0");
  }
  if (dists.empty()) {
    return {};
  }
  if (dists.size() == 1 || max_knots == 1) {
    return {dists.front()};
  }
  uint32_t k = std::min<uint32_t>(max_knots, static_cast<uint32_t>(dists.size()));
  std::vector<double> knots;
  knots.reserve(k);
  std::size_t n = dists.size();
  for (uint32_t i = 0; i < k; ++i) {
    double pos = static_cast<double>(i) * static_cast<double>(n - 1) / static_cast<double>(k - 1);
    std::size_t idx = static_cast<std::size_t>(std::round(pos));
    if (idx >= n) {
      idx = n - 1;
    }
    knots.push_back(dists[idx]);
  }
  for (std::size_t i = 1; i < knots.size(); ++i) {
    if (knots[i] < knots[i - 1]) {
      knots[i] = knots[i - 1];
    }
  }
  return knots;
}

double CdfPiecewiseModel::predict_to_float(const ModelInput& inp) const {
  return cdf_piecewise_eval(knots_.data(), static_cast<uint64_t>(knots_.size()), inp.as_float());
}

std::vector<ModelParam> CdfPiecewiseModel::params() const {
  return {ModelParam(static_cast<uint64_t>(knots_.size())), ModelParam(knots_)};
}

std::string CdfPiecewiseModel::code() const {
  return R"(
inline double cdf_piecewise(const uint64_t count, const double* knots, double x) {
    if (count < 2 || !knots) {
        return 0.0;
    }
    if (x <= knots[0]) {
        return 0.0;
    }
    double last = knots[count - 1];
    if (x >= last) {
        return std::nextafter(1.0, 0.0);
    }
    uint64_t lo = 0;
    uint64_t hi = count;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (x >= knots[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    uint64_t idx = lo;
    if (idx == 0) {
        return 0.0;
    }
    uint64_t left = idx - 1;
    double d0 = knots[left];
    double d1 = knots[idx];
    double t = 0.0;
    if (d1 > d0) {
        t = (x - d0) / (d1 - d0);
    }
    double pos = (static_cast<double>(left) + t) / static_cast<double>(count - 1);
    if (pos < 0.0) {
        pos = 0.0;
    } else if (pos >= 1.0) {
        pos = std::nextafter(1.0, 0.0);
    }
    return pos;
}
)";
}

} // namespace rm_model
