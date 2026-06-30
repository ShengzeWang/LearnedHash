#include "selector_eval_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace vortex::selector_internal {

inline double eval_hash64_exp1(double inp) {
  double x = inp;
  x = 1.0 + x / 64.0;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  x *= x;
  return x;
}

inline double eval_hash64_phi(double x) {
  return 1.0 / (1.0 + eval_hash64_exp1(-1.65451 * x));
}

inline double eval_hash64_clamp_unit(double value) {
  if (value < 0.0) return 0.0;
  if (value >= 1.0) return std::nextafter(1.0, 0.0);
  return value;
}

inline std::size_t eval_hash64_clamp_index(double pred, std::size_t bound) {
  if (bound == 0) return 0;
  if (!std::isfinite(pred)) return 0;
  if (pred < 0.0) return 0;
  double max_val = static_cast<double>(bound - 1);
  if (pred > max_val) {
    return bound - 1;
  }
  return static_cast<std::size_t>(pred);
}

inline double eval_hash64_linear_model(const double* params, double x) {
  return std::fma(params[1], x, params[0]);
}

inline double eval_hash64_cubic_model(const double* params, double x) {
  double v1 = std::fma(params[0], x, params[1]);
  double v2 = std::fma(v1, x, params[2]);
  return std::fma(v2, x, params[3]);
}

double eval_hash64_model(CdfModelType type, const double* params, double x) {
  switch (type) {
    case CdfModelType::Linear:
      return eval_hash64_linear_model(params, x);
    case CdfModelType::LogLinear:
      return eval_hash64_exp1(std::fma(params[1], x, params[0]));
    case CdfModelType::Cubic: {
      return eval_hash64_cubic_model(params, x);
    }
    case CdfModelType::Normal:
      if (params[1] <= 0.0) return 0.0;
      return eval_hash64_phi((x - params[0]) / params[1]) * params[2];
    case CdfModelType::LogNormal: {
      if (params[1] <= 0.0 || x <= 0.0) return 0.0;
      double lx = std::max(std::log(x), 0.0);
      return eval_hash64_phi((lx - params[0]) / params[1]) * params[2];
    }
  }
  return 0.0;
}

template <typename TopModel, typename LeafModel>
double eval_hash64_cdf_fraction_typed(const EvalHash64Plan& plan,
                                      uint32_t centroid,
                                      double dist2,
                                      TopModel top_model,
                                      LeafModel leaf_model) {
  const auto& centroid_plan = plan.centroids[centroid];
  uint64_t rows = centroid_plan.cdf_rows;
  if (rows < 2 || plan.cdf_leaf_count == 0) {
    return 0.0;
  }
  double bound = static_cast<double>(rows - 1);

  double top_pred = top_model(centroid_plan.top_params, dist2);
  if (!std::isfinite(top_pred)) {
    top_pred = 0.0;
  }
  std::size_t leaf_idx = eval_hash64_clamp_index(top_pred, plan.cdf_leaf_count);
  const double* leaf_params =
      centroid_plan.leaf_params + leaf_idx * plan.cdf_leaf_param_count;
  double pred = leaf_model(leaf_params, dist2);
  if (!std::isfinite(pred)) {
    pred = 0.0;
  }

  if (pred < 0.0) pred = 0.0;
  if (pred > bound) pred = bound;
  return eval_hash64_clamp_unit(pred / bound);
}

double eval_hash64_cdf_fraction(const EvalHash64Plan& plan,
                                uint32_t centroid,
                                double dist2) {
  if (!plan.has_cdf) {
    return 0.0;
  }
  switch (plan.cdf_kernel) {
    case EvalHash64CdfKernel::linear_linear:
      return eval_hash64_cdf_fraction_typed(
          plan,
          centroid,
          dist2,
          eval_hash64_linear_model,
          eval_hash64_linear_model);
    case EvalHash64CdfKernel::cubic_linear:
      return eval_hash64_cdf_fraction_typed(
          plan,
          centroid,
          dist2,
          eval_hash64_cubic_model,
          eval_hash64_linear_model);
    case EvalHash64CdfKernel::linear_cubic:
      return eval_hash64_cdf_fraction_typed(
          plan,
          centroid,
          dist2,
          eval_hash64_linear_model,
          eval_hash64_cubic_model);
    case EvalHash64CdfKernel::cubic_cubic:
      return eval_hash64_cdf_fraction_typed(
          plan,
          centroid,
          dist2,
          eval_hash64_cubic_model,
          eval_hash64_cubic_model);
    case EvalHash64CdfKernel::generic:
      break;
  }
  return eval_hash64_cdf_fraction_typed(
      plan,
      centroid,
      dist2,
      [&](const double* params, double x) {
        return eval_hash64_model(plan.cdf_top_type, params, x);
      },
      [&](const double* params, double x) {
        return eval_hash64_model(plan.cdf_leaf_type, params, x);
      });
}

double eval_hash64_fraction_from_centroid_unchecked(
    const EvalHash64Plan& plan,
    const EvalHash64CentroidPlan& centroid_plan,
    uint32_t centroid,
    double dist2);

double eval_hash64_fraction_from_centroid(const EvalHash64Plan& plan,
                                          uint32_t centroid,
                                          double dist2) {
  if (centroid >= plan.centroids.size()) {
    throw std::runtime_error("Selector 64-bit hash centroid index out of bounds");
  }
  if (!std::isfinite(dist2) || dist2 < 0.0) {
    throw std::runtime_error("Selector 64-bit hash received invalid centroid distance");
  }

  const auto& centroid_plan = plan.centroids[centroid];
  return eval_hash64_fraction_from_centroid_unchecked(plan,
                                                      centroid_plan,
                                                      centroid,
                                                      dist2);
}

double eval_hash64_fraction_from_centroid_unchecked(
    const EvalHash64Plan& plan,
    const EvalHash64CentroidPlan& centroid_plan,
    uint32_t centroid,
    double dist2) {
  double value = eval_hash64_cdf_fraction(plan, centroid, dist2);
  if (value <= 0.0) {
    double min_v = centroid_plan.min_dist;
    double max_v = centroid_plan.max_dist;
    if (max_v > min_v && std::isfinite(min_v) && std::isfinite(max_v)) {
      double calib = (dist2 - min_v) / (max_v - min_v);
      if (calib > value) {
        value = calib;
      }
    }
  }
  if (value < 0.0) {
    value = 0.0;
  }
  if (value >= 1.0) {
    value = std::nextafter(1.0, 0.0);
  }
  return value;
}

uint64_t eval_hash64_to_q64(double value) {
  if (!(value > 0.0)) {
    return 0;
  }
  if (value >= 1.0) {
    return std::numeric_limits<uint64_t>::max();
  }
  long double scaled = std::ldexp(static_cast<long double>(value), 64);
  long double max_val = static_cast<long double>(std::numeric_limits<uint64_t>::max());
  if (scaled >= max_val) {
    return std::numeric_limits<uint64_t>::max();
  }
  uint64_t q = static_cast<uint64_t>(scaled);
  if (q == 0) {
    return 1;
  }
  return q;
}

uint64_t eval_hash64_mul_q64_low(const EvalHash64CentroidPlan& centroid_plan,
                                 uint64_t q) {
  if (q == 0) {
    return 0;
  }
#if defined(_MSC_VER)
  uint64_t high0 = 0;
  (void)_umul128(centroid_plan.range_size_low, q, &high0);
  uint64_t high1 = 0;
  uint64_t low1 = _umul128(centroid_plan.range_size_high, q, &high1);
  (void)high1;
  return high0 + low1;
#else
  unsigned __int128 prod0 =
      static_cast<unsigned __int128>(centroid_plan.range_size_low) * q;
  uint64_t scaled = static_cast<uint64_t>(prod0 >> 64);
  if (centroid_plan.range_size_high != 0) {
    unsigned __int128 prod1 =
        static_cast<unsigned __int128>(centroid_plan.range_size_high) * q;
    scaled += static_cast<uint64_t>(prod1);
  }
  return scaled;
#endif
}

uint64_t eval_hash64_mask_bits(uint64_t value, uint32_t bits) {
  if (bits == 0) {
    return 0;
  }
  if (bits >= 64) {
    return value;
  }
  return value & ((1ULL << bits) - 1ULL);
}

template <EvalHash64CdfKernel Kernel>
double eval_hash64_cdf_fraction_kernel(const EvalHash64Plan& plan,
                                       uint32_t centroid,
                                       double dist2) {
  if constexpr (Kernel == EvalHash64CdfKernel::linear_linear) {
    return eval_hash64_cdf_fraction_typed(
        plan,
        centroid,
        dist2,
        eval_hash64_linear_model,
        eval_hash64_linear_model);
  } else if constexpr (Kernel == EvalHash64CdfKernel::cubic_linear) {
    return eval_hash64_cdf_fraction_typed(
        plan,
        centroid,
        dist2,
        eval_hash64_cubic_model,
        eval_hash64_linear_model);
  } else if constexpr (Kernel == EvalHash64CdfKernel::linear_cubic) {
    return eval_hash64_cdf_fraction_typed(
        plan,
        centroid,
        dist2,
        eval_hash64_linear_model,
        eval_hash64_cubic_model);
  } else if constexpr (Kernel == EvalHash64CdfKernel::cubic_cubic) {
    return eval_hash64_cdf_fraction_typed(
        plan,
        centroid,
        dist2,
        eval_hash64_cubic_model,
        eval_hash64_cubic_model);
  } else {
    return eval_hash64_cdf_fraction(plan, centroid, dist2);
  }
}

template <EvalHash64CdfKernel Kernel>
uint64_t eval_hash64_from_centroid_unchecked_kernel(const EvalHash64Plan& plan,
                                                    uint32_t centroid,
                                                    double dist2) {
  const auto& centroid_plan = plan.centroids[centroid];
  double value = eval_hash64_cdf_fraction_kernel<Kernel>(plan, centroid, dist2);
  if (value <= 0.0) {
    double min_v = centroid_plan.min_dist;
    double max_v = centroid_plan.max_dist;
    if (max_v > min_v && std::isfinite(min_v) && std::isfinite(max_v)) {
      double calib = (dist2 - min_v) / (max_v - min_v);
      if (calib > value) {
        value = calib;
      }
    }
  }
  if (value < 0.0) {
    value = 0.0;
  }
  if (value >= 1.0) {
    value = std::nextafter(1.0, 0.0);
  }
  uint64_t q = eval_hash64_to_q64(value);
  uint64_t scaled = eval_hash64_mul_q64_low(centroid_plan, q);
  return eval_hash64_mask_bits(centroid_plan.range_start_low + scaled,
                               plan.hash_bits);
}

template <EvalHash64CdfKernel Kernel>
void fill_eval_hash64_from_nearest_kernel(const EvalHash64Plan& plan,
                                          const EvalNearestCache& nearest_cache,
                                          std::vector<HashEntry64>* out,
                                          uint32_t threads) {
  parallel_for_chunks(out->size(), threads, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      uint64_t hash = eval_hash64_from_centroid_unchecked_kernel<Kernel>(
          plan,
          nearest_cache.centroid[i],
          nearest_cache.dist2[i]);
      (*out)[i] = HashEntry64{hash, static_cast<uint32_t>(i)};
    }
  });
}

uint64_t eval_hash64_from_centroid(const EvalHash64Plan& plan,
                                   uint32_t centroid,
                                   double dist2) {
  double value = eval_hash64_fraction_from_centroid(plan, centroid, dist2);
  uint64_t q = eval_hash64_to_q64(value);
  const auto& centroid_plan = plan.centroids[centroid];
  uint64_t scaled = eval_hash64_mul_q64_low(centroid_plan, q);
  return eval_hash64_mask_bits(centroid_plan.range_start_low + scaled,
                               plan.hash_bits);
}

uint64_t eval_hash64_from_centroid_unchecked(const EvalHash64Plan& plan,
                                             uint32_t centroid,
                                             double dist2) {
  const auto& centroid_plan = plan.centroids[centroid];
  double value =
      eval_hash64_fraction_from_centroid_unchecked(plan,
                                                  centroid_plan,
                                                  centroid,
                                                  dist2);
  uint64_t q = eval_hash64_to_q64(value);
  uint64_t scaled = eval_hash64_mul_q64_low(centroid_plan, q);
  return eval_hash64_mask_bits(centroid_plan.range_start_low + scaled,
                               plan.hash_bits);
}

void fill_eval_hash64_from_nearest(const EvalHash64Plan& plan,
                                   const EvalNearestCache& nearest_cache,
                                   std::vector<HashEntry64>* out,
                                   uint32_t threads) {
  if (out == nullptr) {
    throw std::runtime_error("Selector 64-bit hash fill received null output");
  }
  switch (plan.cdf_kernel) {
    case EvalHash64CdfKernel::linear_linear:
      fill_eval_hash64_from_nearest_kernel<EvalHash64CdfKernel::linear_linear>(
          plan, nearest_cache, out, threads);
      return;
    case EvalHash64CdfKernel::cubic_linear:
      fill_eval_hash64_from_nearest_kernel<EvalHash64CdfKernel::cubic_linear>(
          plan, nearest_cache, out, threads);
      return;
    case EvalHash64CdfKernel::linear_cubic:
      fill_eval_hash64_from_nearest_kernel<EvalHash64CdfKernel::linear_cubic>(
          plan, nearest_cache, out, threads);
      return;
    case EvalHash64CdfKernel::cubic_cubic:
      fill_eval_hash64_from_nearest_kernel<EvalHash64CdfKernel::cubic_cubic>(
          plan, nearest_cache, out, threads);
      return;
    case EvalHash64CdfKernel::generic:
      fill_eval_hash64_from_nearest_kernel<EvalHash64CdfKernel::generic>(
          plan, nearest_cache, out, threads);
      return;
  }
}

uint64_t eval_hash64_vector(const EvalHash64Plan& plan, const float* vector) {
  auto nearest = plan.model->router.nearest(vector);
  return eval_hash64_from_centroid(plan, nearest.centroid, nearest.dist2);
}

EvalHash64Plan make_eval_hash64_plan(const VortexModel& model) {
  if (model.hash_bits > 64) {
    throw std::runtime_error("Selector 64-bit hash plan requires hash_bits <= 64");
  }
  if (model.dim == 0 || model.centroids.empty()) {
    throw std::runtime_error("Model is empty");
  }
  if (model.router.active_centroids().empty()) {
    throw std::runtime_error("Model has no active centroids");
  }

  std::size_t centroid_count = model.centroid_count();
  EvalHash64Plan plan;
  plan.model = &model;
  plan.hash_bits = model.hash_bits;
  plan.cdf_top_type = model.cdf_top_type;
  plan.cdf_leaf_type = model.cdf_leaf_type;
  plan.cdf_leaf_count = model.cdf_leaf_count;
  plan.cdf_top_param_count = model.cdf_top_param_count;
  plan.cdf_leaf_param_count = model.cdf_leaf_param_count;
  plan.has_cdf = !model.cdf_rows.empty() &&
                 !model.cdf_top_params.empty() &&
                 !model.cdf_leaf_params.empty();
  if (plan.has_cdf) {
    if (plan.cdf_top_type == CdfModelType::Linear &&
        plan.cdf_leaf_type == CdfModelType::Linear) {
      plan.cdf_kernel = EvalHash64CdfKernel::linear_linear;
    } else if (plan.cdf_top_type == CdfModelType::Cubic &&
               plan.cdf_leaf_type == CdfModelType::Linear) {
      plan.cdf_kernel = EvalHash64CdfKernel::cubic_linear;
    } else if (plan.cdf_top_type == CdfModelType::Linear &&
               plan.cdf_leaf_type == CdfModelType::Cubic) {
      plan.cdf_kernel = EvalHash64CdfKernel::linear_cubic;
    } else if (plan.cdf_top_type == CdfModelType::Cubic &&
               plan.cdf_leaf_type == CdfModelType::Cubic) {
      plan.cdf_kernel = EvalHash64CdfKernel::cubic_cubic;
    }
  }

  if (plan.has_cdf) {
    if (model.cdf_rows.size() != centroid_count) {
      throw std::runtime_error("CDF row count mismatch in selector 64-bit hash plan");
    }
    if (model.cdf_top_params.size() !=
        centroid_count * static_cast<std::size_t>(plan.cdf_top_param_count)) {
      throw std::runtime_error("CDF top params size mismatch in selector 64-bit hash plan");
    }
    if (model.cdf_leaf_params.size() !=
        centroid_count * static_cast<std::size_t>(plan.cdf_leaf_count) *
            static_cast<std::size_t>(plan.cdf_leaf_param_count)) {
      throw std::runtime_error("CDF leaf params size mismatch in selector 64-bit hash plan");
    }
  }
  if (model.min_dist.size() != centroid_count ||
      model.max_dist.size() != centroid_count) {
    throw std::runtime_error("CDF calibration bounds mismatch in selector 64-bit hash plan");
  }

  plan.centroids.resize(centroid_count);
  for (std::size_t centroid = 0; centroid < centroid_count; ++centroid) {
    const auto& start = model.router.range_start(static_cast<uint32_t>(centroid));
    const auto& size = model.router.range_size(static_cast<uint32_t>(centroid));
    auto& centroid_plan = plan.centroids[centroid];
    centroid_plan.range_start_low = start.words[0];
    centroid_plan.range_size_low = size.words[0];
    centroid_plan.range_size_high = size.words[1];
    centroid_plan.min_dist = model.min_dist[centroid];
    centroid_plan.max_dist = model.max_dist[centroid];
    if (plan.has_cdf) {
      centroid_plan.cdf_rows = model.cdf_rows[centroid];
      centroid_plan.top_params =
          model.cdf_top_params.data() + centroid * plan.cdf_top_param_count;
      centroid_plan.leaf_params =
          model.cdf_leaf_params.data() +
          centroid * static_cast<std::size_t>(plan.cdf_leaf_count) *
              plan.cdf_leaf_param_count;
    }
  }
  return plan;
}

void verify_eval_hash64_plan_sample(const EvalHash64Plan& plan,
                                    const VortexModel& model,
                                    const EvalDataset& eval,
                                    const EvalNearestCache* nearest_cache) {
  if (eval.base.count == 0) {
    return;
  }
  constexpr std::size_t kSampleCount = 8;
  std::size_t samples = std::min<std::size_t>(kSampleCount, eval.base.count);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    std::size_t index = samples == 1
        ? 0
        : sample * (eval.base.count - 1) / (samples - 1);
    uint32_t centroid = 0;
    double dist2 = 0.0;
    if (nearest_cache != nullptr) {
      centroid = nearest_cache->centroid[index];
      dist2 = nearest_cache->dist2[index];
    } else {
      auto nearest = model.router.nearest(eval.base.values.data() +
                                          index * eval.base.dim);
      centroid = nearest.centroid;
      dist2 = nearest.dist2;
    }

    uint64_t expected = model.hash_from_centroid_u64(centroid, dist2);
    uint64_t actual = eval_hash64_from_centroid(plan, centroid, dist2);
    if (actual != expected) {
      throw std::runtime_error(
          "Selector 64-bit bulk hash parity check failed at base index " +
          std::to_string(index));
    }
  }
}

} // namespace vortex::selector_internal
