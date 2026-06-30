#include "vortex_v1/model.h"

#include "vortex_v1/uint256.h"

#include "vortex_v1/detail/binary_io.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace vortex {

namespace {

constexpr char kMagic[8] = {'V','T','X','M','2','\0','\0','\0'};

UInt256 to_uint256(const rm_model::CentroidRouter::Range256& range) {
  UInt256 out{};
  out.words = range.words;
  return out;
}

inline double exp1(double inp) {
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

inline double phi(double x) {
  return 1.0 / (1.0 + exp1(-1.65451 * x));
}

inline double clamp_unit(double value) {
  if (value < 0.0) return 0.0;
  if (value >= 1.0) return std::nextafter(1.0, 0.0);
  return value;
}

inline std::size_t clamp_index(double pred, std::size_t bound) {
  if (bound == 0) return 0;
  if (!std::isfinite(pred)) return 0;
  if (pred < 0.0) return 0;
  double max_val = static_cast<double>(bound - 1);
  if (pred > max_val) {
    return bound - 1;
  }
  return static_cast<std::size_t>(pred);
}

double eval_cdf_model(const VortexModel& model, uint32_t centroid, double dist2) {
  if (model.cdf_rows.empty() || model.cdf_top_params.empty() || model.cdf_leaf_params.empty()) {
    return 0.0;
  }
  uint64_t rows = model.cdf_rows[centroid];
  if (rows < 2 || model.cdf_leaf_count == 0) {
    return 0.0;
  }
  double bound = static_cast<double>(rows - 1);

  std::size_t top_offset = static_cast<std::size_t>(centroid) * model.cdf_top_param_count;
  const double* top_params = model.cdf_top_params.data() + top_offset;

  auto eval_model = [&](CdfModelType type, const double* params, double x) -> double {
    switch (type) {
      case CdfModelType::Linear:
        return std::fma(params[1], x, params[0]);
      case CdfModelType::LogLinear:
        return exp1(std::fma(params[1], x, params[0]));
      case CdfModelType::Cubic: {
        double v1 = std::fma(params[0], x, params[1]);
        double v2 = std::fma(v1, x, params[2]);
        return std::fma(v2, x, params[3]);
      }
      case CdfModelType::Normal:
        if (params[1] <= 0.0) return 0.0;
        return phi((x - params[0]) / params[1]) * params[2];
      case CdfModelType::LogNormal: {
        if (params[1] <= 0.0 || x <= 0.0) return 0.0;
        double lx = std::max(std::log(x), 0.0);
        return phi((lx - params[0]) / params[1]) * params[2];
      }
    }
    return 0.0;
  };

  double top_pred = eval_model(model.cdf_top_type, top_params, dist2);
  if (!std::isfinite(top_pred)) {
    top_pred = 0.0;
  }
  std::size_t leaf_idx = clamp_index(top_pred, model.cdf_leaf_count);
  std::size_t leaf_offset =
      (static_cast<std::size_t>(centroid) * model.cdf_leaf_count + leaf_idx) *
      model.cdf_leaf_param_count;
  const double* leaf_params = model.cdf_leaf_params.data() + leaf_offset;
  double pred = eval_model(model.cdf_leaf_type, leaf_params, dist2);
  if (!std::isfinite(pred)) {
    pred = 0.0;
  }

  if (pred < 0.0) pred = 0.0;
  if (pred > bound) pred = bound;
  return clamp_unit(pred / bound);
}

uint64_t to_q64(double value) {
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

uint64_t mask_u64_bits(uint64_t value, uint32_t bits) {
  if (bits == 0) {
    return 0;
  }
  if (bits >= 64) {
    return value;
  }
  return value & ((1ULL << bits) - 1ULL);
}

uint64_t mul_q64_low_u64(const rm_model::CentroidRouter::Range256& range,
                         uint64_t q) {
  if (q == 0) {
    return 0;
  }
#if defined(_MSC_VER)
  uint64_t high0 = 0;
  (void)_umul128(range.words[0], q, &high0);
  uint64_t high1 = 0;
  uint64_t low1 = _umul128(range.words[1], q, &high1);
  (void)high1;
  return high0 + low1;
#else
  unsigned __int128 prod0 =
      static_cast<unsigned __int128>(range.words[0]) * q;
  uint64_t scaled = static_cast<uint64_t>(prod0 >> 64);
  if (range.words[1] != 0) {
    unsigned __int128 prod1 =
        static_cast<unsigned __int128>(range.words[1]) * q;
    scaled += static_cast<uint64_t>(prod1);
  }
  return scaled;
#endif
}

void validate_hash_centroid_input(const VortexModel& model,
                                  uint32_t centroid,
                                  double dist2) {
  if (model.dim == 0 || model.centroids.empty()) {
    throw std::runtime_error("Model is empty");
  }
  if (model.router.active_centroids().empty()) {
    throw std::runtime_error("Model has no active centroids");
  }
  if (!std::isfinite(dist2) || dist2 < 0.0) {
    throw std::runtime_error("Invalid centroid distance");
  }
  if (centroid >= model.centroid_count()) {
    throw std::runtime_error("Centroid index out of bounds");
  }
}

double hash_fraction_from_centroid(const VortexModel& model,
                                   uint32_t centroid,
                                   double dist2) {
  double value = eval_cdf_model(model, centroid, dist2);
  if (value <= 0.0) {
    double min_v = model.min_dist[centroid];
    double max_v = model.max_dist[centroid];
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

UInt256 mul_q64(const UInt256& range, uint64_t q, bool full_range) {
  if (full_range) {
    if (q == 0) {
      return UInt256::zero();
    }
    UInt256 out{};
    out.words[3] = q;
    return out;
  }
  if (q == 0 || range.is_zero()) {
    return UInt256::zero();
  }
  UInt320 prod = mul_u64_wide(range, q);
  return shr_320_64(prod);
}

} // namespace

const char* cdf_model_name(CdfModelType type) {
  switch (type) {
    case CdfModelType::Linear:
      return "linear";
    case CdfModelType::LogLinear:
      return "loglinear";
    case CdfModelType::Cubic:
      return "cubic";
    case CdfModelType::Normal:
      return "ncdf";
    case CdfModelType::LogNormal:
      return "lncdf";
  }
  return "linear";
}

CdfModelType parse_cdf_model(const std::string& name) {
  if (name == "linear" || name == "linear_spline" || name == "robust_linear") {
    return CdfModelType::Linear;
  }
  if (name == "loglinear") {
    return CdfModelType::LogLinear;
  }
  if (name == "cubic") {
    return CdfModelType::Cubic;
  }
  if (name == "normal" || name == "ncdf") {
    return CdfModelType::Normal;
  }
  if (name == "lognormal" || name == "lncdf") {
    return CdfModelType::LogNormal;
  }
  throw std::runtime_error("Unsupported CDF model type: " + name);
}

uint32_t cdf_param_count(CdfModelType type) {
  switch (type) {
    case CdfModelType::Linear:
    case CdfModelType::LogLinear:
      return 2;
    case CdfModelType::Cubic:
      return 4;
    case CdfModelType::Normal:
    case CdfModelType::LogNormal:
      return 3;
  }
  return 0;
}

VortexModel::VortexModel(const VortexModel& other)
    : dim(other.dim),
      metric(other.metric),
      hash_bits(other.hash_bits),
      centroids(other.centroids),
      order(other.order),
      mass(other.mass),
      range_start(other.range_start),
      range_size(other.range_size),
      range_full(other.range_full),
      min_dist(other.min_dist),
      max_dist(other.max_dist),
      cdf_top_type(other.cdf_top_type),
      cdf_leaf_type(other.cdf_leaf_type),
      cdf_leaf_count(other.cdf_leaf_count),
      cdf_top_param_count(other.cdf_top_param_count),
      cdf_leaf_param_count(other.cdf_leaf_param_count),
      cdf_rows(other.cdf_rows),
      cdf_top_params(other.cdf_top_params),
      cdf_leaf_params(other.cdf_leaf_params) {
  compute_active();
}

VortexModel& VortexModel::operator=(const VortexModel& other) {
  if (this == &other) {
    return *this;
  }
  dim = other.dim;
  metric = other.metric;
  hash_bits = other.hash_bits;
  centroids = other.centroids;
  order = other.order;
  mass = other.mass;
  range_start = other.range_start;
  range_size = other.range_size;
  range_full = other.range_full;
  min_dist = other.min_dist;
  max_dist = other.max_dist;
  cdf_top_type = other.cdf_top_type;
  cdf_leaf_type = other.cdf_leaf_type;
  cdf_leaf_count = other.cdf_leaf_count;
  cdf_top_param_count = other.cdf_top_param_count;
  cdf_leaf_param_count = other.cdf_leaf_param_count;
  cdf_rows = other.cdf_rows;
  cdf_top_params = other.cdf_top_params;
  cdf_leaf_params = other.cdf_leaf_params;
  compute_active();
  return *this;
}

VortexModel::VortexModel(VortexModel&& other) noexcept
    : dim(other.dim),
      metric(other.metric),
      hash_bits(other.hash_bits),
      centroids(std::move(other.centroids)),
      order(std::move(other.order)),
      mass(std::move(other.mass)),
      range_start(std::move(other.range_start)),
      range_size(std::move(other.range_size)),
      range_full(std::move(other.range_full)),
      min_dist(std::move(other.min_dist)),
      max_dist(std::move(other.max_dist)),
      cdf_top_type(other.cdf_top_type),
      cdf_leaf_type(other.cdf_leaf_type),
      cdf_leaf_count(other.cdf_leaf_count),
      cdf_top_param_count(other.cdf_top_param_count),
      cdf_leaf_param_count(other.cdf_leaf_param_count),
      cdf_rows(std::move(other.cdf_rows)),
      cdf_top_params(std::move(other.cdf_top_params)),
      cdf_leaf_params(std::move(other.cdf_leaf_params)) {
  compute_active();
  other.router = rm_model::CentroidRouter();
}

VortexModel& VortexModel::operator=(VortexModel&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  dim = other.dim;
  metric = other.metric;
  hash_bits = other.hash_bits;
  centroids = std::move(other.centroids);
  order = std::move(other.order);
  mass = std::move(other.mass);
  range_start = std::move(other.range_start);
  range_size = std::move(other.range_size);
  range_full = std::move(other.range_full);
  min_dist = std::move(other.min_dist);
  max_dist = std::move(other.max_dist);
  cdf_top_type = other.cdf_top_type;
  cdf_leaf_type = other.cdf_leaf_type;
  cdf_leaf_count = other.cdf_leaf_count;
  cdf_top_param_count = other.cdf_top_param_count;
  cdf_leaf_param_count = other.cdf_leaf_param_count;
  cdf_rows = std::move(other.cdf_rows);
  cdf_top_params = std::move(other.cdf_top_params);
  cdf_leaf_params = std::move(other.cdf_leaf_params);
  compute_active();
  other.router = rm_model::CentroidRouter();
  return *this;
}

void VortexModel::compute_active() {
  if (dim == 0 || centroids.empty() || range_start.empty() || range_size.empty() ||
      range_full.empty()) {
    router = rm_model::CentroidRouter();
    return;
  }
  router.set_centroids_view(dim, centroids.data(), centroid_count());
  router.set_ranges_view(hash_bits,
                         range_start.data(),
                         range_size.data(),
                         range_full.data(),
                         range_start.size());
  router.compute_active_from_ranges();
}

void VortexModel::write(const std::filesystem::path& path) const {
  if (dim == 0 || centroids.empty()) {
    throw std::runtime_error("Model is empty");
  }
  std::size_t k = centroid_count();
  if (cdf_rows.size() != k) {
    throw std::runtime_error("CDF rows size mismatch");
  }
  if (cdf_top_param_count == 0 || cdf_leaf_param_count == 0 || cdf_leaf_count == 0) {
    throw std::runtime_error("CDF model configuration is incomplete");
  }
  if (cdf_top_params.size() != k * cdf_top_param_count) {
    throw std::runtime_error("CDF top param size mismatch");
  }
  if (cdf_leaf_params.size() != k * cdf_leaf_count * cdf_leaf_param_count) {
    throw std::runtime_error("CDF leaf param size mismatch");
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Unable to open model file for writing: " + path.string());
  }

  VortexModelHeader header;
  std::memcpy(header.magic, kMagic, sizeof(kMagic));
  header.dim = dim;
  header.metric = static_cast<uint32_t>(metric);
  header.hash_bits = hash_bits;
  header.centroid_count = static_cast<uint32_t>(centroid_count());
  header.cdf_top_type = static_cast<uint32_t>(cdf_top_type);
  header.cdf_leaf_type = static_cast<uint32_t>(cdf_leaf_type);
  header.cdf_leaf_count = cdf_leaf_count;
  header.cdf_top_param_count = cdf_top_param_count;
  header.cdf_leaf_param_count = cdf_leaf_param_count;

  detail::write_bytes(out, &header, sizeof(header), "Failed to write model file");
  detail::write_vector(out, centroids, "Failed to write model file");
  detail::write_vector(out, order, "Failed to write model file");
  detail::write_vector(out, mass, "Failed to write model file");
  detail::write_vector(out, range_start, "Failed to write model file");
  detail::write_vector(out, range_size, "Failed to write model file");
  detail::write_vector(out, range_full, "Failed to write model file");
  detail::write_vector(out, min_dist, "Failed to write model file");
  detail::write_vector(out, max_dist, "Failed to write model file");
  detail::write_vector(out, cdf_rows, "Failed to write model file");
  detail::write_vector(out, cdf_top_params, "Failed to write model file");
  detail::write_vector(out, cdf_leaf_params, "Failed to write model file");
}

VortexModel VortexModel::read(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Unable to open model file: " + path.string());
  }

  VortexModelHeader header{};
  detail::read_bytes(in, &header, sizeof(header), "Failed to read model file");
  if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("Invalid vortex model magic");
  }
  if (header.version != 2) {
    throw std::runtime_error("Unsupported vortex model version");
  }

  VortexModel model;
  model.dim = header.dim;
  model.metric = static_cast<Metric>(header.metric);
  model.hash_bits = header.hash_bits;

  std::size_t k = header.centroid_count;
  detail::read_vector(in, model.centroids, k * model.dim, "Failed to read model file");
  detail::read_vector(in, model.order, k, "Failed to read model file");
  detail::read_vector(in, model.mass, k, "Failed to read model file");
  detail::read_vector(in, model.range_start, k, "Failed to read model file");
  detail::read_vector(in, model.range_size, k, "Failed to read model file");
  detail::read_vector(in, model.range_full, k, "Failed to read model file");
  detail::read_vector(in, model.min_dist, k, "Failed to read model file");
  detail::read_vector(in, model.max_dist, k, "Failed to read model file");

  model.cdf_top_type = static_cast<CdfModelType>(header.cdf_top_type);
  model.cdf_leaf_type = static_cast<CdfModelType>(header.cdf_leaf_type);
  model.cdf_leaf_count = header.cdf_leaf_count;
  model.cdf_top_param_count = header.cdf_top_param_count;
  model.cdf_leaf_param_count = header.cdf_leaf_param_count;

  detail::read_vector(in, model.cdf_rows, k, "Failed to read model file");
  detail::read_vector(in, model.cdf_top_params,
                      static_cast<std::size_t>(k) * model.cdf_top_param_count,
                      "Failed to read model file");
  detail::read_vector(in, model.cdf_leaf_params,
                      static_cast<std::size_t>(k) * model.cdf_leaf_count *
                          model.cdf_leaf_param_count,
                      "Failed to read model file");

  if (model.cdf_leaf_count == 0 || model.cdf_top_param_count == 0 || model.cdf_leaf_param_count == 0) {
    throw std::runtime_error("Invalid CDF configuration in model file");
  }
  if (model.cdf_rows.size() != k) {
    throw std::runtime_error("CDF rows size mismatch in model file");
  }
  if (model.cdf_top_params.size() != k * model.cdf_top_param_count) {
    throw std::runtime_error("CDF top params size mismatch in model file");
  }
  if (model.cdf_leaf_params.size() != k * model.cdf_leaf_count * model.cdf_leaf_param_count) {
    throw std::runtime_error("CDF leaf params size mismatch in model file");
  }

  model.compute_active();
  return model;
}

HashOutput VortexModel::hash(const float* vector) const {
  return hash_with_pred(vector, nullptr);
}

uint64_t VortexModel::hash_u64(const float* vector, double* pred) const {
  if (hash_bits > 64) {
    throw std::runtime_error("hash_u64 requires hash_bits <= 64");
  }
  if (dim == 0 || centroids.empty()) {
    throw std::runtime_error("Model is empty");
  }
  if (router.active_centroids().empty()) {
    throw std::runtime_error("Model has no active centroids");
  }

  auto nearest = router.nearest(vector);
  return hash_from_centroid_u64(nearest.centroid, nearest.dist2, pred);
}

HashOutput VortexModel::hash_with_pred(const float* vector, double* pred) const {
  if (dim == 0 || centroids.empty()) {
    throw std::runtime_error("Model is empty");
  }
  if (router.active_centroids().empty()) {
    throw std::runtime_error("Model has no active centroids");
  }

  auto nearest = router.nearest(vector);
  return hash_from_centroid(nearest.centroid, nearest.dist2, pred);
}

HashOutput VortexModel::hash_from_centroid(uint32_t centroid, double dist2, double* pred) const {
  validate_hash_centroid_input(*this, centroid, dist2);
  double value = hash_fraction_from_centroid(*this, centroid, dist2);
  if (pred) {
    *pred = value;
  }

  uint64_t q = to_q64(value);
  bool full_range = (hash_bits == 256) && router.range_full(centroid);
  UInt256 range_size_u = to_uint256(router.range_size(centroid));
  UInt256 range_start_u = to_uint256(router.range_start(centroid));
  UInt256 scaled = mul_q64(range_size_u, q, full_range);
  UInt256 out = add(range_start_u, scaled);
  if (hash_bits < 256) {
    out.mask_bits(hash_bits);
  }

  HashOutput result;
  result.value = out;
  result.bits = hash_bits;
  return result;
}

uint64_t VortexModel::hash_from_centroid_u64(uint32_t centroid,
                                             double dist2,
                                             double* pred) const {
  if (hash_bits > 64) {
    throw std::runtime_error("hash_from_centroid_u64 requires hash_bits <= 64");
  }
  validate_hash_centroid_input(*this, centroid, dist2);
  double value = hash_fraction_from_centroid(*this, centroid, dist2);
  if (pred) {
    *pred = value;
  }

  uint64_t q = to_q64(value);
  const auto& start = router.range_start(centroid);
  const auto& size = router.range_size(centroid);
  uint64_t scaled = mul_q64_low_u64(size, q);
  return mask_u64_bits(start.words[0] + scaled, hash_bits);
}

} // namespace vortex
