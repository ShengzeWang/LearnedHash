#include "codegen_internal.h"

#include "vortex_v1/detail/binary_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vortex::codegen_internal {

constexpr char kCentroidIndexMagic[8] = {'V', 'T', 'X', 'C', 'I', '1', '\0', '\0'};
constexpr uint32_t kCentroidIndexVersion = 1;

struct CentroidIndexBlob {
  std::vector<uint32_t> active_ids;
  std::vector<uint32_t> pivot_ids;
  std::vector<float> centroid_pivot_l2;
};

double l2_dist2_dense(const float* a, const float* b, uint32_t dim) {
  double acc = 0.0;
  for (uint32_t d = 0; d < dim; ++d) {
    double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
    acc += diff * diff;
  }
  return acc;
}

CentroidIndexBlob build_centroid_index_blob(const VortexModel& model, uint32_t pivot_count) {
  CentroidIndexBlob blob;
  if (model.dim == 0 || model.centroids.empty()) {
    return blob;
  }

  const auto& active = model.router.active_centroids();
  if (!active.empty()) {
    blob.active_ids = active;
  } else {
    std::size_t count = model.centroid_count();
    blob.active_ids.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      blob.active_ids[i] = static_cast<uint32_t>(i);
    }
  }
  if (blob.active_ids.empty()) {
    return blob;
  }

  auto centroid_ptr = [&](uint32_t idx) {
    return model.centroids.data() + static_cast<std::size_t>(idx) * model.dim;
  };

  std::size_t active_count = blob.active_ids.size();
  std::size_t pivots = static_cast<std::size_t>(pivot_count == 0 ? 1 : pivot_count);
  if (pivots > active_count) {
    pivots = active_count;
  }

  blob.pivot_ids.reserve(pivots);
  blob.pivot_ids.push_back(blob.active_ids.front());
  if (pivots > 1) {
    std::vector<double> min_d2(active_count, std::numeric_limits<double>::infinity());
    const float* first = centroid_ptr(blob.pivot_ids.front());
    for (std::size_t i = 0; i < active_count; ++i) {
      min_d2[i] = l2_dist2_dense(centroid_ptr(blob.active_ids[i]), first, model.dim);
    }
    while (blob.pivot_ids.size() < pivots) {
      std::size_t best_pos = 0;
      double best_val = -1.0;
      for (std::size_t i = 0; i < active_count; ++i) {
        if (min_d2[i] > best_val) {
          best_val = min_d2[i];
          best_pos = i;
        }
      }
      uint32_t pivot_id = blob.active_ids[best_pos];
      blob.pivot_ids.push_back(pivot_id);
      const float* pivot_vec = centroid_ptr(pivot_id);
      for (std::size_t i = 0; i < active_count; ++i) {
        double d2 = l2_dist2_dense(centroid_ptr(blob.active_ids[i]), pivot_vec, model.dim);
        if (d2 < min_d2[i]) {
          min_d2[i] = d2;
        }
      }
    }
  }

  std::size_t pivot_count_sz = blob.pivot_ids.size();
  blob.centroid_pivot_l2.resize(active_count * pivot_count_sz);
  for (std::size_t i = 0; i < active_count; ++i) {
    const float* centroid_vec = centroid_ptr(blob.active_ids[i]);
    for (std::size_t p = 0; p < pivot_count_sz; ++p) {
      const float* pivot_vec = centroid_ptr(blob.pivot_ids[p]);
      double d2 = l2_dist2_dense(centroid_vec, pivot_vec, model.dim);
      blob.centroid_pivot_l2[i * pivot_count_sz + p] = static_cast<float>(std::sqrt(d2));
    }
  }
  return blob;
}

void write_centroid_index_file(const VortexModel& model,
                               const std::filesystem::path& path,
                               uint32_t pivot_count) {
  CentroidIndexBlob blob = build_centroid_index_blob(model, pivot_count);
  if (blob.active_ids.empty() || blob.pivot_ids.empty()) {
    return;
  }
  if (blob.active_ids.size() > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
      blob.pivot_ids.size() > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("Centroid index is too large to serialize");
  }
  if (model.centroid_count() > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("Centroid count is too large to serialize");
  }

  struct Header {
    char magic[8];
    uint32_t version = 0;
    uint32_t dim = 0;
    uint32_t centroid_count = 0;
    uint32_t active_count = 0;
    uint32_t pivot_count = 0;
  } header{};

  std::memcpy(header.magic, kCentroidIndexMagic, sizeof(header.magic));
  header.version = kCentroidIndexVersion;
  header.dim = model.dim;
  header.centroid_count = static_cast<uint32_t>(model.centroid_count());
  header.active_count = static_cast<uint32_t>(blob.active_ids.size());
  header.pivot_count = static_cast<uint32_t>(blob.pivot_ids.size());

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Unable to write centroid index file: " + path.string());
  }
  detail::write_bytes(out, &header, sizeof(header), "Unable to write codegen centroid index");
  detail::write_bytes(out,
                      blob.active_ids.data(),
                      blob.active_ids.size() * sizeof(uint32_t),
                      "Unable to write codegen centroid index");
  detail::write_bytes(out,
                      blob.pivot_ids.data(),
                      blob.pivot_ids.size() * sizeof(uint32_t),
                      "Unable to write codegen centroid index");
  detail::write_bytes(out,
                      blob.centroid_pivot_l2.data(),
                      blob.centroid_pivot_l2.size() * sizeof(float),
                      "Unable to write codegen centroid index");
}

} // namespace vortex::codegen_internal
