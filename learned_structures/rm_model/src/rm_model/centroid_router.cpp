#include "rm_model/centroid_router.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rm_model {

void CentroidRouter::set_centroids_view(uint32_t dim, const float* data, std::size_t count) {
  if (dim == 0 || data == nullptr || count == 0) {
    throw std::runtime_error("CentroidRouter: centroids view is empty");
  }
  dim_ = dim;
  centroids_ = data;
  centroid_count_ = count;
  active_is_full_ordered_ = false;
}

void CentroidRouter::set_ranges_view(uint32_t hash_bits,
                                     const Range256* starts,
                                     const Range256* sizes,
                                     const uint8_t* full_flags,
                                     std::size_t count) {
  if (count == 0 || starts == nullptr || sizes == nullptr || full_flags == nullptr) {
    throw std::runtime_error("CentroidRouter: ranges view is empty");
  }
  if (hash_bits == 0 || hash_bits > 256) {
    throw std::runtime_error("CentroidRouter: hash_bits must be in [1, 256]");
  }
  if (centroid_count_ != 0 && count != centroid_count_) {
    throw std::runtime_error("CentroidRouter: range count must match centroid count");
  }
  hash_bits_ = hash_bits;
  range_start_ = starts;
  range_size_ = sizes;
  range_full_ = full_flags;
  centroid_count_ = count;
  active_is_full_ordered_ = false;
}

void CentroidRouter::compute_active_from_ranges() {
  active_.clear();
  active_is_full_ordered_ = false;
  if (centroid_count_ == 0 || range_size_ == nullptr || range_full_ == nullptr) {
    return;
  }
  active_.reserve(centroid_count_);
  for (std::size_t i = 0; i < centroid_count_; ++i) {
    bool full = range_full_[i] != 0;
    bool nonzero = false;
    if (!full) {
      const auto& words = range_size_[i].words;
      nonzero = (words[0] != 0 || words[1] != 0 || words[2] != 0 || words[3] != 0);
    }
    if (full || nonzero) {
      active_.push_back(static_cast<uint32_t>(i));
    }
  }
  active_is_full_ordered_ = active_.size() == centroid_count_;
}

CentroidRouter::QueryResult CentroidRouter::nearest(const float* vector) const {
  if (centroids_ == nullptr || centroid_count_ == 0 || dim_ == 0) {
    throw std::runtime_error("CentroidRouter: centroids not configured");
  }
  if (vector == nullptr) {
    throw std::runtime_error("CentroidRouter: input vector is null");
  }

  double best = std::numeric_limits<double>::infinity();
  uint32_t best_idx = 0;

  if (active_.empty() || active_is_full_ordered_) {
    for (std::size_t idx = 0; idx < centroid_count_; ++idx) {
      const float* c = centroids_ + idx * dim_;
      double acc = 0.0;
      for (uint32_t d = 0; d < dim_; ++d) {
        double diff = static_cast<double>(vector[d]) - static_cast<double>(c[d]);
        acc += diff * diff;
      }
      if (acc < best) {
        best = acc;
        best_idx = static_cast<uint32_t>(idx);
      }
    }
  } else {
    for (uint32_t idx : active_) {
      const float* c = centroids_ + static_cast<std::size_t>(idx) * dim_;
      double acc = 0.0;
      for (uint32_t d = 0; d < dim_; ++d) {
        double diff = static_cast<double>(vector[d]) - static_cast<double>(c[d]);
        acc += diff * diff;
      }
      if (acc < best) {
        best = acc;
        best_idx = idx;
      }
    }
  }

  return {best_idx, best};
}

const CentroidRouter::Range256& CentroidRouter::range_start(uint32_t idx) const {
  if (!range_start_ || idx >= centroid_count_) {
    throw std::runtime_error("CentroidRouter: range_start out of bounds");
  }
  return range_start_[idx];
}

const CentroidRouter::Range256& CentroidRouter::range_size(uint32_t idx) const {
  if (!range_size_ || idx >= centroid_count_) {
    throw std::runtime_error("CentroidRouter: range_size out of bounds");
  }
  return range_size_[idx];
}

bool CentroidRouter::range_full(uint32_t idx) const {
  if (!range_full_ || idx >= centroid_count_) {
    throw std::runtime_error("CentroidRouter: range_full out of bounds");
  }
  return range_full_[idx] != 0;
}

} // namespace rm_model
