#ifndef RM_MODEL_CENTROID_ROUTER_H
#define RM_MODEL_CENTROID_ROUTER_H

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rm_model {

class CentroidRouter {
 public:
  struct Range256 {
    std::array<uint64_t, 4> words{};
  };

  struct QueryResult {
    uint32_t centroid = 0;
    double dist2 = 0.0;
  };

  void set_centroids_view(uint32_t dim, const float* data, std::size_t count);
  void set_ranges_view(uint32_t hash_bits,
                       const Range256* starts,
                       const Range256* sizes,
                       const uint8_t* full_flags,
                       std::size_t count);

  void compute_active_from_ranges();

  QueryResult nearest(const float* vector) const;

  uint32_t dim() const { return dim_; }
  std::size_t centroid_count() const { return centroid_count_; }
  uint32_t hash_bits() const { return hash_bits_; }

  const Range256& range_start(uint32_t idx) const;
  const Range256& range_size(uint32_t idx) const;
  bool range_full(uint32_t idx) const;

  const std::vector<uint32_t>& active_centroids() const { return active_; }

 private:
  const float* centroids_ = nullptr;
  const Range256* range_start_ = nullptr;
  const Range256* range_size_ = nullptr;
  const uint8_t* range_full_ = nullptr;
  std::size_t centroid_count_ = 0;
  uint32_t dim_ = 0;
  uint32_t hash_bits_ = 0;
  std::vector<uint32_t> active_;
  bool active_is_full_ordered_ = false;
};

} // namespace rm_model

#endif // RM_MODEL_CENTROID_ROUTER_H
