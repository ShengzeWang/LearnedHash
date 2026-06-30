#ifndef LEAD_MAPPING_H
#define LEAD_MAPPING_H

#include "lead/hash.h"

#include <array>
#include <cstdint>
#include <string>

namespace lead {

struct MappingConfig {
  uint64_t max_index = 0;
  uint32_t space_bits = 64;
  double scale_factor = 1.0;
  std::array<uint64_t, 4> scale_words{}; // little-endian

  static MappingConfig compute(uint64_t max_index,
                               uint32_t space_bits,
                               double scale_factor,
                               std::string* warning);
};

class IndexMapper {
 public:
  explicit IndexMapper(MappingConfig config);

  HashOutput map(uint64_t index) const;
  HashOutput map_double(double prediction) const;
  uint64_t max_index() const { return config_.max_index; }
  uint32_t space_bits() const { return config_.space_bits; }

 private:
  MappingConfig config_;
  UInt256 scaled_range_{};
  uint64_t scaled_range_u64_ = 0;
};

} // namespace lead

#endif // LEAD_MAPPING_H
