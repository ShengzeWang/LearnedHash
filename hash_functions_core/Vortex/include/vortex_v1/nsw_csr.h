#ifndef VORTEX_V1_NSW_CSR_H
#define VORTEX_V1_NSW_CSR_H

#include "vortex_v1/types.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace vortex {

struct NswCsrHeader {
  char magic[8];
  uint32_t version = 1;
  uint32_t dim = 0;
  uint32_t metric = 0;
  int32_t layer = 0;
  uint64_t node_count = 0;
  uint64_t edge_count = 0;
};

struct NswCsr {
  uint32_t dim = 0;
  Metric metric = Metric::L2;
  int32_t layer = 0;
  std::vector<uint64_t> node_ids;
  std::vector<uint64_t> offsets;
  std::vector<uint32_t> neighbors;

  void write(const std::filesystem::path& path) const;
  static NswCsr read(const std::filesystem::path& path);
};

} // namespace vortex

#endif // VORTEX_V1_NSW_CSR_H
