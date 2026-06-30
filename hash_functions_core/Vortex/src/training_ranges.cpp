#include "vortex_v1/training.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vortex {
namespace {

void validate_centroid_order(const std::vector<uint32_t>& order, std::size_t expected) {
  if (order.size() != expected) {
    throw std::runtime_error("Centroid order size mismatch");
  }
  std::vector<uint8_t> seen(expected, 0);
  for (uint32_t cid : order) {
    if (static_cast<std::size_t>(cid) >= expected) {
      throw std::runtime_error("Centroid order contains out-of-range index");
    }
    if (seen[cid] != 0) {
      throw std::runtime_error("Centroid order contains duplicate index");
    }
    seen[cid] = 1;
  }
}

uint64_t checked_total_mass(const std::vector<uint64_t>& mass) {
  unsigned __int128 total = 0;
  for (uint64_t m : mass) {
    total += static_cast<unsigned __int128>(m);
    if (total > static_cast<unsigned __int128>(std::numeric_limits<uint64_t>::max())) {
      throw std::runtime_error("Total centroid mass overflow");
    }
  }
  if (total == 0) {
    throw std::runtime_error("Total mass is zero");
  }
  return static_cast<uint64_t>(total);
}

void validate_allocated_ranges(const std::vector<uint32_t>& order,
                               const std::vector<UInt256>& range_start,
                               const std::vector<UInt256>& range_size,
                               const std::vector<uint8_t>& range_full,
                               uint32_t hash_bits) {
  if (range_start.size() != order.size() || range_size.size() != order.size() ||
      range_full.size() != order.size()) {
    throw std::runtime_error("Range vectors size mismatch");
  }

  UInt256 cursor = UInt256::zero();
  for (uint32_t cid : order) {
    if (compare(range_start[cid], cursor) != 0) {
      throw std::runtime_error("Allocated ranges are not contiguous");
    }
    cursor = add(cursor, range_size[cid]);
  }

  if (hash_bits < 256) {
    UInt256 expected = UInt256::one_shifted(hash_bits);
    if (compare(cursor, expected) != 0) {
      throw std::runtime_error("Allocated ranges do not cover hash space");
    }
  } else {
    uint32_t full_count = 0;
    for (uint8_t full : range_full) {
      if (full != 0) {
        ++full_count;
      }
    }
    if (full_count > 1) {
      throw std::runtime_error("At most one centroid can own the full 256-bit range");
    }
  }
}

} // namespace

RangeAllocResult allocate_ranges(const std::vector<uint64_t>& mass,
                                 const std::vector<uint32_t>& order,
                                 uint32_t hash_bits) {
  if (mass.size() != order.size()) {
    throw std::runtime_error("Mass and order sizes differ");
  }
  if (hash_bits == 0 || hash_bits > 256) {
    throw std::runtime_error("hash_bits must be in [1, 256]");
  }

  RangeAllocResult result;
  std::size_t k = mass.size();
  validate_centroid_order(order, k);
  std::vector<UInt256> range_start_u(k, UInt256::zero());
  std::vector<UInt256> range_size_u(k, UInt256::zero());
  result.range_start.assign(k, rm_model::CentroidRouter::Range256{});
  result.range_size.assign(k, rm_model::CentroidRouter::Range256{});
  result.range_full.assign(k, 0);

  uint64_t total_mass = checked_total_mass(mass);

  std::vector<uint64_t> remainders(k, 0);
  unsigned __int128 sum_remainders = 0;

  bool full_range_assigned = false;
  for (std::size_t idx = 0; idx < k; ++idx) {
    uint32_t cid = order[idx];
    uint64_t m = mass[cid];
    if (m == 0) {
      continue;
    }
    if (hash_bits == 256 && m == total_mass && !full_range_assigned) {
      result.range_full[cid] = 1;
      full_range_assigned = true;
      continue;
    }
    UInt320 numerator = shl_u64_to_320(m, hash_bits);
    uint64_t rem = 0;
    UInt256 q = div_u64(numerator, total_mass, &rem);
    range_size_u[cid] = q;
    remainders[cid] = rem;
    sum_remainders += rem;
  }

  uint64_t remainder_count = static_cast<uint64_t>(sum_remainders / total_mass);
  if (remainder_count > 0) {
    std::vector<std::pair<uint64_t, uint32_t>> frac;
    frac.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
      if (mass[i] == 0 || result.range_full[i]) {
        continue;
      }
      frac.emplace_back(remainders[i], static_cast<uint32_t>(i));
    }
    std::sort(frac.begin(), frac.end(), [](const auto& a, const auto& b) {
      if (a.first != b.first) return a.first > b.first;
      return a.second < b.second;
    });
    for (uint64_t i = 0; i < remainder_count && i < frac.size(); ++i) {
      uint32_t cid = frac[static_cast<std::size_t>(i)].second;
      range_size_u[cid] = add_u64(range_size_u[cid], 1);
    }
  }

  UInt256 cursor = UInt256::zero();
  for (std::size_t idx = 0; idx < k; ++idx) {
    uint32_t cid = order[idx];
    range_start_u[cid] = cursor;
    cursor = add(cursor, range_size_u[cid]);
  }

  for (std::size_t i = 0; i < k; ++i) {
    result.range_start[i].words = range_start_u[i].words;
    result.range_size[i].words = range_size_u[i].words;
  }
  validate_allocated_ranges(order, range_start_u, range_size_u, result.range_full, hash_bits);

  return result;
}

} // namespace vortex
