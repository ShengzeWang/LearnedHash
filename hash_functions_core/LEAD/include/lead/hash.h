#ifndef LEAD_HASH_H
#define LEAD_HASH_H

#include "lead/uint256.h"

#include <array>
#include <cstdint>
#include <string>

namespace lead {

struct HashOutput {
  UInt256 value;
  uint32_t bits = 256;

  std::string to_hex() const { return lead::to_hex(value, bits); }
  std::array<uint8_t, 32> bytes_be() const { return lead::to_bytes_be(value); }
  uint64_t to_u64_msb() const { return lead::msb_u64(value, bits); }
  std::array<uint64_t, 2> to_u128_msb() const { return lead::msb_u128(value, bits); }
};

} // namespace lead

#endif // LEAD_HASH_H
