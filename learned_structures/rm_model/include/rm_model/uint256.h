#ifndef RM_MODEL_UINT256_H
#define RM_MODEL_UINT256_H

#include <array>
#include <cstdint>
#include <string>

namespace rm_model {

struct UInt256 {
  std::array<uint64_t, 4> words{}; // little-endian: words[0] is least significant

  static UInt256 zero();
  static UInt256 from_u64(uint64_t value);
  static UInt256 one_shifted(uint32_t bits);
  static UInt256 max_for_bits(uint32_t bits);

  bool is_zero() const;
  void mask_bits(uint32_t bits);
};

struct UInt320 {
  std::array<uint64_t, 5> words{}; // little-endian
};

UInt320 mul_u64_wide(const UInt256& value, uint64_t mul);
UInt256 mul_u64(const UInt256& value, uint64_t mul, uint64_t* overflow);
UInt256 div_u64(const UInt256& value, uint64_t divisor, uint64_t* remainder);
UInt256 div_u64(const UInt320& value, uint64_t divisor, uint64_t* remainder);
UInt256 shr_320(const UInt320& value, uint32_t shift_bits);
UInt256 shr_320_64(const UInt320& value);
UInt256 shr_256(const UInt256& value, uint32_t shift_bits);
UInt320 shl_u64_to_320(uint64_t value, uint32_t shift_bits);
UInt256 add(const UInt256& a, const UInt256& b);
UInt256 add_u64(const UInt256& value, uint64_t addend);
UInt256 sub(const UInt256& a, const UInt256& b);
int compare(const UInt256& a, const UInt256& b);

uint64_t to_u64(const UInt256& value, uint32_t bits);
uint64_t msb_u64(const UInt256& value, uint32_t bits);
std::array<uint64_t, 2> msb_u128(const UInt256& value, uint32_t bits);
std::string to_hex(const UInt256& value, uint32_t bits);
std::array<uint8_t, 32> to_bytes_be(const UInt256& value);

} // namespace rm_model

#endif // RM_MODEL_UINT256_H
