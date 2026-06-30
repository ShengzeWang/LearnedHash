#include "rm_model/uint256.h"

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace rm_model {

UInt256 UInt256::zero() {
  return UInt256{};
}

UInt256 UInt256::from_u64(uint64_t value) {
  UInt256 out{};
  out.words[0] = value;
  return out;
}

UInt256 UInt256::one_shifted(uint32_t bits) {
  if (bits == 0) {
    return UInt256::zero();
  }
  if (bits > 256) {
    throw std::runtime_error("bits must be <= 256");
  }
  if (bits == 256) {
    return UInt256::zero();
  }

  UInt256 out{};
  uint32_t word = bits / 64;
  uint32_t rem = bits % 64;
  out.words[word] = 1ULL << rem;
  return out;
}

UInt256 UInt256::max_for_bits(uint32_t bits) {
  UInt256 out{};
  if (bits == 0) {
    return out;
  }
  if (bits >= 256) {
    out.words = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    return out;
  }

  uint32_t full_words = bits / 64;
  uint32_t rem_bits = bits % 64;
  for (uint32_t i = 0; i < full_words && i < 4; ++i) {
    out.words[i] = UINT64_MAX;
  }
  if (full_words < 4 && rem_bits != 0) {
    out.words[full_words] = (1ULL << rem_bits) - 1ULL;
  }
  return out;
}

bool UInt256::is_zero() const {
  return words[0] == 0 && words[1] == 0 && words[2] == 0 && words[3] == 0;
}

void UInt256::mask_bits(uint32_t bits) {
  if (bits >= 256) {
    return;
  }
  if (bits == 0) {
    words = {0, 0, 0, 0};
    return;
  }

  uint32_t full_words = bits / 64;
  uint32_t rem_bits = bits % 64;
  for (uint32_t i = full_words + (rem_bits ? 1U : 0U); i < 4; ++i) {
    words[i] = 0;
  }
  if (rem_bits != 0 && full_words < 4) {
    words[full_words] &= (1ULL << rem_bits) - 1ULL;
  }
}

UInt320 mul_u64_wide(const UInt256& value, uint64_t mul) {
  UInt320 out{};
#if defined(_MSC_VER)
  uint64_t carry = 0;
#else
  unsigned __int128 carry = 0;
#endif
  for (size_t i = 0; i < 4; ++i) {
#if defined(_MSC_VER)
    uint64_t hi = 0;
    uint64_t lo = _umul128(value.words[i], mul, &hi);
    uint64_t lo2 = lo + carry;
    if (lo2 < lo) {
      hi += 1;
    }
    out.words[i] = lo2;
    carry = hi;
#else
    unsigned __int128 prod = static_cast<unsigned __int128>(value.words[i]) * mul + carry;
    out.words[i] = static_cast<uint64_t>(prod);
    carry = prod >> 64;
#endif
  }
  out.words[4] = static_cast<uint64_t>(carry);
  return out;
}

UInt256 mul_u64(const UInt256& value, uint64_t mul, uint64_t* overflow) {
  UInt256 out{};
#if defined(_MSC_VER)
  uint64_t carry = 0;
#else
  unsigned __int128 carry = 0;
#endif
  for (size_t i = 0; i < 4; ++i) {
#if defined(_MSC_VER)
    uint64_t hi = 0;
    uint64_t lo = _umul128(value.words[i], mul, &hi);
    uint64_t lo2 = lo + carry;
    if (lo2 < lo) {
      hi += 1;
    }
    out.words[i] = lo2;
    carry = hi;
#else
    unsigned __int128 prod = static_cast<unsigned __int128>(value.words[i]) * mul + carry;
    out.words[i] = static_cast<uint64_t>(prod);
    carry = prod >> 64;
#endif
  }
  if (overflow) {
    *overflow = static_cast<uint64_t>(carry);
  }
  return out;
}

UInt256 div_u64(const UInt256& value, uint64_t divisor, uint64_t* remainder) {
  if (divisor == 0) {
    throw std::runtime_error("Division by zero");
  }

  UInt256 out{};
#if defined(_MSC_VER)
  uint64_t rem = 0;
#else
  unsigned __int128 rem = 0;
#endif
  for (int i = 3; i >= 0; --i) {
#if defined(_MSC_VER)
    uint64_t q = _udiv128(rem, value.words[static_cast<size_t>(i)], divisor, &rem);
    out.words[static_cast<size_t>(i)] = q;
#else
    unsigned __int128 cur = (rem << 64) | value.words[static_cast<size_t>(i)];
    uint64_t q = static_cast<uint64_t>(cur / divisor);
    rem = cur % divisor;
    out.words[static_cast<size_t>(i)] = q;
#endif
  }

  if (remainder) {
    *remainder = static_cast<uint64_t>(rem);
  }
  return out;
}

UInt256 div_u64(const UInt320& value, uint64_t divisor, uint64_t* remainder) {
  if (divisor == 0) {
    throw std::runtime_error("Division by zero");
  }

  UInt256 out{};
#if defined(_MSC_VER)
  uint64_t rem = 0;
#else
  unsigned __int128 rem = 0;
#endif
  for (int i = 4; i >= 0; --i) {
#if defined(_MSC_VER)
    uint64_t q = _udiv128(rem, value.words[static_cast<size_t>(i)], divisor, &rem);
#else
    unsigned __int128 cur = (rem << 64) | value.words[static_cast<size_t>(i)];
    uint64_t q = static_cast<uint64_t>(cur / divisor);
    rem = cur % divisor;
#endif
    if (i < 4) {
      out.words[static_cast<size_t>(i)] = q;
    }
  }

  if (remainder) {
    *remainder = static_cast<uint64_t>(rem);
  }
  return out;
}

UInt256 shr_320(const UInt320& value, uint32_t shift_bits) {
  if (shift_bits >= 64) {
    throw std::runtime_error("shift_bits must be < 64");
  }

  UInt256 out{};
  if (shift_bits == 0) {
    out.words[0] = value.words[0];
    out.words[1] = value.words[1];
    out.words[2] = value.words[2];
    out.words[3] = value.words[3];
    return out;
  }

  uint32_t inv = 64 - shift_bits;
  for (size_t i = 0; i < 4; ++i) {
    uint64_t low = value.words[i];
    uint64_t high = value.words[i + 1];
    out.words[i] = (low >> shift_bits) | (high << inv);
  }
  return out;
}

UInt256 shr_320_64(const UInt320& value) {
  UInt256 out{};
  out.words[0] = value.words[1];
  out.words[1] = value.words[2];
  out.words[2] = value.words[3];
  out.words[3] = value.words[4];
  return out;
}

UInt256 shr_256(const UInt256& value, uint32_t shift_bits) {
  if (shift_bits >= 256) {
    return UInt256::zero();
  }

  UInt256 out{};
  if (shift_bits == 0) {
    out.words = value.words;
    return out;
  }

  uint32_t word_shift = shift_bits / 64;
  uint32_t bit_shift = shift_bits % 64;
  for (size_t i = 0; i < 4; ++i) {
    size_t src = i + word_shift;
    uint64_t low = (src < 4) ? value.words[src] : 0;
    if (bit_shift == 0) {
      out.words[i] = low;
      continue;
    }
    uint64_t high = (src + 1 < 4) ? value.words[src + 1] : 0;
    out.words[i] = (low >> bit_shift) | (high << (64 - bit_shift));
  }
  return out;
}

UInt320 shl_u64_to_320(uint64_t value, uint32_t shift_bits) {
  if (shift_bits > 256) {
    throw std::runtime_error("shift_bits must be <= 256");
  }

  UInt320 out{};
  uint32_t word_shift = shift_bits / 64;
  uint32_t bit_shift = shift_bits % 64;
  if (word_shift >= 5) {
    return out;
  }
  out.words[word_shift] = value;
  if (bit_shift != 0) {
    out.words[word_shift] = value << bit_shift;
    if (word_shift + 1 < 5) {
      out.words[word_shift + 1] = value >> (64 - bit_shift);
    }
  }
  return out;
}

UInt256 add(const UInt256& a, const UInt256& b) {
  UInt256 out{};
#if defined(_MSC_VER)
  unsigned char carry = 0;
  carry = _addcarry_u64(carry, a.words[0], b.words[0], &out.words[0]);
  carry = _addcarry_u64(carry, a.words[1], b.words[1], &out.words[1]);
  carry = _addcarry_u64(carry, a.words[2], b.words[2], &out.words[2]);
  _addcarry_u64(carry, a.words[3], b.words[3], &out.words[3]);
#else
  unsigned __int128 sum = static_cast<unsigned __int128>(a.words[0]) + b.words[0];
  out.words[0] = static_cast<uint64_t>(sum);
  sum = static_cast<unsigned __int128>(a.words[1]) + b.words[1] + (sum >> 64);
  out.words[1] = static_cast<uint64_t>(sum);
  sum = static_cast<unsigned __int128>(a.words[2]) + b.words[2] + (sum >> 64);
  out.words[2] = static_cast<uint64_t>(sum);
  sum = static_cast<unsigned __int128>(a.words[3]) + b.words[3] + (sum >> 64);
  out.words[3] = static_cast<uint64_t>(sum);
#endif
  return out;
}

UInt256 add_u64(const UInt256& value, uint64_t addend) {
  UInt256 out = value;
#if defined(_MSC_VER)
  unsigned char carry = 0;
  carry = _addcarry_u64(carry, out.words[0], addend, &out.words[0]);
  if (carry) {
    carry = _addcarry_u64(carry, out.words[1], 0, &out.words[1]);
  }
  if (carry) {
    carry = _addcarry_u64(carry, out.words[2], 0, &out.words[2]);
  }
  if (carry) {
    _addcarry_u64(carry, out.words[3], 0, &out.words[3]);
  }
#else
  unsigned __int128 sum = static_cast<unsigned __int128>(out.words[0]) + addend;
  out.words[0] = static_cast<uint64_t>(sum);
  uint64_t carry = static_cast<uint64_t>(sum >> 64);
  for (size_t i = 1; i < 4 && carry; ++i) {
    unsigned __int128 s = static_cast<unsigned __int128>(out.words[i]) + carry;
    out.words[i] = static_cast<uint64_t>(s);
    carry = static_cast<uint64_t>(s >> 64);
  }
#endif
  return out;
}

UInt256 sub(const UInt256& a, const UInt256& b) {
  UInt256 out{};
#if defined(_MSC_VER)
  unsigned char borrow = 0;
  borrow = _subborrow_u64(borrow, a.words[0], b.words[0], &out.words[0]);
  borrow = _subborrow_u64(borrow, a.words[1], b.words[1], &out.words[1]);
  borrow = _subborrow_u64(borrow, a.words[2], b.words[2], &out.words[2]);
  _subborrow_u64(borrow, a.words[3], b.words[3], &out.words[3]);
#else
  unsigned __int128 diff = static_cast<unsigned __int128>(a.words[0]) - b.words[0];
  out.words[0] = static_cast<uint64_t>(diff);
  diff = static_cast<unsigned __int128>(a.words[1]) - b.words[1] - ((diff >> 127) & 1);
  out.words[1] = static_cast<uint64_t>(diff);
  diff = static_cast<unsigned __int128>(a.words[2]) - b.words[2] - ((diff >> 127) & 1);
  out.words[2] = static_cast<uint64_t>(diff);
  diff = static_cast<unsigned __int128>(a.words[3]) - b.words[3] - ((diff >> 127) & 1);
  out.words[3] = static_cast<uint64_t>(diff);
#endif
  return out;
}

int compare(const UInt256& a, const UInt256& b) {
  for (int i = 3; i >= 0; --i) {
    if (a.words[static_cast<size_t>(i)] < b.words[static_cast<size_t>(i)]) {
      return -1;
    }
    if (a.words[static_cast<size_t>(i)] > b.words[static_cast<size_t>(i)]) {
      return 1;
    }
  }
  return 0;
}

uint64_t to_u64(const UInt256& value, uint32_t bits) {
  if (bits == 0) {
    return 0;
  }
  if (bits >= 64) {
    return value.words[0];
  }
  return value.words[0] & ((1ULL << bits) - 1ULL);
}

uint64_t msb_u64(const UInt256& value, uint32_t bits) {
  if (bits == 0) {
    return 0;
  }
  if (bits <= 64) {
    return bits == 64 ? value.words[0] : (value.words[0] & ((1ULL << bits) - 1ULL));
  }

  UInt256 shifted = shr_256(value, bits - 64);
  return shifted.words[0];
}

std::array<uint64_t, 2> msb_u128(const UInt256& value, uint32_t bits) {
  if (bits == 0) {
    return {0, 0};
  }
  if (bits <= 128) {
    return {value.words[0], value.words[1]};
  }

  UInt256 shifted = shr_256(value, bits - 128);
  return {shifted.words[0], shifted.words[1]};
}

std::string to_hex(const UInt256& value, uint32_t bits) {
  if (bits == 0) {
    return "0";
  }

  uint32_t full_words = bits / 64;
  uint32_t rem_bits = bits % 64;
  uint32_t total_words = full_words + (rem_bits ? 1U : 0U);
  if (total_words == 0) {
    total_words = 1;
  }

  std::ostringstream oss;
  oss << std::hex << std::nouppercase << std::setfill('0');
  for (int idx = static_cast<int>(total_words) - 1; idx >= 0; --idx) {
    uint64_t word = value.words[static_cast<size_t>(idx)];
    if (idx == static_cast<int>(total_words) - 1 && rem_bits != 0) {
      word &= (1ULL << rem_bits) - 1ULL;
      uint32_t width = (rem_bits + 3) / 4;
      oss << std::setw(static_cast<int>(width)) << word;
    } else {
      oss << std::setw(16) << word;
    }
  }

  std::string out = oss.str();
  return out.empty() ? "0" : out;
}

std::array<uint8_t, 32> to_bytes_be(const UInt256& value) {
  std::array<uint8_t, 32> out{};
  for (size_t i = 0; i < 4; ++i) {
    uint64_t word = value.words[3 - i];
    for (size_t b = 0; b < 8; ++b) {
      out[i * 8 + b] = static_cast<uint8_t>((word >> (56 - b * 8)) & 0xFF);
    }
  }
  return out;
}

} // namespace rm_model
