#include "lead/mapping.h"

#include "lead/uint256.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace lead {

namespace {

uint32_t clamp_space_bits(uint32_t bits) {
  if (bits == 0 || bits > 256) {
    throw std::runtime_error("space_bits must be in [1, 256]");
  }
  return bits;
}

uint64_t scale_factor_fixed(double scale_factor, std::string* warning) {
  if (!(scale_factor > 0.0)) {
    throw std::runtime_error("scale_factor must be > 0");
  }

  if (scale_factor > 1.0) {
    if (warning) {
      *warning = "scale_factor > 1.0 is clamped to 1.0";
    }
    scale_factor = 1.0;
  }

  constexpr uint64_t kFixedDen = 1ULL << 32;
  double scaled = scale_factor * static_cast<double>(kFixedDen);
  uint64_t fixed = static_cast<uint64_t>(std::llround(scaled));
  if (fixed == 0) {
    fixed = 1;
  }
  if (fixed > kFixedDen) {
    fixed = kFixedDen;
  }
  return fixed;
}

uint64_t scaled_range_u64(uint32_t space_bits, double scale_factor) {
  uint64_t max_range = (space_bits == 64)
      ? UINT64_MAX
      : ((space_bits == 0) ? 0ULL : ((1ULL << space_bits) - 1ULL));
  uint64_t fixed = scale_factor_fixed(scale_factor, nullptr);
#if defined(_MSC_VER)
  uint64_t hi = 0;
  uint64_t lo = _umul128(max_range, fixed, &hi);
  uint64_t upper = hi;
  uint64_t lower = lo >> 32;
  return (upper << 32) | lower;
#else
  unsigned __int128 prod = static_cast<unsigned __int128>(max_range) * fixed;
  return static_cast<uint64_t>(prod >> 32);
#endif
}

uint64_t mul_div_u64(uint64_t a, uint64_t b, uint64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
#if defined(_MSC_VER)
  uint64_t hi = 0;
  uint64_t lo = _umul128(a, b, &hi);
  uint64_t rem = 0;
  return _udiv128(hi, lo, divisor, &rem);
#else
  unsigned __int128 prod = static_cast<unsigned __int128>(a) * b;
  return static_cast<uint64_t>(prod / divisor);
#endif
}

uint64_t fraction_to_fixed(double fraction) {
  if (!(fraction > 0.0)) {
    return 0;
  }
  if (fraction >= 1.0) {
    return UINT64_MAX;
  }
  double scaled = fraction * std::ldexp(1.0, 64);
  double max_u64 = static_cast<double>(std::numeric_limits<uint64_t>::max());
  if (scaled >= max_u64) {
    return UINT64_MAX;
  }
  uint64_t fixed = static_cast<uint64_t>(scaled);
  if (fixed == 0) {
    fixed = 1;
  }
  return fixed;
}

} // namespace

MappingConfig MappingConfig::compute(uint64_t max_index,
                                     uint32_t space_bits,
                                     double scale_factor,
                                     std::string* warning) {
  MappingConfig cfg;
  cfg.max_index = max_index;
  cfg.space_bits = clamp_space_bits(space_bits);

  uint64_t fixed = scale_factor_fixed(scale_factor, warning);
  cfg.scale_factor = static_cast<double>(fixed) / static_cast<double>(1ULL << 32);
  UInt256 max_range = UInt256::max_for_bits(cfg.space_bits);
  UInt320 scaled_wide = mul_u64_wide(max_range, fixed);
  UInt256 scaled_range = shr_320(scaled_wide, 32);

  if (fixed == (1ULL << 32)) {
    scaled_range = max_range;
  }

  UInt256 scale{};
  if (cfg.max_index == 0) {
    scale = UInt256::zero();
  } else {
    uint64_t rem = 0;
    scale = div_u64(scaled_range, cfg.max_index, &rem);
  }

  cfg.scale_words = scale.words;
  return cfg;
}

IndexMapper::IndexMapper(MappingConfig config)
    : config_(std::move(config)) {
  if (config_.space_bits <= 64) {
    scaled_range_u64_ = scaled_range_u64(config_.space_bits, config_.scale_factor);
    return;
  }

  uint64_t fixed = scale_factor_fixed(config_.scale_factor, nullptr);
  UInt256 max_range = UInt256::max_for_bits(config_.space_bits);
  UInt320 scaled_wide = mul_u64_wide(max_range, fixed);
  UInt256 scaled_range = shr_320(scaled_wide, 32);
  if (fixed == (1ULL << 32)) {
    scaled_range = max_range;
  }
  scaled_range_ = scaled_range;
}

HashOutput IndexMapper::map(uint64_t index) const {
  uint64_t idx = std::min(index, config_.max_index);

  if (config_.space_bits <= 64) {
    uint64_t out = 0;
    uint64_t scale_low = config_.scale_words[0];
    if (scale_low != 0) {
      out = idx * scale_low;
    } else if (config_.max_index != 0) {
      uint64_t scaled_range = scaled_range_u64(config_.space_bits, config_.scale_factor);
      if (scaled_range != 0) {
        out = mul_div_u64(idx, scaled_range, config_.max_index);
      }
    }
    HashOutput result;
    result.value.words = {out, 0, 0, 0};
    result.bits = config_.space_bits;
    return result;
  }

  UInt256 scale{};
  scale.words = config_.scale_words;

  uint64_t overflow = 0;
  UInt256 out = mul_u64(scale, idx, &overflow);
  out.mask_bits(config_.space_bits);

  HashOutput result;
  result.value = out;
  result.bits = config_.space_bits;
  return result;
}

HashOutput IndexMapper::map_double(double prediction) const {
  HashOutput result;
  result.bits = config_.space_bits;

  if (config_.max_index == 0) {
    result.value = UInt256::zero();
    return result;
  }

  double pred = prediction;
  if (!std::isfinite(pred) || pred <= 0.0) {
    result.value = UInt256::zero();
    return result;
  }

  double max_index_d = static_cast<double>(config_.max_index);
  if (pred >= max_index_d) {
    if (config_.space_bits <= 64) {
      result.value.words = {scaled_range_u64_, 0, 0, 0};
    } else {
      result.value = scaled_range_;
    }
    return result;
  }

  double fraction = pred / max_index_d;
  uint64_t fixed = fraction_to_fixed(fraction);
  if (fixed == 0) {
    result.value = UInt256::zero();
    return result;
  }

  if (config_.space_bits <= 64) {
#if defined(_MSC_VER)
    uint64_t hi = 0;
    _umul128(scaled_range_u64_, fixed, &hi);
    uint64_t out = hi;
#else
    unsigned __int128 prod = static_cast<unsigned __int128>(scaled_range_u64_) * fixed;
    uint64_t out = static_cast<uint64_t>(prod >> 64);
#endif
    result.value.words = {out, 0, 0, 0};
    return result;
  }

  UInt320 prod = mul_u64_wide(scaled_range_, fixed);
  UInt256 out = shr_320_64(prod);
  out.mask_bits(config_.space_bits);
  result.value = out;
  return result;
}

} // namespace lead
