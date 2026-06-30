#ifndef VORTEX_V1_UINT256_H
#define VORTEX_V1_UINT256_H

#include "rm_model/uint256.h"

namespace vortex {

using rm_model::UInt256;
using rm_model::UInt320;
using rm_model::add;
using rm_model::add_u64;
using rm_model::compare;
using rm_model::div_u64;
using rm_model::mul_u64;
using rm_model::mul_u64_wide;
using rm_model::shl_u64_to_320;
using rm_model::shr_320_64;
using rm_model::sub;
using rm_model::to_hex;
using rm_model::to_u64;

} // namespace vortex

#endif // VORTEX_V1_UINT256_H
