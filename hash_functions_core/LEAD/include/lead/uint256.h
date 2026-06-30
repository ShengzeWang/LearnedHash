#ifndef LEAD_UINT256_H
#define LEAD_UINT256_H

#include "rm_model/uint256.h"

namespace lead {

using rm_model::UInt256;
using rm_model::UInt320;
using rm_model::div_u64;
using rm_model::msb_u128;
using rm_model::msb_u64;
using rm_model::mul_u64;
using rm_model::mul_u64_wide;
using rm_model::shr_256;
using rm_model::shr_320;
using rm_model::shr_320_64;
using rm_model::to_bytes_be;
using rm_model::to_hex;

} // namespace lead

#endif // LEAD_UINT256_H
