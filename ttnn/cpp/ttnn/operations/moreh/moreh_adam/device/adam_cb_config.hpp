#pragma once

#include "hostdevcommon/kernel_structs.h"

namespace cb_adam {
using namespace tt;
inline constexpr auto param_in = CBIndex::c_0;
inline constexpr auto grad = CBIndex::c_1;
inline constexpr auto exp_avg_in = CBIndex::c_2;
inline constexpr auto exp_avg_sq_in = CBIndex::c_3;
inline constexpr auto max_exp_avg_sq_in = CBIndex::c_4;
inline constexpr auto param_out = CBIndex::c_5;
inline constexpr auto exp_avg_out = CBIndex::c_6;
inline constexpr auto exp_avg_sq_out = CBIndex::c_7;
inline constexpr auto max_exp_avg_sq_out = CBIndex::c_8;
inline constexpr auto mask_w = CBIndex::c_9;
inline constexpr auto grad_i = CBIndex::c_10;
inline constexpr auto exp_avg_i = CBIndex::c_11;
inline constexpr auto exp_avg_sq_i = CBIndex::c_12;
inline constexpr auto max_exp_avg_sq_i = CBIndex::c_13;
}  // namespace cb_adam
