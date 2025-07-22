// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpi.h"
#include "adam_cb_config.hpp"

// void WORK_PARTITION(uint32_t& start_id, uint32_t& num_unit) {
//         const auto upcg1 = get_common_arg_val<uint32_t>(0);
//         const auto upcg2 = get_common_arg_val<uint32_t>(0);
//         const auto g1_cores = get_common_arg_val<uint32_t>(0);
//         const auto g2_cores = get_common_arg_val<uint32_t>(0);
//         const auto grid_y = get_common_arg_val<uint32_t>(0);

//         uint32_t x = get_relative_logical_x();
//         uint32_t y = get_relative_logical_y();
//         uint32_t core_linear_id = x * grid_y + y;
//         uint32_t total_cores = g1_cores + g2_cores;

//         if (core_linear_id >= total_cores) return;

//         if (core_linear_id < g1_cores) {
//                 num_unit = upcg1;
//                 start_id = core_linear_id * upcg1;
//         } else {
//                 num_unit = upcg2;
//                 start_id = g1_cores * upcg1 + (core_linear_id - g1_cores) * upcg2;
//         }

//         return;
// }

union Scalar {
    float f;
    uint32_t u;
};

ALWI float to_float(uint32_t bits) {
    Scalar u2f{.u = bits};
    return u2f.f;
}

ALWI uint32_t to_bits(float f) {
    Scalar f2u{.f = f};
    return f2u.u;
}

using namespace sfpi;

enum {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    RSUB = 3,
};

ALWI void pack_onetile_to_cb(uint32_t icb = 16, uint32_t ifrom_dst = 0) {
    tile_regs_wait();
    cb_reserve_back(icb, 1);
    pack_tile_with_dt(ifrom_dst, icb);
    cb_push_back(icb, 1);
    tile_regs_release();
}

template <bool APPROXIMATION_MODE, int BINOP_MODE, int ITERATIONS = 8>
inline void calculate_binop_fp32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 64);
        if constexpr (BINOP_MODE == ADD) {
            TTI_SFPADD(
                p_sfpu::LREG0,
                p_sfpu::LCONST_1,
                p_sfpu::LREG1,
                p_sfpu::LREG1,
                0);      // LREG[1] = LREG[0] * 1 + LREG[1]
            TTI_SFPNOP;  // SFPADD is two-cycle instruction
        } else if constexpr (BINOP_MODE == SUB) {
            TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG1, 1);  // LREG[1] = -LREG[1]
            TTI_SFPADD(
                p_sfpu::LREG0,
                p_sfpu::LCONST_1,
                p_sfpu::LREG1,
                p_sfpu::LREG1,
                0);      // LREG[1] = LREG[0] * 1 + LREG[1]
            TTI_SFPNOP;  // SFPADD is two-cycle instruction
        } else if constexpr (BINOP_MODE == MUL) {
            TTI_SFPMUL(
                p_sfpu::LREG0,
                p_sfpu::LREG1,
                p_sfpu::LCONST_0,
                p_sfpu::LREG1,
                0);      // LREG[1] = LREG[0] * LREG[1] + 0
            TTI_SFPNOP;  // SFPMUL is two-cycle instruction
        }
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int UNARY_MODE, int ITERATIONS = 8>
inline void calculate_binop_with_scalar_fp32(uint param) {
    if constexpr (UNARY_MODE == SUB) {
        param ^= 0x80000000u;
    }
    TT_SFPLOADI(p_sfpu::LREG0, 10, param & 0xFFFF);  // load lower 16 bits
    TT_SFPLOADI(p_sfpu::LREG0, 8, param >> 16);      // load upper 16 bits

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        if constexpr (UNARY_MODE == ADD || UNARY_MODE == SUB) {
            TTI_SFPADD(
                p_sfpu::LREG0,
                p_sfpu::LCONST_1,
                p_sfpu::LREG1,
                p_sfpu::LREG1,
                0);   // LREG[1] = LREG[0] * 1 + LREG[1]
            TTI_NOP;  // SFPADD is two-cycle instruction
        } else if constexpr (UNARY_MODE == RSUB) {
            TTI_SFPADD(
                p_sfpu::LREG1,
                p_sfpu::LCONST_neg1,
                p_sfpu::LREG0,
                p_sfpu::LREG1,
                0);   // LREG[1] = LREG[0] * -1 + LREG[1]
            TTI_NOP;  // SFPADD is two-cycle instruction
        } else if constexpr (UNARY_MODE == MUL) {
            TTI_SFPMUL(
                p_sfpu::LREG0,
                p_sfpu::LREG1,
                p_sfpu::LCONST_0,
                p_sfpu::LREG1,
                0);   // LREG[1] = LREG[0] * LREG[1] + 0
            TTI_NOP;  // SFPMUL is two-cycle instruction
        }
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        dst_reg++;
    }
}

inline void init_sfpu_default() { llk_math_eltwise_unary_sfpu_init<SfpuType::unused, false>(); }

template <bool APPROXIMATE, int BINOP_MODE>
inline void binary_fp32(uint dst_index, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        calculate_binop_fp32<APPROXIMATE, BINOP_MODE>, dst_index, vector_mode);
}

template <bool APPROXIMATE, int UNARY_MODE>
inline void unary_fp32(uint dst_index, uint32_t param1, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        calculate_binop_with_scalar_fp32<APPROXIMATE, UNARY_MODE>, dst_index, vector_mode, param1);
}

namespace NAMESPACE {
void MAIN {
    namespace cb = cb_adam;

    // Compile Time Arguments
    constexpr bool amsgrad = get_compile_time_arg_val(0) == 1;

    // Common Runtime Arguments
    uint32_t num_tiles = 0;
    // WORK_PARTITION(tile_offset, num_tiles);
    const auto upcg1 = get_common_arg_val<uint32_t>(0);
    const auto upcg2 = get_common_arg_val<uint32_t>(1);
    const auto g1_cores = get_common_arg_val<uint32_t>(2);
    const auto g2_cores = get_common_arg_val<uint32_t>(3);
    const auto grid_y = get_common_arg_val<uint32_t>(4);

    uint32_t x = get_relative_logical_x();
    uint32_t y = get_relative_logical_y();
    uint32_t core_linear_id = x * grid_y + y;
    uint32_t total_cores = g1_cores + g2_cores;

    if (core_linear_id >= total_cores) {
        return;
    }

    if (core_linear_id < g1_cores) {
        num_tiles = upcg1;
    } else {
        num_tiles = upcg2;
    }

    const auto step = get_common_arg_val<uint32_t>(5);
    const auto lr_bits = get_common_arg_val<uint32_t>(6);
    const auto beta1_bits = get_common_arg_val<uint32_t>(7);
    const auto beta2_bits = get_common_arg_val<uint32_t>(8);
    const auto eps_bits = get_common_arg_val<uint32_t>(9);
    const auto weight_decay_bits = get_common_arg_val<uint32_t>(10);
    const auto bias_correction1_bits = get_common_arg_val<uint32_t>(11);
    const auto bias_correction2_bits = get_common_arg_val<uint32_t>(12);

    const float lr = to_float(lr_bits);
    const float beta1 = to_float(beta1_bits);
    const float beta2 = to_float(beta2_bits);
    const float eps = to_float(eps_bits);
    const float weight_decay = to_float(weight_decay_bits);
    const float bias_correction1 = to_float(bias_correction1_bits);
    const float bias_correction2 = to_float(bias_correction2_bits);

    constexpr uint32_t onetile = 1;

    // Variables
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t dst2 = 2;

    binary_op_init_common(cb::param_in, cb::grad_i, cb::param_out);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb::param_in, onetile);
        cb_wait_front(cb::grad, onetile);
        cb_wait_front(cb::exp_avg_in, onetile);
        cb_wait_front(cb::exp_avg_sq_in, onetile);
        if constexpr (amsgrad) {
            cb_wait_front(cb::max_exp_avg_sq_in, onetile);
        }

        // grad_i = grad + param_in * weight_decay;
        tile_regs_acquire();

        copy_tile_init_with_dt(cb::grad);
        copy_tile(cb::grad, 0, dst0);
        copy_tile_init_with_dt(cb::param_in);
        copy_tile(cb::param_in, 0, dst1);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst1, weight_decay_bits);
        init_sfpu_default();
        binary_fp32<false, ADD>(dst0);
        tile_regs_commit();

        pack_onetile_to_cb(cb::grad_i, dst0);

        // exp_avg_out = exp_avg_i = exp_avg_in * beta1 + grad_i * (1 - beta1);
        tile_regs_acquire();
        cb_wait_front(cb::grad_i, onetile);
        cb_reserve_back(cb::exp_avg_i, onetile);
        cb_reserve_back(cb::exp_avg_out, onetile);
        copy_tile_init_with_dt(cb::exp_avg_in);
        copy_tile(cb::exp_avg_in, 0, dst0);
        copy_tile_init_with_dt(cb::grad_i);
        copy_tile(cb::grad_i, 0, dst1);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst0, beta1_bits);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst1, to_bits(1 - beta1));
        init_sfpu_default();
        binary_fp32<false, ADD>(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb::exp_avg_i);
        pack_tile_with_dt(dst0, cb::exp_avg_out);
        cb_push_back(cb::exp_avg_i, onetile);
        cb_push_back(cb::exp_avg_out, onetile);
        tile_regs_release();

        // exp_avg_sq_out = exp_avg_sq_i = exp_avg_sq_in * beta2 + grad_i^2 * (1 -
        // beta2);
        tile_regs_acquire();
        cb_reserve_back(cb::exp_avg_sq_i, onetile);
        cb_reserve_back(cb::exp_avg_sq_out, onetile);
        copy_tile_init_with_dt(cb::exp_avg_sq_in);
        copy_tile(cb::exp_avg_sq_in, 0, dst0);
        copy_tile_init_with_dt(cb::grad_i);
        copy_tile(cb::grad_i, 0, dst1);
        copy_tile_init_with_dt(cb::grad_i);
        copy_tile(cb::grad_i, 0, dst2);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst0, beta2_bits);
        init_sfpu_default();
        binary_fp32<false, MUL>(dst1);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst1, to_bits(1.0f - beta2));
        init_sfpu_default();
        binary_fp32<false, ADD>(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb::exp_avg_sq_i);
        pack_tile_with_dt(dst0, cb::exp_avg_sq_out);
        cb_push_back(cb::exp_avg_sq_i, onetile);
        cb_push_back(cb::exp_avg_sq_out, onetile);
        cb_pop_front(cb::grad_i, onetile);
        tile_regs_release();

        if constexpr (amsgrad) {
            // max_exp_avg_sq_out = max_exp_avg_sq_i = max(max_exp_avg_sq_in,
            // exp_avg_sq_i)
            tile_regs_acquire();
            cb_wait_front(cb::exp_avg_sq_i, onetile);
            cb_reserve_back(cb::max_exp_avg_sq_i, onetile);
            cb_reserve_back(cb::max_exp_avg_sq_out, onetile);
            copy_tile_init_with_dt(cb::max_exp_avg_sq_in);
            copy_tile(cb::max_exp_avg_sq_in, 0, dst0);
            copy_tile_init_with_dt(cb::exp_avg_sq_i);
            copy_tile(cb::exp_avg_sq_i, 0, dst1);
            max_tile_init();
            max_tile(dst0, dst1);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb::max_exp_avg_sq_i);
            pack_tile_with_dt(dst0, cb::max_exp_avg_sq_out);
            cb_push_back(cb::max_exp_avg_sq_i, onetile);
            cb_push_back(cb::max_exp_avg_sq_out, onetile);
            cb_pop_front(cb::exp_avg_sq_i, onetile);
            tile_regs_release();
        }

        // if amsgrad:
        //     param_out = param_in - lr * (exp_avg_i / bias_corr1) /
        //     (sqrt(max_exp_avg_sq_i / bias_corr2) + eps)
        // else:
        //     param_out = param_in - lr * (exp_avg_i / bias_corr1) /
        //     (sqrt(exp_avg_sq_i / bias_corr2) + eps)
        tile_regs_acquire();
        cb_wait_front(cb::exp_avg_i, onetile);
        if constexpr (amsgrad) {
            cb_wait_front(cb::max_exp_avg_sq_i, onetile);
        } else {
            cb_wait_front(cb::exp_avg_sq_i, onetile);
        }

        copy_tile_init_with_dt(cb::param_in);
        copy_tile(cb::param_in, 0, dst0);
        copy_tile_init_with_dt(cb::exp_avg_i);
        copy_tile(cb::exp_avg_i, 0, dst1);
        if constexpr (amsgrad) {
            copy_tile_init_with_dt(cb::max_exp_avg_sq_i);
            copy_tile(cb::max_exp_avg_sq_i, 0, dst2);
        } else {
            copy_tile_init_with_dt(cb::exp_avg_sq_i);
            copy_tile(cb::exp_avg_sq_i, 0, dst2);
        }
        binop_with_scalar_tile_init();
        unary_fp32<false, MUL>(dst1, to_bits(lr / bias_correction1));
        binop_with_scalar_tile_init();
        unary_fp32<false, MUL>(dst2, to_bits(1.f / bias_correction2));
        sqrt_tile_init();
        sqrt_tile(dst2);
        binop_with_scalar_tile_init();
        unary_fp32<false, ADD>(dst2, eps_bits);
        recip_tile_init();
        recip_tile(dst2);
        init_sfpu_default();
        binary_fp32<false, MUL>(dst1);
        init_sfpu_default();
        binary_fp32<false, SUB>(dst0);
        tile_regs_commit();

        cb_pop_front(cb::exp_avg_i, onetile);
        if constexpr (amsgrad) {
            cb_pop_front(cb::max_exp_avg_sq_i, onetile);
        } else {
            cb_pop_front(cb::exp_avg_sq_i, onetile);
        }
        pack_onetile_to_cb(cb::param_out, dst0);
        cb_pop_front(cb::param_in, onetile);
        cb_pop_front(cb::grad, onetile);
        cb_pop_front(cb::exp_avg_in, onetile);
        cb_pop_front(cb::exp_avg_sq_in, onetile);
        if constexpr (amsgrad) {
            cb_pop_front(cb::max_exp_avg_sq_in, onetile);
        }
    }
}  // void MAIN
}  // namespace NAMESPACE
