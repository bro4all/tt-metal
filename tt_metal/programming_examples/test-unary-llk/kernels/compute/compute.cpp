// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/identity.h"

namespace NAMESPACE {
void MAIN {
    // Get compile-time arguments
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t input_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);

    // Initialize the compute kernel
    init_sfpu(input_cb, output_cb);

    // Initialize the identity kernel
    identity_tile_init();

    constexpr uint32_t TILE0 = 0;
    constexpr uint32_t FIRST_TILE = 0;

    // Process each tile
    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx++) {
        // Wait for registers to be available
        tile_regs_acquire();

        // Wait for input tile to be available
        cb_wait_front(input_cb, 1);

        // Copy tile from circular buffer to registers
        copy_tile(input_cb, FIRST_TILE, TILE0);
        identity_tile(TILE0);

        // Apply identity operation (in this case, no-op since we're just copying)
        // For identity, we can simply skip any SFPU operation

        // Commit the tile computation
        tile_regs_commit();
        tile_regs_wait();

        // Reserve space in output buffer
        cb_reserve_back(output_cb, 1);

        // Pack the result into output buffer
        pack_tile(TILE0, output_cb, FIRST_TILE);

        // Pop the input tile and push the output tile
        cb_pop_front(input_cb, 1);

        // Mark output tile as ready
        cb_push_back(output_cb, 1);

        // Release registers
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
