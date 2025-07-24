// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);
    constexpr bool direction = get_compile_time_arg_val(4);

    binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
    add_tiles_init(input_cb_id, intermediate_cb, false);

    for (uint32_t i = 0; i < ring_size - 1; i++) {  // Don't reduce on the first slice
        uint32_t rows_read = 0;
        uint32_t rows_to_read = input_tensor_num_pages;

        // Skip first row if we're going backwards
        if constexpr (!direction) {
            rows_read++;
        }

        while (rows_read < rows_to_read) {
            cb_wait_front(input_cb_id, 1);
            cb_wait_front(intermediate_cb, 1);
            cb_reserve_back(output_cb, 1);
            acquire_dst();

            // TODO: (GR) This is assuming tiles_per_slice_row < tile_granularity
            for (uint32_t tile_id = 0; tile_id < tiles_per_slice_row; tile_id++) {
                add_tiles(input_cb_id, intermediate_cb, tile_id, tile_id, tile_id);
                pack_tile(tile_id, output_cb);
            }

            release_dst();
            cb_pop_front(input_cb_id, 1);
            cb_pop_front(intermediate_cb, 1);
            cb_push_back(output_cb, 1);

            // Increment rows_read by two
            // Once for the page we just read, and once for the row going the other direction
            rows_read += 2;
        }
    }
}
}  // namespace NAMESPACE
