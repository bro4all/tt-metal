// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

#include "debug/dprint.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);
    constexpr uint32_t input_tensor_num_pages = get_compile_time_arg_val(4);
    constexpr uint32_t tiles_per_slice_row = get_compile_time_arg_val(5);
    constexpr bool direction = get_compile_time_arg_val(6);

    binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
    add_tiles_init(input_cb_id, intermediate_cb, false);

    // TODO: (GR) Is this right?
    const uint32_t tile_granularity = 8;

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

            uint32_t cb_tile_id = 0;
            uint32_t tiles_per_slice_row_processed = 0;
            uint32_t tiles_per_slice_row_to_process = tiles_per_slice_row;

            // acquire_dst();
            // for (uint32_t tile_id = 0; tile_id < tiles_per_slice_row; tile_id++) {
            //     add_tiles(input_cb_id, intermediate_cb, tile_id, tile_id, tile_id);
            //     pack_tile(tile_id, output_cb);
            // }
            // release_dst();

            while (tiles_per_slice_row_processed < tiles_per_slice_row_to_process) {
                acquire_dst();
                for (uint32_t dst_tile_id = 0; dst_tile_id < tile_granularity; ++dst_tile_id) {
                    add_tiles(input_cb_id, intermediate_cb, cb_tile_id, cb_tile_id, dst_tile_id);
                    pack_tile(dst_tile_id, output_cb, cb_tile_id);

                    cb_tile_id++;
                    tiles_per_slice_row_processed++;
                    if (tiles_per_slice_row_processed == tiles_per_slice_row_to_process) {
                        break;
                    }
                }
                release_dst();
            }

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
