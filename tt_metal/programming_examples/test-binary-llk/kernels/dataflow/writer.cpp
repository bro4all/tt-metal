// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Get compile-time arguments
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb = get_compile_time_arg_val(1);

    // Get runtime arguments
    uint32_t output_addr = get_arg_val<uint32_t>(0);

    // Set up address generator for output buffer
    const uint32_t tile_bytes = get_tile_size(output_cb);
    const DataFormat data_format = get_dataformat(output_cb);

    const InterleavedAddrGenFast<true> output_addr_gen = {
        .bank_base_address = output_addr, .page_size = tile_bytes, .data_format = data_format};

    // Read tiles from circular buffer and write to DRAM
    for (uint32_t i = 0; i < Wt; i++) {
        // Wait for tile to be available in circular buffer
        cb_wait_front(output_cb, 1);

        // Get read address from circular buffer
        uint32_t l1_read_addr = get_read_ptr(output_cb);

        // Write tile to DRAM
        noc_async_write_tile(i, output_addr_gen, l1_read_addr);
        noc_async_write_barrier();

        // Mark tile as consumed
        cb_pop_front(output_cb, 1);
    }
}
