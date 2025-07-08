// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Get compile-time arguments
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t input_cb = get_compile_time_arg_val(1);

    // Get runtime arguments
    uint32_t input_addr = get_arg_val<uint32_t>(0);

    // Set up address generator for input buffer
    const uint32_t tile_bytes = get_tile_size(input_cb);
    const DataFormat data_format = get_dataformat(input_cb);

    const InterleavedAddrGenFast<true> input_addr_gen = {
        .bank_base_address = input_addr, .page_size = tile_bytes, .data_format = data_format};

    // Read tiles from DRAM and write to circular buffer
    for (uint32_t i = 0; i < Wt; i++) {
        // Reserve space in circular buffer
        cb_reserve_back(input_cb, 1);

        // Get write address in circular buffer
        uint32_t l1_write_addr = get_write_ptr(input_cb);

        // Read tile from DRAM
        noc_async_read_tile(i, input_addr_gen, l1_write_addr);
        noc_async_read_barrier();

        // Mark tile as ready
        cb_push_back(input_cb, 1);
    }
}
