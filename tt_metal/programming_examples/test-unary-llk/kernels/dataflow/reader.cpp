// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // Get compile-time arguments
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t input0_cb = get_compile_time_arg_val(1);
    constexpr uint32_t input1_cb = get_compile_time_arg_val(2);

    // Get runtime arguments
    uint32_t input0_addr = get_arg_val<uint32_t>(0);
    uint32_t input1_addr = get_arg_val<uint32_t>(1);

    // Set up address generator for input buffer
    const uint32_t input0_tile_bytes = get_tile_size(input0_cb);
    const DataFormat input0_data_format = get_dataformat(input0_cb);

    const uint32_t input1_tile_bytes = get_tile_size(input1_cb);
    const DataFormat input1_data_format = get_dataformat(input1_cb);

    const InterleavedAddrGenFast<true> input0_addr_gen = {
        .bank_base_address = input0_addr, .page_size = input0_tile_bytes, .data_format = input0_data_format};

    const InterleavedAddrGenFast<true> input1_addr_gen = {
        .bank_base_address = input1_addr, .page_size = input1_tile_bytes, .data_format = input1_data_format};

    // Read tiles from DRAM and write to circular buffer
    for (uint32_t i = 0; i < Wt; i++) {
        // Reserve space in circular buffer
        cb_reserve_back(input0_cb, 1);
        cb_reserve_back(input1_cb, 1);

        // Get write address in circular buffer
        uint32_t input0_l1_write_addr = get_write_ptr(input0_cb);
        uint32_t input1_l1_write_addr = get_write_ptr(input1_cb);

        // Read tile from DRAM
        noc_async_read_tile(i, input0_addr_gen, input0_l1_write_addr);
        noc_async_read_tile(i, input1_addr_gen, input1_l1_write_addr);
        noc_async_read_barrier();

        // Mark tile as ready
        cb_push_back(input0_cb, 1);
        cb_push_back(input1_cb, 1);
    }
}
