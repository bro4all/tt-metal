// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "accessor/tensor_accessor.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t src2_addr = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);

    const auto s0 = TensorAccessor(src0_args, src0_addr, src0_tile_bytes);
    const auto s1 = TensorAccessor(src1_args, src1_addr, src1_tile_bytes);

    constexpr uint32_t onetile = 1;

    for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(tile_id, s0, l1_write_addr_in0);

        cb_reserve_back(cb_id_in1, onetile);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        noc_async_read_tile(tile_id, s1, l1_write_addr_in1);

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, onetile);
        cb_push_back(cb_id_in1, onetile);
    }
}
