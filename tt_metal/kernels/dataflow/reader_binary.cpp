// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT << (uint32_t)r << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

inline void print_cb_details(uint32_t cb_id) {
    DPRINT <<
        "cb_id " << cb_id << ": { "
                 << "size: " << get_local_cb_interface(cb_id).fifo_size << ", "
                 << "limit: " << get_local_cb_interface(cb_id).fifo_limit << ", "
                 << "page_size: " << get_local_cb_interface(cb_id).fifo_page_size << ", "
                 << "num_pages: " << get_local_cb_interface(cb_id).fifo_num_pages << ", "
                 << "rd_ptr: " << get_local_cb_interface(cb_id).fifo_rd_ptr << ", "
                 << "wr_ptr: " << get_local_cb_interface(cb_id).fifo_wr_ptr << ", "
                 << "wr_tile_ptr: " << get_local_cb_interface(cb_id).fifo_wr_tile_ptr << " }" << ENDL();
}

void kernel_main() {
/*
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);
    uint32_t ublock_size_tiles = 1;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    DPRINT << "src0_addr " << src0_addr << " src0_bank_id " << src0_bank_id << " src1_addr " << src1_addr << " src1_bank_id " << src1_bank_id << ENDL();
    print_cb_details(cb_id_in0);

    // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);

        cb_reserve_back(cb_id_in0, ublock_size_tiles);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);

	DPRINT << "src0_noc_addr " << src0_noc_addr << " l1_write_addr_in0 " << l1_write_addr_in0 << " ublocksize_bytes_0 " << ublock_size_bytes_0 << ENDL();
        noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, ublock_size_tiles);

        src0_addr += ublock_size_bytes_0;

        print_full_tile(cb_id_in0);

        uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_addr);

        cb_reserve_back(cb_id_in1, ublock_size_tiles);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);

        noc_async_read_barrier();

        cb_push_back(cb_id_in1, ublock_size_tiles);

        src1_addr += ublock_size_bytes_1;
    }
    */
}
