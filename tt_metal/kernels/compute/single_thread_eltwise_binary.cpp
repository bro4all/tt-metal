// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary_st.h"
#include "dataflow_api.h"

#include "debug/dprint.h"
#include "debug/dprint_tensix.h"
#include "debug/dprint_pages.h"

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

namespace NAMESPACE {
void MAIN {
    // Args for reading data from DRAM
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_bank_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);

    // Args for computing the results
    // How many blocks of tiles to work on
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(5);
    // How many tiles per block
    uint32_t per_core_block_size = get_arg_val<uint32_t>(6);

    // Input and output circular buffer ids.
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_in1);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    // Initialize the parts that are common among binary operations
    binary_op_init_common_st(cb_in0, cb_in1, cb_out0);

    // Initialize the parts that required specifically for this binary operatoins
    binary_tiles_init_st<false, EltwiseBinaryType::ELWADD>(cb_in0, cb_in1);

    UNPACK(
    DPRINT << "NOC INDEX" << noc_index << ENDL();
    DPRINT << "src0_addr " << src0_addr << " src0_bank_id " << src0_bank_id << " src1_addr " << src1_addr << " src1_bank_id " << src1_bank_id << ENDL();
    print_cb_details(cb_in0););
    for (uint32_t block = 0; block < per_core_block_cnt; block += per_core_block_size) {
	    
            UNPACK(
                // Read a tile to cb_in0 from DRAM
                uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);
                cb_reserve_back_df(cb_in0, per_core_block_size);
                l1_write_addr_in0 = get_write_ptr(cb_in0);
		DPRINT << "src0_noc_addr " << src0_noc_addr << " l1_write_addr_in0 " << l1_write_addr_in0 << " ublocksize_bytes_0 " << ublock_size_bytes_0 << ENDL();

		DPRINT << "rd_ptr: " << get_local_cb_interface(cb_in0).fifo_rd_ptr << ENDL();
                noc_async_read(src0_noc_addr, l1_write_addr_in0 << 4, ublock_size_bytes_0, 1);
                noc_async_read_barrier(1);
                cb_push_back_df(cb_in0, per_core_block_size);
                src0_addr += ublock_size_bytes_0;
                print_full_tile(cb_in0, 0);


                // Read a tile to cb_in0 from DRAM
                uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_addr);
                cb_reserve_back_df(cb_in1, per_core_block_size);
                l1_write_addr_in1 = get_write_ptr(cb_in1);
                noc_async_read(src1_noc_addr, l1_write_addr_in1 << 4, ublock_size_bytes_1, 1);
                noc_async_read_barrier(1);
                cb_push_back_df(cb_in1, per_core_block_size);
                src1_addr += ublock_size_bytes_1;
            print_full_tile(cb_in1, 0););


        // Wait for the input circular buffers to be filled with per_core_block_size tiles
        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);

        // Wait for enough space to be available in the output circular buffer
        cb_reserve_back_st(cb_out0, per_core_block_size);

        // Perform the elementwise operation on the tiles in the block
        // and store them in the destination register
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            add_tiles_st(cb_in0, cb_in1, i, i, i);
        }

        // Pack all the output tiles from destination register out to
        // the output circular buffer that resides in L1 memory
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile_st(i, cb_out0);
        }

        // Update the write pointer and counts for the output circular buffer.
        cb_push_back_st(cb_out0, per_core_block_size);

        // Pop out the used input tiles
        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);
    }
}
}  // namespace NAMESPACE
