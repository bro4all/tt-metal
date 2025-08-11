// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary_st.h"
#include "dataflow_api.h"

inline void cb_push_back_from_dram(
    uint32_t dram_bank_id, uint32_t dram_addr, uint32_t cb_id, uint32_t num_tiles, uint32_t noc = noc_index) {
    UNPACK(uint32_t tile_size_bytes = get_tile_size(cb_id);
           uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, dram_addr);
           uint32_t l1_write_addr = get_write_ptr(cb_id);
           noc_async_read(dram_noc_addr, l1_write_addr << 4, tile_size_bytes * num_tiles, noc);
           noc_async_read_barrier(noc);
           cb_push_back_df(cb_id, num_tiles););
}

inline void cb_pop_front_to_dram(
    uint32_t dram_bank_id, uint32_t dram_addr, uint32_t cb_id, uint32_t num_tiles, uint32_t noc = noc_index) {
    UNPACK(uint32_t tile_size_bytes = get_tile_size(cb_id);
           uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, dram_addr);
           uint32_t l1_read_addr = get_read_ptr(cb_id);
           noc_async_write(l1_read_addr << 4, dram_noc_addr, tile_size_bytes * num_tiles, noc);
           noc_async_write_barrier(noc);
           cb_pop_front_df(cb_id, num_tiles););
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

    // How many tiles per block (1 or 2 due to limitations)
    uint32_t per_core_block_size = get_arg_val<uint32_t>(6);

    // For writing out the results
    uint32_t dst_addr = get_arg_val<uint32_t>(7);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(8);

    // Input and output circular buffer ids.
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // single-tile ublocks
    uint32_t ublock_size_bytes_0 = get_tile_size(cb_in0);
    uint32_t ublock_size_bytes_1 = get_tile_size(cb_in1);
    uint32_t ublock_size_bytes_dst = get_tile_size(cb_out0);

    // Initialize the parts that are common among binary operations
    binary_op_init_common_st(cb_in0, cb_in1, cb_out0);

    // Initialize the parts that required specifically for this binary operatoins
    binary_tiles_init_st<false, EltwiseBinaryType::ELWADD>(cb_in0, cb_in1);

    for (uint32_t block = 0; block < per_core_block_cnt; block += per_core_block_size) {
        cb_push_back_from_dram(src0_bank_id, src0_addr, cb_in0, per_core_block_size);
        src0_addr += ublock_size_bytes_0 * per_core_block_size;

        cb_push_back_from_dram(src1_bank_id, src1_addr, cb_in1, per_core_block_size);
        src1_addr += ublock_size_bytes_1 * per_core_block_size;

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
        cb_pop_front_to_dram(dst_bank_id, dst_addr, cb_out0, per_core_block_size);
        dst_addr += ublock_size_bytes_dst * per_core_block_size;

        // Pop out the used input tiles
        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);
    }
}
}  // namespace NAMESPACE
