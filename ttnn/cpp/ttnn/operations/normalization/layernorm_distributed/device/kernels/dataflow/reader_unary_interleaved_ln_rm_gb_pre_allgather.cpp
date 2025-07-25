// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs from interleaved dram.
 */

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "debug/assert.h"
#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "READER_LAYERNORM: Starting kernel" << ENDL();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);     // Source address in dram
    const uint32_t NCHt = get_arg_val<uint32_t>(1);         // Number of NCH tiles
    const uint32_t Wt = get_arg_val<uint32_t>(2);           // Width in tiles
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);  // Tile offset for this core
    const bool is_merge_core = get_arg_val<uint32_t>(4);
    const uint32_t reduce_core_noc_x = get_arg_val<uint32_t>(5);
    const uint32_t reduce_core_noc_y = get_arg_val<uint32_t>(6);
    const uint32_t y = get_arg_val<uint32_t>(7);

    DPRINT << "READER_LAYERNORM: src_addr=" << src_addr << " NCHt=" << NCHt << " Wt=" << Wt
           << " tile_offset=" << tile_offset << ENDL();
    DPRINT << "READER_LAYERNORM: is_merge_core=" << (int)is_merge_core << " reduce_core_noc_x=" << reduce_core_noc_x
           << " reduce_core_noc_y=" << reduce_core_noc_y << ENDL();

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_14;
    constexpr uint32_t cb_x2_merge = tt::CBIndex::c_15;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);
    const DataFormat src0_data_format = get_dataformat(cb_inp);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    uint32_t reducer_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));  // semaphore for reducer
    constexpr uint32_t num_cores_to_wait = get_compile_time_arg_val(3);

    DPRINT << "READER_LAYERNORM: src0_tile_bytes=" << src0_tile_bytes << " blk=" << blk
           << " num_cores_to_wait=" << num_cores_to_wait << ENDL();

    const uint64_t in0_sender_semaphore_noc_addr =
        get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, reducer_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    const InterleavedAddrGenFast<src0_is_dram> src_a = {
        .bank_base_address = src_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    // Generate constant tiles for reduce scalar
    uint32_t scaler = get_arg_val<uint32_t>(8);

    DPRINT << "READER_LAYERNORM: Generating reduce scaler with value=" << scaler << ENDL();
    generate_reduce_scaler(cb_reduce, scaler);

    uint32_t inp_tile_idx = tile_offset;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        DPRINT << "READER_LAYERNORM: Processing ncht=" << ncht << "/" << NCHt << ENDL();

        // read input tiles
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            DPRINT << "READER_LAYERNORM: Processing wt=" << wt << "/" << Wt << " blk=" << blk << ENDL();

            DPRINT << "READER_LAYERNORM: Reserving cb_inp" << ENDL();
            cb_reserve_back(cb_inp, blk);
            uint32_t inp_wr_ptr = get_write_ptr(cb_inp);

            for (uint32_t r = 0; r < blk; r++) {
                DPRINT << "READER_LAYERNORM: Reading tile " << inp_tile_idx << ENDL();
                noc_async_read_tile(inp_tile_idx, src_a, inp_wr_ptr);
                inp_wr_ptr += src0_tile_bytes;
                inp_tile_idx++;
            }
            DPRINT << "READER_LAYERNORM: Waiting for async read barrier" << ENDL();
            noc_async_read_barrier();
            DPRINT << "READER_LAYERNORM: Pushing cb_inp" << ENDL();
            // print cb inp data
            uint32_t* inp_data = (uint32_t*)get_read_ptr(cb_inp);
            DPRINT << "READER_LAYERNORM: cb_inp data: " << inp_data[0] << " " << inp_data[1] << " " << inp_data[2]
                   << ENDL();
            cb_push_back(cb_inp, blk);

        }  // wt loop

    }  // ncht loop

    DPRINT << "READER_LAYERNORM: Waiting for cb_out" << ENDL();
    // wait on cb_out and then write to merge core over noc
    cb_wait_front(cb_out, 1);

    DPRINT << "READER_LAYERNORM: Got cb_out" << ENDL();
    // print cb out data
    uint32_t* out_data = (uint32_t*)get_read_ptr(cb_out);
    DPRINT << "READER_LAYERNORM: cb_out data: " << out_data[0] << " " << out_data[1] << " " << out_data[2] << ENDL();

    uint32_t o_write_size = 2 * 32 * 32;
    uint32_t worker_offset = o_write_size * y;  // Fixed: added semicolon and space
    uint64_t output_write_addr =
        get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, get_write_ptr(cb_x2_merge)) + worker_offset;

    DPRINT << "READER_LAYERNORM: Writing to output_write_addr=" << output_write_addr << " size=" << o_write_size
           << ENDL();
    noc_async_write(get_read_ptr(cb_out), output_write_addr, o_write_size);
    DPRINT << "READER_LAYERNORM: Waiting for async write barrier" << ENDL();
    noc_async_write_barrier();

    // increase semaphore

    DPRINT << "READER_LAYERNORM: Incrementing semaphore" << in0_sender_semaphore_noc_addr << ENDL();
    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
    noc_async_atomic_barrier();
    DPRINT << "READER_LAYERNORM: Atomic barrier complete" << ENDL();

    if (is_merge_core) {
        DPRINT << "READER_LAYERNORM: Merge core - waiting for semaphore" << reducer_semaphore_addr << ENDL();

        for (uint32_t i = 0; i < num_cores_to_wait; i++) {
            noc_semaphore_wait_min(in0_receiver_semaphore_addr_ptr, i);
            // print cb_x2_merge data
            uint32_t* x2_data = (uint32_t*)get_read_ptr(cb_x2_merge);
            DPRINT << "READER_LAYERNORM: Merge core - cb_x2_merge data: " << x2_data[0] << " " << x2_data[1] << " "
                   << x2_data[2] << ENDL();
            DPRINT << "READER_LAYERNORM: Merge core - waited for semaphore " << i << "/" << num_cores_to_wait << ENDL();
        }
        noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, num_cores_to_wait);
        DPRINT << "READER_LAYERNORM: Merge core - got semaphore" << ENDL();
        cb_push_back(cb_x2_merge, num_cores_to_wait);
        DPRINT << "READER_LAYERNORM: Merge core - pushed cb_x2_merge" << ENDL();
    }

    DPRINT << "READER_LAYERNORM: Kernel complete" << ENDL();
}
