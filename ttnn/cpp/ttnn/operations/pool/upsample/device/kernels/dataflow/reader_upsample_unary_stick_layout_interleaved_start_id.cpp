// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// Target 8KB of data before a single barrier for 8x8 grid of readers
// const uint32_t get_barrier_read_threshold(const uint32_t stick_size_bytes) {
//     const uint32_t threshold = 8096 * 8 / stick_size_bytes;
//     return threshold > 0 ? threshold : 1;
// }

void kernel_main() {
    constexpr uint32_t cb_half_buffer = 8;
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t num_sticks = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr bool src0_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool src_stick_size_is_pow2 = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t src_log_base_2_of_page_size = get_compile_time_arg_val(3);

    // const uint32_t barrier_threshold = get_barrier_read_threshold(stick_size);
    uint32_t barrier_count = 0;

    const auto s0 = get_interleaved_addr_gen<src0_is_dram, src_stick_size_is_pow2>(
        src_addr, stick_size, src_log_base_2_of_page_size);
    DeviceZoneScopedN("reader");
    // reader copied the data from DRAM to CB buffer.
    uint64_t src_noc_addr = get_noc_addr(0, s0);

    noc_async_read_one_packet_set_state(src_noc_addr, stick_size);

    // for (uint32_t i = 0; i < num_sticks; ++i) {
    //     {
    //         DeviceZoneScopedN("cb_reserve_back");
    //         cb_reserve_back(cb_id_in0, 1);
    //     }
    //     uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    //     src_noc_addr = get_noc_addr(i, s0);
    //     {
    //         DeviceZoneScopedN("noc_read");
    //         noc_async_read_one_packet_update_state(src_noc_addr);
    //         noc_async_read_one_packet_with_state<true>(src_noc_addr, l1_write_addr);
    //         // noc_async_read(src_noc_addr, l1_write_addr, stick_size);

    //         if (++barrier_count == 8) { // batch reads to avoid constantly waiting barrier

    //             noc_async_read_barrier();
    //             barrier_count = 0;
    //             {
    //                 DeviceZoneScopedN("cb_push_back");
    //                 cb_push_back(cb_id_in0, 8);
    //             }
    //         }
    //     }

    // }

    for (uint32_t i = 0; i < num_sticks / cb_half_buffer; ++i) {
        {
            DeviceZoneScopedN("cb_reserve_back");
            cb_reserve_back(cb_id_in0, cb_half_buffer);
        }

        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t j = 0; j < cb_half_buffer; j++) {
            src_noc_addr = get_noc_addr(i * cb_half_buffer + j, s0);
            {
                DeviceZoneScopedN("noc_read");
                noc_async_read_one_packet_update_state(src_noc_addr);
                noc_async_read_one_packet_with_state<true>(src_noc_addr, l1_write_addr);
            }
            l1_write_addr += stick_size;
        }
        noc_async_read_barrier();
        {
            DeviceZoneScopedN("cb_push_back");
            cb_push_back(cb_id_in0, cb_half_buffer);
        }
    }
}
