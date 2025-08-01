// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <cstdint>
#include <utility>

#include "debug/dprint.h"

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType input_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr BufferType intermediate_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr uint32_t cb_input_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(4);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(5);
constexpr uint32_t input_tensor_num_pages = get_compile_time_arg_val(6);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(7);
constexpr uint32_t slice_width_row_size = get_compile_time_arg_val(8);
constexpr uint32_t ring_size = get_compile_time_arg_val(9);
constexpr bool direction = get_compile_time_arg_val(10);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    constexpr bool input_tensor_is_dram = input_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto input_tensor_addrgen = InterleavedAddrGen<input_tensor_is_dram>{
        .bank_base_address = input_tensor_address, .page_size = input_tensor_page_size};
    constexpr bool intermediate_tensor_is_dram = intermediate_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto intermediate_tensor_addrgen = InterleavedAddrGen<intermediate_tensor_is_dram>{
        .bank_base_address = intermediate_tensor_address, .page_size = input_tensor_page_size};

    int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
    for (uint32_t i = 0; i < ring_size; ++i) {
        const bool do_reduce = i != 0;
        uint32_t cb_in0 = do_reduce ? cb_input_id : cb_reader_output_id;

        uint32_t actual_slice_idx;
        if constexpr (direction) {
            actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
        } else {
            actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
        }

        // TODO: (GR) Handle num_links > 1
        uint32_t single_slice_row_offset_size = actual_slice_idx * slice_width_row_size;
        uint32_t rows_read = 0;
        uint32_t rows_to_read = input_tensor_num_pages;

        // Skip first row if we're going backwards
        if constexpr (!direction) {
            rows_read++;
        }

        if (do_reduce) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), i);
            if (i == (ring_size - 1)) {
                // Reset the semaphore
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
            }
        }

        while (rows_read < rows_to_read) {
            // Read input tensor slice
            cb_reserve_back(cb_in0, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_in0);
            uint32_t page_id = rows_read;
            uint64_t noc_read_addr = get_noc_addr(page_id, input_tensor_addrgen, single_slice_row_offset_size);
            noc_async_read(noc_read_addr, l1_write_addr, slice_width_row_size);

            // Read intermediate tensor slice
            if (do_reduce) {
                cb_reserve_back(cb_intermediate_id, 1);
                uint32_t intermediate_l1_write_addr = get_write_ptr(cb_intermediate_id);
                uint32_t intermediate_page_id = rows_read;
                uint64_t intermediate_noc_read_addr =
                    get_noc_addr(intermediate_page_id, intermediate_tensor_addrgen, single_slice_row_offset_size);
                noc_async_read(intermediate_noc_read_addr, intermediate_l1_write_addr, slice_width_row_size);

                noc_async_read_barrier();
                cb_push_back(cb_intermediate_id, 1);
            }

            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);

            // Increment rows_read by two
            // Once for the page we just read, and once for the row going the other direction
            rows_read += 2;
        }

        // Next slice idx
        if constexpr (direction) {
            slice_idx--;
        } else {
            slice_idx++;
        }
    }
}
