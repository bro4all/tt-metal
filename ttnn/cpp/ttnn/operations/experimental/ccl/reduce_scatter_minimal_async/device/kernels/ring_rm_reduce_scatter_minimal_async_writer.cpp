// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr BufferType intermediate_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr BufferType output_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr uint32_t cb_compute_output_id = get_compile_time_arg_val(4);
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
    address_t intermediate_address = get_arg_val<address_t>(arg_idx++);
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    pkt_hdr->to_chip_unicast(1);

    volatile PACKET_HEADER_TYPE* pkt_hdr_seminc =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);

    // interleaved addrgen
    constexpr bool intermediate_is_dram = intermediate_type == tt::tt_metal::BufferType::DRAM;
    auto intermediate_addrgen = InterleavedAddrGenFast<intermediate_is_dram>{
        .bank_base_address = intermediate_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_compute_output_id)};
    constexpr bool output_is_dram = output_type == tt::tt_metal::BufferType::DRAM;
    auto output_addrgen = InterleavedAddrGenFast<output_is_dram>{
        .bank_base_address = output_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_compute_output_id)};

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    auto* fabric_direction_connection =
        direction ? &fabric_connection.get_forward_connection() : &fabric_connection.get_backward_connection();

    int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
    for (uint32_t i = 0; i < ring_size; ++i) {
        uint32_t cb_output_id = i > 0 ? cb_compute_output_id : cb_reader_output_id;

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

        if (i < (ring_size - 1)) {
            // If not the last slice, write what's on cb_output_id forward
            while (rows_read < rows_to_read) {
                cb_wait_front(cb_output_id, 1);
                size_t l1_read_addr = get_read_ptr(cb_output_id);
                uint32_t page_id = rows_read;
                uint64_t remote_noc0_dest_noc_addr =
                    get_noc_addr(page_id, intermediate_addrgen, single_slice_row_offset_size, 0 /*noc_id*/);

                uint32_t packet_size_in_bytes = /* TODO: (GR) */

                    uint32_t packets_to_send = div_up(slice_width_row_size, /* TODO: (GR) packet_size_in_bytes */);
                while (packets_to_send > 0) {
                    if (packets_to_send == 1) {
                        // Standard fabric write
                        write_and_advance_local_read_address_for_fabric(
                            remote_noc0_dest_noc_addr,
                            pkt_hdr,
                            fabric_direction_connection,
                            l1_read_addr,
                            packet_size_in_bytes, );

                        remote_noc0_dest_noc_addr += packet_size_in_bytes;
                        packets_to_send--;
                    } else {
                        // Scatter fabric write
                        scatter_write_and_advance_local_read_address_for_fabric(
                            remote_noc0_dest_noc_addr,
                            remote_noc0_dest_noc_addr + packet_size_in_bytes,
                            pkt_hdr,
                            fabric_direction_connection,
                            l1_read_addr,
                            packet_size_in_bytes,
                            packet_size_in_bytes, );

                        remote_noc0_dest_noc_addr += 2 * packet_size_in_bytes;
                        packets_to_send -= 2;
                    }
                }

                cb_pop_front(cb_output_id, 1);

                // Increment rows_read by two
                // Once for the page we just read, and once for the row going the other direction
                rows_read += 2;
            }

            noc_async_write_barrier();

            // 2. unicast output ready semaphore
            uint64_t out_ready_sem_noc_addr_in_pkt =
                safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
            pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                out_ready_sem_noc_addr_in_pkt,
                static_cast<uint16_t>(1),  // increment 1
                32});
            // Write the unicast packet (forward)
            pkt_hdr_seminc->to_chip_unicast(1);
            fabric_direction_connection->wait_for_empty_write_slot();
            fabric_direction_connection->send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
            noc_async_writes_flushed();
        } else {
            // Otherwise, on the last slice, write it to output buffer
            while (rows_read < rows_to_read) {
                cb_wait_front(cb_output_id, 1);
                size_t l1_read_addr = get_read_ptr(cb_output_id);
                uint32_t page_id = rows_read;
                uint64_t local_noc_addr = get_noc_addr(page_id, output_addrgen);
                noc_async_write(l1_read_addr, local_noc_addr, input_tensor_page_size, single_slice_row_offset_size);

                noc_async_write_barrier();
                cb_pop_front(cb_output_id, 1);

                // Increment rows_read by two
                // Once for the page we just read, and once for the row going the other direction
                rows_read += 2;
            }
        }

        // Next slice idx
        if constexpr (direction) {
            slice_idx--;
        } else {
            slice_idx++;
        }
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
}
