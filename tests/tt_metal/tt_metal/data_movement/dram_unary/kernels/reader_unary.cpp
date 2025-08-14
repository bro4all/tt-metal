// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

#define WALL_CLOCK_0_PTR ((volatile uint32_t*)(RISCV_DEBUG_REG_WALL_CLOCK_L))
#define WALL_CLOCK_1_PTR ((volatile uint32_t*)(RISCV_DEBUG_REG_WALL_CLOCK_H))
inline uint64_t get_wall_clock() {
    uint64_t lo = *WALL_CLOCK_0_PTR;
    uint64_t hi = *WALL_CLOCK_1_PTR;
    return lo | (hi << 32);
}

// DRAM to L1 read
void kernel_main() {
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t pages_per_transaction = get_compile_time_arg_val(2);
    constexpr uint32_t bytes_per_page = get_compile_time_arg_val(3);
    constexpr uint32_t dram_addr = get_compile_time_arg_val(4);
    constexpr uint32_t dram_channel = get_compile_time_arg_val(5);
    constexpr uint32_t local_l1_addr = get_compile_time_arg_val(6);
    constexpr uint32_t sem_id = get_compile_time_arg_val(7);

    constexpr uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    constexpr bool dram = true;
    uint64_t dram_noc_addr = get_noc_addr_from_bank_id<dram>(dram_channel, dram_addr);
    auto curr_l1_addr = local_l1_addr;

    uint32_t sem_addr = get_semaphore(sem_id);
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    auto start_wall_clock = get_wall_clock();
    DPRINT << "start_wall_clock: " << start_wall_clock << " " << (start_wall_clock >> 32) << ENDL();
    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            noc_async_read(dram_noc_addr, curr_l1_addr, bytes_per_transaction);
            // curr_l1_addr += bytes_per_transaction;
            dram_noc_addr += bytes_per_transaction;
        }
        noc_async_read_barrier();
    }
    auto end_wall_clock = get_wall_clock();
    DPRINT << "end_wall_clock: " << end_wall_clock << " " << (end_wall_clock >> 32) << ENDL();
    DPRINT << "wall_clock_diff: " << (end_wall_clock - start_wall_clock) << ENDL();

    // Set the semaphore to indicate that the writer can proceed
    noc_semaphore_set(sem_ptr, 1);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
    DeviceTimestampedData("Test id", test_id);
}
