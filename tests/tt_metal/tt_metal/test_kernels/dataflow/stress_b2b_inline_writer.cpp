// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_noc_x = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_y = get_arg_val<uint32_t>(1);
    uint32_t dst_addr_0 = get_arg_val<uint32_t>(2);
    uint32_t dst_addr_1 = get_arg_val<uint32_t>(3);
    uint32_t num_writes = 100000;  // get_arg_val<uint32_t>(4);

    uint64_t dst_noc_addr_0 = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr_0);
    uint64_t dst_noc_addr_1 = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr_1);

    for (uint32_t i = 0; i < num_writes; i++) {
        noc_inline_dw_write(dst_noc_addr_0, i);
        noc_inline_dw_write(dst_noc_addr_1, i);
    }

    noc_async_write_barrier();
}
