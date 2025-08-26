// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"

#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t x_block_size = get_compile_time_arg_val(0);
    uint32_t num_blocks = get_arg_val<uint32_t>(0);
    constexpr auto cb_in = tt::CBIndex::c_0;

    for (uint32_t n = 0; n < num_blocks; n++) {
        cb_wait_front(cb_in, x_block_size);

        // UNPACK(DPRINT << "BLOCK NUM " << n << ENDL(); DPRINT << "X block size " << x_block_size << ENDL();
        //        tt::compute::common::print_full_tile(cb_in, 0, true));

        cb_pop_front(cb_in, x_block_size);
    }
}
}  // namespace NAMESPACE
