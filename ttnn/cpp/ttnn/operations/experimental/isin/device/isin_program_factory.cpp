// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isin_program_factory.hpp"

#include "tt-metalium/device.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

namespace ttnn::operations::experimental::scatter {

namespace {
constexpr uint32_t BIT_MASK_32 = 32 - 1;

uint64_t ceil32(const uint64_t& number) {
    return ((number & BIT_MASK_32) == 0) ? number : ((number | BIT_MASK_32) + 1);
}

bool is_pow2_min32(const uint64_t& number) { return ((number & (number - 1)) == 0) && number >= 32; }
}  // namespace

}  // namespace ttnn::operations::experimental::scatter
