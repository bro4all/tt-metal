// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include <variant>
#include <cstdint>

namespace ttnn::operations::experimental {

using ttnn::Tensor;

using Any = std::variant<Tensor, int32_t, float>;

struct IsInOP_WIPTensors {
    const Tensor processed_elements_tnenor;  // [64*Mult, FullVolume/64/Mult], row-major
    const Tensor processed_test_tensor;      // 1D, row-major
    const Tensor index_tensor;               // [64*Mult, FullVolume/64/Mult], row-major
};

}  // namespace ttnn::operations::experimental
