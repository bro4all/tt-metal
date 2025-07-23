// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "isin_common.hpp"

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::experimental {

struct IsinOperation {
    static Tensor invoke(
        const QueueId& queue_id,
        const IsInOP_WIPTensors& wip_tensors,
        const bool& assume_unique,
        const bool& invert,
        std::optional<Tensor>& opt_out);
};

}  // namespace operations::experimental

namespace experimental {
constexpr auto isin =
    ttnn::register_operation<"ttnn::experimental::isin", ttnn::operations::experimental::IsinOperation>();
}  // namespace experimental

}  // namespace ttnn
