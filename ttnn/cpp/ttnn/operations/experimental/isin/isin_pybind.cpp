// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isin_pybind.hpp"

#include "isin_common.hpp"

#include "ttnn-pybind/decorators.hpp"

#include "isin.hpp"

namespace ttnn::operations::experimental::isin::detail {

void bind_isin_operation(py::module& module) {
    auto doc = "";

    using OperationType = decltype(ttnn::experimental::isin);
    bind_registered_operation(
        module,
        ttnn::experimental::isin,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const Any& elements,
               const Any& test_elements,
               const bool& assume_unique,
               const bool& invert,
               std::optional<Tensor>& output_tensor,
               const QueueId& queue_id = DefaultQueueId) -> Tensor {
                return self(queue_id, elements, test_elements, assume_unique, invert, output_tensor);
            },
            py::arg("elements"),
            py::arg("test_elements"),
            py::kw_only(),
            py::arg("assume_unique") = false,
            py::arg("invert") = false,
            py::arg("out") = std::nullopt,
            py::arg("queue_id") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::isin::detail
