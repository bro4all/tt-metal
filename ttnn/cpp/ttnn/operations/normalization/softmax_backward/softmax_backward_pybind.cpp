// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_pybind.hpp"

#include "softmax_backward.hpp"
#include "ttnn-pybind/decorators.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn::operations::normalization::detail {
namespace py = pybind11;

// Softmax backward operation base
void bind_normalization_softmax_backward_operation(py::module& module) {
    const auto doc =
        R"doc(
            Computes the softmax function over the specified dimension of the input tensor.

            The softmax function is defined as:

            .. math::
                \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}

            Args:
                input_tensor (ttnn.Tensor): The input tensor to apply softmax to.
                dim (int, optional): The dimension along which to compute softmax. Defaults to -1 (last dimension).

            Keyword Args:
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. If not provided, inherits from input tensor.
                compute_kernel_config (DeviceComputeKernelConfig, optional): Compute kernel configuration for the operation.
                numeric_stable (bool, optional): Whether to use numerically stable softmax computation. Defaults to False.

            Returns:
                ttnn.Tensor: Output tensor with softmax applied along the specified dimension.

            Supported dtypes and layouts

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, BFLOAT8_B
                 - TILE

            Example:
                .. code-block:: python

                    tensor = ttnn.rand((1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                    result = ttnn.softmax(tensor, dim=-1)
    )doc";

    using OperationType = decltype(ttnn::softmax_backward);

    ttnn::bind_registered_operation(
        module,
        ttnn::softmax_backward,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& softmax_output_tensor,
               const ttnn::Tensor& grad_tensor,
               const uint32_t dim,
               QueueId queue_id) -> ttnn::Tensor {
                return self(
                    queue_id,
                    softmax_output_tensor,
                    grad_tensor,
                    dim /*, memory_config, compute_kernel_config, numeric_stable*/);
            },
            py::arg("softmax_output_tensor").noconvert(),
            py::arg("grad_tensor").noconvert(),
            py::arg("dim") = -1,
            py::arg("queue_id") = DefaultQueueId});
}

void bind_normalization_softmax_backward(py::module& module) { bind_normalization_softmax_backward_operation(module); }
}  // namespace ttnn::operations::normalization::detail
