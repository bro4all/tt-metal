// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isin.hpp"

#include "device/isin_device_operation.hpp"

#include "isin_common.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

Tensor normalize_to_tensor(const Any& input) {
    return std::visit(
        [](auto&& value) -> Tensor {
            using ValueType = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<ValueType, Tensor>) {
                // Already a Tensor, just return it
                return value;
            } else {
                // Numeric case: construct a scalar Tensor from the value
                return ttnn::on{value};
            }
        },
        input);
}

void validate_inputs(const Tensor& input_tensor, const Tensor& test_tensor) {
    //
}

void check_support(const Tensor& input_tensor, const Tensor& test_tensor) {
    //
}

IsInOP_WIPTensors isin_preprocessing(const Any& input_tensor, const Any& test_tensor) {
    // flatten variants
    Tensor processed_input_tensor = normalize_to_tensor(input_tensor);
    Tensor processed_test_tensor = normalize_to_tensor(test_tensor);
    Tensor index_tensor;

    // calculate mult
    // reshape to [64*mult, ceil(Dprod/64/Mult)]
    // sort
    // cast to row-major

    return {processed_input_tensor, processed_test_tensor, index_tensor};
}

Tensor isin_postprocessing(const Tensor& output_tensor) {
    Tensor processed_output_tensor = output_tensor;
    // scatter
    // cast to tiled
    // reshape to original

    return processed_output_tensor;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

Tensor IsinOperation::invoke(
    const QueueId& queue_id,
    const Any& elements,
    const Any& test_elements,
    const bool& assume_unique,
    const bool& invert,
    std::optional<Tensor>& opt_out) {
    using CMAKE_UNIQUE_NAMESPACE::isin_postprocessing;
    using CMAKE_UNIQUE_NAMESPACE::isin_preprocessing;
    const auto wip_tensors = isin_preprocessing(elements, test_elements);

    const auto output_tensor = ttnn::prim::isin(queue_id, wip_tensors, assume_unique, invert);

    return isin_postprocessing(output_tensor);
}

}  // namespace ttnn::operations::experimental
