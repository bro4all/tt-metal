# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range_dtype, compare_pcc


# TODO: move to utility_funcs.py ?
def torch_to_device(pt_tensor: torch.Tensor, ttnn_dtype: ttnn.types, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.Tensor(pt_tensor, ttnn_dtype).to(ttnn.TILE_LAYOUT).to(device)


def reference_softmax_backward_output(y: torch.Tensor, grad: torch.Tensor, axis: int) -> torch.Tensor:
    dot = (y * grad).sum(dim=axis, keepdim=True)
    return y * (grad - dot)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        #        (torch.Size([1, 1, 320, 384])),
        #        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        #        ttnn.float32
    ],
)
def test_bw_softmax(input_shapes, dtype, device):
    grad_data, grad_tensor = data_gen_with_range_dtype(input_shapes, -200, 199, device, ttnn_dtype=dtype)
    in_data, input_tensor = data_gen_with_range_dtype(
        input_shapes, -200, 199, device, required_grad=True, ttnn_dtype=dtype
    )

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    pt_softmax_tensor = torch.softmax(in_data, dim=3, dtype=torch_dtype)
    tt_softmax_tensor = torch_to_device(pt_softmax_tensor, dtype, device)

    # tt_output_tensor_on_device = ttnn.operations.moreh.softmax_backward(tt_softmax_tensor, grad_tensor, dim=3)
    tt_output_tensor_on_device = ttnn.softmax_backward(tt_softmax_tensor, grad_tensor, dim=3)
    pt_output_tensor_on_device = ttnn.to_torch(tt_output_tensor_on_device)
    pt_output_tensor_reference = reference_softmax_backward_output(pt_softmax_tensor, grad_data, axis=3)
    print(pt_output_tensor_on_device - pt_output_tensor_reference)
    # print(pt_output_tensor_reference)
    assert torch.allclose(pt_output_tensor_on_device, pt_output_tensor_reference, rtol=1e-2, atol=1e-2)
    # tt_output_tensor_reference = torch_to_device(pt_output_tensor_reference, dtype, device)

    # comp_pass = compare_pcc(tt_output_tensor_on_device, pt_output_tensor_reference, pcc=0.99)
    # assert comp_pass
