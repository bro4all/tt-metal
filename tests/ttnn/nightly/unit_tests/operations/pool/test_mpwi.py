# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import math
import pytest

# Set PyTorch print options to show full tensors without abbreviation
torch.set_printoptions(threshold=float("inf"), linewidth=200, precision=5, sci_mode=False)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_max_pool2d_with_indices(device):
    in_n = 1
    in_h = 3
    in_w = 3
    in_c = 16
    kernel_size = [3, 3]
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]
    shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    ceil_mode = False
    ttnn_dtype = ttnn.bfloat16
    # ttnn_dtype = ttnn.bfloat8_b

    tensor_shape = (in_n, in_c, in_h, in_w)  # NCHW format

    pad_h = padding[0] * 2  # padding is [top/bottom, left/right] but we need total padding
    pad_w = padding[1] * 2
    dilation_h = dilation[0]
    dilation_w = dilation[1]
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    stride_h = stride[0]
    stride_w = stride[1]

    if ceil_mode:
        out_h = math.ceil((in_h + pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.ceil((in_w + pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1
        if ((out_h - 1) * stride_h) >= (in_h + padding[0]):
            ceil_mode_out_shape_adj = True
            out_h -= 1
        if ((out_w - 1) * stride_w) >= (in_w + padding[1]):
            ceil_mode_out_shape_adj = True
            out_w -= 1
    else:
        out_h = math.floor((in_h + pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h) + 1
        out_w = math.floor((in_w + pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w) + 1

    # Create tensor filled with height and width coordinates
    torch.manual_seed(0)
    # torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    # Create tensor where each element equals its HW coordinate (h * in_w + w)
    # torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    torch_input = torch.zeros(tensor_shape, dtype=torch.bfloat16)
    for n in range(in_n):
        for c in range(in_c):
            for h in range(in_h):
                for w in range(in_w):
                    torch_input[n, c, h, w] = h * in_w + w

    ttnn_input_shape = (1, 1, in_n * in_h * in_w, in_c)
    torch_input_permuted = torch.permute(torch_input, (0, 2, 3, 1))  # N, H, W, C
    torch_input_reshaped = torch_input_permuted.reshape(ttnn_input_shape)  # NHW, C
    ttnn_layout = ttnn.ROW_MAJOR_LAYOUT
    if ttnn_dtype == ttnn.bfloat8_b:
        ttnn_layout = ttnn.TILE_LAYOUT
    # ttnn_input = ttnn.from_torch(torch_input_reshaped, ttnn_dtype, layout=ttnn_layout, device=device)

    # Memory configuration for ttnn_input
    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [9, 32],
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardMode.PHYSICAL,
        ),
    )
    ttnn_input = ttnn.from_torch(
        torch_input_reshaped, ttnn_dtype, layout=ttnn_layout, memory_config=memory_config, device=device
    )

    ttnn_input_sliced = ttnn.slice(ttnn_input, (0, 0, 0, 0), (1, 1, 9, 16))  # NHW, C

    core_coord = ttnn.CoreCoord(0, 0)
    ttnn_input_shard = ttnn_input.extract_shard(core_coord)
    ttnn_input_sliced_shard = ttnn_input_sliced.extract_shard(core_coord)

    # Convert to torch tensors to see full values without abbreviation
    ttnn_input_torch = ttnn.to_torch(ttnn_input)
    ttnn_input_sliced_torch = ttnn.to_torch(ttnn_input_sliced)

    print("TTNN Input Torch:")
    print(ttnn_input_torch)
    print("TTNN Input Sliced Torch:")
    print(ttnn_input_sliced_torch)

    # print("TTNN Input Torch Shard:")
    # print(ttnn.to_torch(ttnn_input_shard))
    # print("TTNN Input Sliced Torch Shard:")
    # print(ttnn.to_torch(ttnn_input_sliced_shard))

    ttnn_output, indices = ttnn.max_pool2d(
        input_tensor=ttnn_input_sliced,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        # applied_shard_scheme=shard_scheme,
        ceil_mode=ceil_mode,
        in_place_halo=False,
        deallocate_input=False,
        reallocate_halo_output=True,
        return_indices=True,
    )
