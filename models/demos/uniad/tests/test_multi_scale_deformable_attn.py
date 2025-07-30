import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: torch.Tensor,
) -> torch.Tensor:
    bs, num_keys, num_heads, head_dim = value.shape
    num_levels = value_spatial_shapes.shape[0]
    num_queries = sampling_locations.shape[1]
    num_points = sampling_locations.shape[4]

    # Split value into a list of tensors for each level
    value_list = []
    start = 0
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        len_l = h_l * w_l
        value_l = value[:, start : start + len_l, :, :]
        value_list.append(value_l)
        start += len_l

    # Normalize sampling locations to [-1, 1]
    sampling_grids = []
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        grid = sampling_locations[:, :, :, lvl, :, :]
        grid = grid.clone()
        grid[..., 0] = grid[..., 0] / w_l * 2 - 1
        grid[..., 1] = grid[..., 1] / h_l * 2 - 1
        sampling_grids.append(grid)

    # Perform sampling and attention
    output = torch.zeros(bs, num_queries, num_heads, head_dim, device=value.device)
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        value_l = value_list[lvl].permute(0, 2, 3, 1).reshape(bs * num_heads, head_dim, h_l, w_l)
        grid = sampling_grids[lvl].permute(0, 2, 1, 3, 4).reshape(bs * num_heads, num_queries * num_points, 1, 2)
        sampled = F.grid_sample(value_l, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.view(bs, num_heads, head_dim, num_queries, num_points).permute(0, 3, 1, 4, 2)
        attn = attention_weights[:, :, :, lvl, :].unsqueeze(-1)
        output += (sampled * attn).sum(-2)

    return output.view(bs, num_queries, num_heads * head_dim)


def multi_scale_deformable_attn_ttnn(
    value_ttnn: ttnn.Tensor,
    spatial_shapes_ttnn: torch.Tensor,
    level_start_index_ttnn: torch.Tensor,
    sampling_locations_ttnn: ttnn.Tensor,
    attention_weights_ttnn: torch.Tensor,
    im2col_step_ttnn: torch.Tensor,
    device: ttnn.Device,
) -> ttnn.Tensor:
    """
    TTNN implementation of multi-scale deformable attention.
    Uses ttnn operations where available, falls back to torch for grid_sample.
    """
    bs, num_keys, num_heads, head_dim = value_ttnn.shape
    num_levels = spatial_shapes_ttnn.shape[0]
    num_queries = sampling_locations_ttnn.shape[1]
    num_points = sampling_locations_ttnn.shape[4]

    # Split value into a list of tensors for each level using ttnn operations
    value_list = []
    start = 0
    print("1st for loop")
    for lvl in range(num_levels):
        h_l, w_l = spatial_shapes_ttnn[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        len_l = h_l * w_l
        # Use ttnn slicing
        value_l = value_ttnn[:, start : start + len_l, :, :]
        value_list.append(value_l)
        start += len_l
    # ttnn.deallocate(value_ttnn)

    print("2nd for loop")
    # Normalize sampling locations to [-1, 1] using ttnn operations where possible
    sampling_grids = []
    # for lvl in range(num_levels):
    #     h_l, w_l = spatial_shapes[lvl]
    #     h_l = int(h_l.item())
    #     w_l = int(w_l.item())
    #     grid = sampling_locations_ttnn[:, :, :, lvl, :, :]

    #     grid = ttnn.clone(grid)
    #     grid_slice_0 = grid[..., 0]
    #     grid_slice_0 = ttnn.divide(grid_slice_0, w_l)
    #     grid_slice_0 = ttnn.multiply(grid_slice_0, 2)
    #     grid_slice_0 = ttnn.subtract(grid_slice_0, 1)
    #     original_shape = grid.shape
    #     # combine first two dimensions
    #     grid_4d = ttnn.reshape(grid, (original_shape[0] * original_shape[1], original_shape[2], original_shape[3], original_shape[4]))
    #     begins = [0, 0, 0, 0]
    #     ends = [grid_4d.shape[0], grid_4d.shape[1], grid_4d.shape[2], 1]
    #     strides = [1, 1, 1, 1]
    #     ttnn.slice_write(grid_slice_0, grid_4d, begins, ends, strides)
    #     ttnn.deallocate(grid_slice_0)
    #     grid = ttnn.reshape(grid_4d, original_shape)
    #     print("slice assignment done")
    #     grid_slice_1 = grid[..., 1]
    #     grid_slice_1 = ttnn.divide(grid_slice_1, h_l)
    #     grid_slice_1 = ttnn.multiply(grid_slice_1, 2)
    #     grid_slice_1 = ttnn.subtract(grid_slice_1, 1)
    #     begins = [0, 0, 0, 1]
    #     ends = [grid.shape[0], grid.shape[1], grid.shape[2], 2]
    #     strides = [1, 1, 1, 1]
    #     ttnn.slice_write(grid_slice_1, grid, begins, ends, strides)
    #     ttnn.deallocate(grid_slice_1)
    #     print("slice assignment done")
    #     sampling_grids.append(grid)
    for lvl in range(num_levels):
        h_l, w_l = spatial_shapes_ttnn[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        grid = sampling_locations_ttnn[:, :, :, lvl, :, :]
        grid = ttnn.clone(grid)
        grid = ttnn.to_torch(grid)
        grid[..., 0] = grid[..., 0] / w_l * 2 - 1
        grid[..., 1] = grid[..., 1] / h_l * 2 - 1
        grid = ttnn.from_torch(grid, layout=ttnn.TILE_LAYOUT, device=device)
        sampling_grids.append(grid)
    # ttnn.deallocate(sampling_locations_ttnn)
    # for grid in sampling_grids:
    #     ttnn.reallocate(grid)

    print("3rd for loop")
    # Create output tensor using ttnn
    output = ttnn.zeros(
        (bs, num_queries, num_heads, head_dim),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for lvl in range(num_levels):
        h_l, w_l = spatial_shapes_ttnn[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        value_l = ttnn.permute(value_list[lvl], (0, 2, 3, 1))
        value_l = ttnn.reshape(value_l, (bs * num_heads, head_dim, h_l, w_l))
        grid = ttnn.permute(sampling_grids[lvl], (0, 2, 1, 3, 4))
        grid = ttnn.reshape(grid, (bs * num_heads, num_queries * num_points, 1, 2))
        value_l_torch = ttnn.to_torch(value_l)
        grid_torch = ttnn.to_torch(grid)
        sampled = F.grid_sample(value_l_torch, grid_torch, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled_ttnn = ttnn.from_torch(sampled, layout=ttnn.TILE_LAYOUT, device=device)
        sampled_view = ttnn.reshape(sampled_ttnn, (bs, num_heads, head_dim, num_queries, num_points))
        sampled_permuted = ttnn.permute(sampled_view, (0, 3, 1, 4, 2))
        attn = attention_weights_ttnn[:, :, :, lvl, :]
        attn = ttnn.unsqueeze(attn, dim=-1)
        weighted = ttnn.multiply(sampled_permuted, attn)
        level_output = ttnn.sum(weighted, dim=-2)
        output = ttnn.add(output, level_output)
    output_reshaped = ttnn.reshape(output, (bs, num_queries, num_heads * head_dim))
    return output_reshaped


def mcw_multi_scale_deformable_attn(
    value,
    value_spatial_shapes,
    level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step,
    device,
    ttnn_list,
):
    bs, num_keys, num_heads, head_dim = value.shape
    num_levels = value_spatial_shapes.shape[0]
    num_queries = sampling_locations.shape[1]
    num_points = sampling_locations.shape[4]

    # Split value into a list of tensors for each level
    value_list = []
    start = 0
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        len_l = h_l * w_l
        value_l = value[:, start : start + len_l, :, :]
        value_list.append(value_l)
        start += len_l

    # Normalize sampling locations to [-1, 1]
    sampling_grids = []
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        grid = sampling_locations[:, :, :, lvl, :, :]
        grid = ttnn.clone(grid)
        grid = ttnn.to_torch(grid)
        grid[..., 0] = grid[..., 0] / w_l * 2 - 1
        grid[..., 1] = grid[..., 1] / h_l * 2 - 1
        grid = ttnn.from_torch(grid, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        sampling_grids.append(grid)

    # Perform sampling and attention
    output = ttnn.zeros(
        [bs, num_queries, num_heads, head_dim], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        value_l = ttnn.permute(value_list[lvl], (0, 2, 3, 1))
        value_l = ttnn.reshape(value_l, (bs * num_heads, head_dim, h_l, w_l))
        grid = ttnn.permute(sampling_grids[lvl], (0, 2, 1, 3, 4))
        grid = ttnn.reshape(grid, (bs * num_heads, num_queries * num_points, 1, 2))
        value_l = ttnn.to_torch(value_l).to(dtype=torch.float)
        grid = ttnn.to_torch(grid).to(dtype=torch.float)
        sampled = F.grid_sample(value_l, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = ttnn.from_torch(sampled, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)
        sampled = ttnn.reshape(sampled, (bs, num_heads, head_dim, num_queries, num_points))
        sampled = ttnn.permute(sampled, (0, 3, 1, 4, 2))
        attn = attention_weights[:, :, :, lvl, :]
        attn = ttnn.unsqueeze(attn, dim=-1)
        output += ttnn.sum((sampled * attn), dim=-2)
    output = ttnn.reshape(output, (bs, num_queries, num_heads * head_dim))
    return output


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_multi_scale_deformable_attn_ttnn(device):
    """
    Test the ttnn implementation of multi-scale deformable attention against PyTorch reference.
    """
    torch.manual_seed(0)

    spatial_shapes = torch.tensor([[50, 50]], dtype=torch.long)
    total_keys = 2500

    # Create test tensors with exact expected shapes
    value = torch.randn(2, total_keys, 8, 32, dtype=torch.float32)
    level_start_index = torch.tensor([0])
    sampling_locations = torch.rand(2, total_keys, 8, 1, 4, 2, dtype=torch.float32)
    attention_weights = torch.rand(2, total_keys, 8, 1, 4, dtype=torch.float32)
    im2col_step = torch.tensor([64])

    # Normalize attention weights to sum to 1
    attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

    # Scale sampling locations to valid ranges
    for lvl in range(spatial_shapes.shape[0]):
        h, w = spatial_shapes[lvl].tolist()
        sampling_locations[:, :, :, lvl, :, 0] *= w
        sampling_locations[:, :, :, lvl, :, 1] *= h

    # Run PyTorch reference implementation
    torch_output = multi_scale_deformable_attn_pytorch(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
    )

    # Convert tensors to ttnn
    value_ttnn = ttnn.from_torch(value, layout=ttnn.TILE_LAYOUT, device=device)
    sampling_locations_ttnn = ttnn.from_torch(sampling_locations, layout=ttnn.TILE_LAYOUT, device=device)
    attention_weights_ttnn = ttnn.from_torch(attention_weights, layout=ttnn.TILE_LAYOUT, device=device)
    spatial_shapes_ttnn = ttnn.from_torch(spatial_shapes, layout=ttnn.TILE_LAYOUT)
    level_start_index_ttnn = ttnn.from_torch(level_start_index, layout=ttnn.TILE_LAYOUT, device=device)
    im2col_step_ttnn = ttnn.from_torch(im2col_step, layout=ttnn.TILE_LAYOUT, device=device)

    # Run ttnn implementation
    ttnn_output = multi_scale_deformable_attn_ttnn(
        value_ttnn,
        spatial_shapes_ttnn,
        level_start_index_ttnn,
        sampling_locations_ttnn,
        attention_weights_ttnn,
        im2col_step_ttnn,
        device,
    )

    # ttnn_output = mcw_multi_scale_deformable_attn(
    #     value_ttnn,
    #     spatial_shapes_ttnn,
    #     level_start_index_ttnn,
    #     sampling_locations_ttnn,
    #     attention_weights_ttnn,
    #     im2col_step_ttnn,
    #     device,
    #     ttnn_list
    # )

    # Convert ttnn output back to torch for comparison
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    # for i in range(len(ttnn_list)):
    #     print("\nIteration: ", i)
    #     print("torch_list[i].shape", torch_list[i].shape, "ttnn_list[i].shape", ttnn_list[i].shape)
    #     assert_with_pcc(torch_list[i], ttnn_list[i], 0.99)

    # Assert outputs match with reasonable tolerance
    assert_with_pcc(torch_output, ttnn_output_torch, 0.95)

    # Additional shape checks
    assert (
        torch_output.shape == ttnn_output_torch.shape
    ), f"Shape mismatch: torch {torch_output.shape} vs ttnn {ttnn_output_torch.shape}"
