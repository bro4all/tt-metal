#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive examples of TT-NN sharding APIs for tensor creation.

This test file demonstrates:
1. Multi-device sharding APIs (ShardTensorToMesh, ReplicateTensorToMesh)
2. Single-device sharding APIs (Height, Width, Block sharding)
3. Usage with ttnn.distribute context manager
4. Practical model examples
"""

import pytest
import torch
import ttnn
from typing import Optional


class TestMultiDeviceShardingAPIs:
    """Examples of multi-device sharding APIs for model parallelism."""

    @pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4_grid")], indirect=True)
    def test_data_parallel_sharding(self, mesh_device):
        """Data parallel: shard input along batch dimension, replicate weights."""
        batch_size, sequence_length, hidden_size = 4, 128, 1024
        torch.manual_seed(0)

        # Create sample data
        torch_hidden_states = torch.rand(batch_size, 1, sequence_length, hidden_size, dtype=torch.float32)
        torch_weights = torch.rand(hidden_size, hidden_size * 4, dtype=torch.float32)

        # Method 2: Direct mesh_mapper specification
        shard_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

        hidden_states_direct = ttnn.from_torch(
            torch_hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=shard_mapper,
        )

        weights_direct = ttnn.from_torch(
            torch_weights,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicate_mapper,
        )

    @pytest.mark.parametrize("mesh_device", [pytest.param((1, 8), id="1x8_grid")], indirect=True)
    def test_tensor_parallel_sharding(self, mesh_device):
        """Tensor parallel: shard weights along width dimension."""
        batch_size, sequence_length, hidden_size = 1, 256, 4096
        torch.manual_seed(42)

        torch_hidden_states = torch.rand(batch_size, 1, sequence_length, hidden_size, dtype=torch.float32)
        torch_weights = torch.rand(hidden_size, hidden_size * 4, dtype=torch.float32)

        # Replicate inputs across all devices
        with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
            hidden_states = ttnn.from_torch(
                torch_hidden_states,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )

        # Shard weights along width dimension (last dimension)
        with ttnn.distribute(ttnn.ShardTensorToMesh(mesh_device, dim=-1)):
            weights = ttnn.from_torch(
                torch_weights,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )

        # Perform tensor parallel linear operation
        output = ttnn.linear(hidden_states, weights)

        # Collect results using concatenation
        with ttnn.distribute(ttnn.ConcatMeshToTensor(mesh_device, dim=-1)):
            result = ttnn.to_torch(output)

        # Verify output shape is correct
        expected_shape = (batch_size, 1, sequence_length, hidden_size * 4)
        assert result.shape == expected_shape

    @pytest.mark.parametrize("mesh_device", [pytest.param((2, 2), id="2x2_grid")], indirect=True)
    def test_2d_mesh_sharding(self, mesh_device):
        print(f"mesh_device shape: {mesh_device.shape}")
        """Advanced 2D mesh sharding with custom placements."""
        batch_size, height, width, channels = 2, 64, 64, 256
        torch.manual_seed(123)

        torch_tensor = torch.rand(batch_size, height, width, channels, dtype=torch.float32)

        # Create custom mesh mapper configuration
        config = ttnn.MeshMapperConfig(
            placements=[
                ttnn.PlacementReplicate(),  # Replicate along first mesh dimension
                ttnn.PlacementShard(3),  # Shard along channels dimension
            ],
            mesh_shape_override=ttnn.MeshShape(2, 2),
        )

        mesh_mapper = ttnn.create_mesh_mapper(mesh_device, config)

        distributed_tensor = ttnn.from_torch(
            torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mesh_mapper
        )

        print(distributed_tensor.device().shape)
        mesh_device.reshape(ttnn.MeshShape(1, 4))
        print(distributed_tensor.device().shape)


class TestSingleDeviceShardingAPIs:
    """Examples of single-device sharding APIs for memory optimization."""

    def test_height_sharding_strategy(self, device):
        """Height sharding for computer vision workloads."""
        batch_size, channels, height, width = 1, 16, 560, 640
        torch.manual_seed(0)

        torch_input = torch.rand(batch_size, channels, height, width, dtype=torch.float32)

        # Create height-sharded memory configuration
        input_mem_config = ttnn.create_sharded_memory_config(
            [batch_size, channels, height, width],
            ttnn.CoreGrid(x=8, y=7),
            ttnn.ShardStrategy.HEIGHT,
        )

        sharded_tensor = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_mem_config,
        )

        # Verify tensor is sharded
        assert sharded_tensor.is_sharded()
        assert sharded_tensor.memory_config().is_sharded()

        # Convert back and verify data integrity
        result = ttnn.to_torch(sharded_tensor)
        assert result.shape == torch_input.shape

    def test_width_sharding_strategy(self, device):
        """Width sharding for specific tensor layouts."""
        batch_size, height, width = 1, 32, 128
        torch.manual_seed(42)

        torch_tensor = torch.rand(batch_size, height, width, dtype=torch.float32)

        # Create width-sharded memory configuration
        memory_config = ttnn.create_sharded_memory_config(
            [batch_size, height, width],
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )

        sharded_tensor = ttnn.from_torch(
            torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )

        assert sharded_tensor.is_sharded()

    def test_block_sharding_strategy(self, device):
        """Block sharding for 2D tensor distribution."""
        batch_size, height, width = 1, 128, 128
        torch.manual_seed(123)

        torch_tensor = torch.rand(batch_size, height, width, dtype=torch.float32)

        # Create block-sharded memory configuration with 2x2 core grid
        sharded_memory_config = ttnn.create_sharded_memory_config(
            [batch_size, height, width],
            core_grid=ttnn.CoreGrid(y=2, x=2),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )

        sharded_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=sharded_memory_config,
        )

        assert sharded_tensor.is_sharded()
        assert sharded_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED

    def test_custom_tensorspec_sharding(self, device):
        """Advanced single-device sharding using TensorSpec."""
        shape = (2, 3, 64, 96)
        torch.manual_seed(456)

        torch_tensor = torch.rand(shape, dtype=torch.float32)

        # Create custom core range set
        core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 5))})

        # Create tensor spec with custom N-D sharding
        tensor_spec = ttnn.TensorSpec(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).sharded(
            ttnn.NdShardSpec(
                [1, 1, 32, 32],  # shard shape
                core_ranges,
                ttnn.ShardOrientation.ROW_MAJOR,
                ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
            )
        )

        sharded_tensor = ttnn.from_torch(torch_tensor, spec=tensor_spec, device=device)

        assert sharded_tensor.spec == tensor_spec
        assert sharded_tensor.is_sharded()
