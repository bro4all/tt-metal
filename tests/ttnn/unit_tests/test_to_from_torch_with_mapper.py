# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import time

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_to_from_torch_(device):
    num_devices = device.get_num_devices()

    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        output_mesh_composer = None

    torch_input_tensor = torch.rand((2, 3, 640, 32 * num_devices), dtype=torch.bfloat16)
    t0 = time.time()
    ttnn_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=inputs_mesh_mapper,
    )
    t1 = time.time()
    torch_output = ttnn.to_torch(
        ttnn_tensor,
        mesh_composer=output_mesh_composer,
    )
    t2 = time.time()

    logger.info(f"Time taken for from_torch : {(t1 - t0):.6f} seconds")
    logger.info(f"Time taken for to_torch : {(t2 - t1):.6f} seconds")

    assert_with_pcc(torch_output, torch_input_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
def test_single_device_to_and_from_torch(device):
    run_to_from_torch_(device)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
def test_multi_device_to_and_from_torch(mesh_device):
    run_to_from_torch_(mesh_device)
