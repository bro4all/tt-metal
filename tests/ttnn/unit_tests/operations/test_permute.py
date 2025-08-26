# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import torch

import ttnn
import itertools

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal, comp_pcc
from models.utility_functions import is_blackhole, skip_for_wormhole_b0

# @pytest.mark.parametrize("shape", [(3, 65, 3, 3, 65), (1, 6, 256, 20, 50), (6, 20, 50, 1, 256)])
# @pytest.mark.parametrize("perm", [(4, 0, 3, 2, 1), (1, 3, 4, 0, 2), (3, 0, 4, 1, 2)])
# @pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
# @pytest.mark.parametrize("shape", [(3, 65, 3, 3, 65)])
# @pytest.mark.parametrize("shape", [(3, 33, 3, 3, 33)])
# @pytest.mark.parametrize("shape", [(3, 33, 3, 2, 33)]) # Most efficient so far
# @pytest.mark.parametrize("shape", [(3, 33, 2, 2, 33)])
# @pytest.mark.parametrize("shape", [(1, 33, 1, 1, 31)])
# pytest.mark.parametrize("perm", [(4, 0, 3, 2, 1)])


@pytest.mark.parametrize("shape", [(1, 1, 1, 1, 63)])
@pytest.mark.parametrize("perm", [(1, 3, 4, 0, 2)])
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
    ],
)
def test_permute_5d_blocked(device, shape, perm, memory_config, dtype):
    magic_num = 0.0
    input_a = torch.full(shape, magic_num, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, memory_config=memory_config
    )

    tt_output = ttnn.permute(tt_input, perm)
