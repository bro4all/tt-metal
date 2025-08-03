# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from models.experimental.uniad.reference.detection import Detection

from models.experimental.uniad.tt.ttnn_detection import TtDetection
from models.experimental.uniad.tt.model_preprocessing import create_uniad_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_detection(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"
    torch_model = Detection()
    torch_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    torch_dict = torch_dict["state_dict"]

    state_dict = {k: v for k, v in torch_dict.items() if k.startswith("img_backbone") or k.startswith("img_neck")}

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input = torch.randn(6, 3, 640, 360)
    torch_output = torch_model(torch_input)
    parameters = {}
    parameters = create_uniad_model_parameters(torch_model, torch_input)

    ttnn_model = TtDetection(device=device, parameters=parameters)

    torch_input_permute = torch_input.permute(0, 2, 3, 1)
    torch_input_permute = torch_input_permute.reshape(
        1,
        1,
        (torch_input_permute.shape[0] * torch_input_permute.shape[1] * torch_input_permute.shape[2]),
        torch_input_permute.shape[3],
    )
    ttnn_input = ttnn.from_torch(torch_input_permute, device=device, dtype=ttnn.bfloat16)

    ttnn_output = ttnn_model(ttnn_input)

    for i in range(4):
        ttnn_output_final = ttnn.to_torch(ttnn_output[i])

        ttnn_output_final = torch.reshape(
            ttnn_output_final,
            (torch_output[i].shape[0], torch_output[i].shape[2], torch_output[i].shape[3], torch_output[i].shape[1]),
        )
        ttnn_output_final = torch.permute(ttnn_output_final, (0, 3, 1, 2))

        _, x = assert_with_pcc(torch_output[i], ttnn_output_final, 0)
        logger.info(f"{x}")
