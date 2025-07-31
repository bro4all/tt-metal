# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3DecoderLayer
from models.demos.deepseek_v3.tt.decoder_block.decoder_block import DecoderBlock
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    load_reference_io_tensors_for_module,
    load_state_dict,
)
from models.tt_transformers.tt.common import PagedAttentionConfig


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "DecoderBlockClass,module_path,reference_layer_idx",
    [
        (DecoderBlock, None, 0),
        # (MoEDecoderBlock, None, 3), # TODO: Uncomment once PCC is fixed for MoE
        (DecoderBlock, "model.layers.0", None),
        # (MoEDecoderBlock, "model.layers.3", None), # TODO: Uncomment once PCC is fixed for MoE
    ],
)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 32),
        # ("prefill", 512), # TODO: Uncomment once MLA prefill works
        # ("prefill", 2048),  # Test chunking # TODO: Uncomment once MLA prefill works
    ],
)
def test_forward_pass(
    DecoderBlockClass,
    module_path,
    reference_layer_idx,
    mode,
    prefill_seq_len,
    decode_batch_size,
    hf_config,
    tmp_path,
    mesh_device,
    model_path,
    ccl,
):
    num_module_layers, _ = mesh_device.shape

    # Get the reference IO
    if module_path is None:
        torch.set_default_dtype(torch.bfloat16)
        reference_model = DeepseekV3DecoderLayer(hf_config, layer_idx=reference_layer_idx).eval()
        state_dict = reference_model.state_dict()
        torch_input = torch.randn(num_module_layers, 1, prefill_seq_len + 1, hf_config.hidden_size)
        reference_output = reference_model(torch_input)
    else:
        state_dict = load_state_dict(model_path, module_path)
        torch_input, reference_output = load_reference_io_tensors_for_module(
            mode, module_path, prefill_seq_len + 1, num_module_layers
        )

    # Generate module configs and state
    is_padding_layer = [False] * num_module_layers
    state_dicts = [state_dict] * num_module_layers

    weight_config = DecoderBlockClass.convert_weights(hf_config, state_dicts, tmp_path, mesh_device)
    prefill_model_config = DecoderBlockClass.prefill_model_config(hf_config, mesh_device, is_padding_layer)
    decode_model_config = DecoderBlockClass.decode_model_config(hf_config, mesh_device, is_padding_layer)
    model_state = DecoderBlockClass.create_state(
        hf_config,
        mesh_device,
        is_padding_layer=[False] * num_module_layers,
        ccl=ccl,
        paged_config=PagedAttentionConfig(),
    )
    prefill_run_config = create_run_config(prefill_model_config, weight_config, model_state)
    decode_run_config = create_run_config(decode_model_config, weight_config, model_state)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, (0, -1)),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass
    if mode == "prefill":
        raise NotImplementedError("Prefill mode is not implemented for this test yet.")
    else:
        tt_output = DecoderBlockClass.forward_decode(
            tt_input, 0, position_idxs, rope_tensors, page_table, decode_run_config
        )

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Cleanup
    ttnn.deallocate(tt_input)


def check_output_and_dealloc(tt_output, reference_output, expected_memory_config):
    """Check the output tensor against the reference and deallocate."""
    assert (
        tt_output.memory_config() == expected_memory_config
    ), f"Output tensor memory config {tt_output.memory_config()} does not match expected {expected_memory_config}"

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape)),
    )
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)
    ttnn.deallocate(tt_output_torch)


if __name__ == "__main__":
    pytest.main([__file__])
