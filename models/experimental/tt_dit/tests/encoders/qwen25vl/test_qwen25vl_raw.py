# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import ttnn
from loguru import logger
from models.demos.qwen25_vl.tt.common import multimodal_rope_from_hf, preprocess_inputs_prefill
from models.demos.qwen25_vl.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs
from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize("prompts", [["Neon-lit cyberpunk alley, rain-soaked, cinematic wide shot"]])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 31000000}],
    indirect=True,
)
def test_qwen25vl(
    mesh_device: ttnn.MeshDevice,
    prompts: list[str],
) -> None:
    checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"
    max_sequence_length = 512

    template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompts = [template.format(e) for e in prompts]

    tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint)
    pad_token_id = tokenizer.pad_token_id

    torch_model, tt_model, model_args = _load_model(
        checkpoint,
        device=mesh_device,
        max_batch_size=len(prompts),
        max_sequence_length=max_sequence_length,
    )

    tokenizer_out = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
        max_length=max_sequence_length,
        truncation=True,
    )

    _, seq_len = tokenizer_out.input_ids.shape

    logger.info("running torch model...")
    with torch.no_grad():
        with torch.no_grad():
            output = torch_model.forward(
                tokenizer_out.input_ids, attention_mask=tokenizer_out.attention_mask, output_hidden_states=True
            )
        hidden_states = output.hidden_states[-1].to("cpu")

    logger.info("running ttnn model...")
    input_embeds = torch_model.model.language_model.embed_tokens(tokenizer_out.input_ids)
    pad_embedding = torch_model.model.language_model.embed_tokens(torch.tensor(pad_token_id))

    input_prefill_pt, _decoding_pos, _prefill_lens = preprocess_inputs_prefill(
        input_embeds,
        model_args,
        tokenizer_out.attention_mask,
        pad_embedding=pad_embedding,
    )

    rope = multimodal_rope_from_hf(tokenizer_out, input_embeds, torch_model, model_args, pad_token_id=pad_token_id)

    prefill_input, rot_mats_prefill, page_table_tt, _ = tt_model.prepare_inputs_prefill(input_prefill_pt, rot_mats=rope)

    tt_hidden_states = tt_model.ttnn_prefill_forward(
        prefill_input,
        rot_mats_global=rot_mats_prefill,
        page_table=page_table_tt,
    )
    tt_hidden_states = tt_model.norm(tt_hidden_states, mode="prefill")

    tt_hidden_states_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_hidden_states)[0])
    tt_hidden_states_torch = tt_hidden_states_torch[:, :, :seq_len, :].squeeze(1)

    assert_with_pcc(hidden_states, tt_hidden_states_torch, pcc=0.98)


def _load_model(
    checkpoint: str,
    *,
    device: ttnn.MeshDevice,
    max_batch_size: int,
    max_sequence_length: int,
) -> tuple[Qwen2_5_VLForConditionalGeneration, Transformer, ModelArgs]:
    os.environ["HF_MODEL"] = checkpoint
    model_args = ModelArgs(
        device,
        instruct=True,
        max_batch_size=max_batch_size,
        optimizations=lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
        max_seq_len=max_sequence_length,
        cache_hf=True,
    )

    state_dict = model_args.load_state_dict()
    torch_model = model_args.cached_hf_model
    assert isinstance(torch_model, Qwen2_5_VLForConditionalGeneration)

    dtype = ttnn.bfloat8_b

    model = Transformer(
        args=model_args,
        mesh_device=device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )

    return torch_model, model, model_args
