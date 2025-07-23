# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from tracy import signpost
from typing_extensions import override

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen25_vl.reference.functional import qwen2_5_vision_transformer_preprocess
from models.demos.qwen25_vl.tt.attention import Attention as QwenVLAttentionModule
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.demos.qwen25_vl.tt.patch_merger import PatchMerger
from models.demos.qwen25_vl.tt.rope import RotarySetup
from models.demos.qwen25_vl.tt.vision_block import VisionBlock
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import (
    convert_hf_to_meta,
    convert_rope_style_hf_to_meta,
    standardize_hf_keys,
)
from models.tt_transformers.tt.model import Transformer as TTTransformer
from models.utility_functions import comp_pcc


def _create_attention_mask(cu_seqlens_now, seq_len, mesh_device):
    """
    Creates a windowed attention mask.
    """
    num_sequences = len(cu_seqlens_now) - 1
    assert num_sequences > 0, f"num_sequences is {num_sequences} for {cu_seqlens_now}"
    # Create a matrix where each row is a one-hot vector indicating sequence membership
    sequence_indicators = torch.zeros(seq_len, num_sequences, dtype=torch.bfloat16)
    for j in range(num_sequences):
        start, end = cu_seqlens_now[j], cu_seqlens_now[j + 1]
        sequence_indicators[start:end, j] = 1.0
    # Create a block-diagonal mask via matrix multiplication
    # (A @ A.T) is 1 if two tokens are in the same sequence, 0 otherwise
    tt_sequence_indicators = ttnn.from_torch(
        sequence_indicators, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    tt_binary_mask = ttnn.matmul(
        tt_sequence_indicators, ttnn.transpose(tt_sequence_indicators, 0, 1), dtype=ttnn.bfloat4_b
    )
    # Convert binary mask to the required attention mask format
    tt_attention_mask_windowed_att = (ttnn.ones_like(tt_binary_mask) - tt_binary_mask) * -1e9
    ttnn.deallocate(tt_sequence_indicators)
    ttnn.deallocate(tt_binary_mask)

    tt_attention_mask_windowed_att = ttnn.unsqueeze_to_4D(tt_attention_mask_windowed_att)
    return tt_attention_mask_windowed_att


class VisionTransformer(LightweightModule):
    """
    Vision Transformer model for Qwen 2.5 VL.
    This implements only the transformer blocks part of the vision transformer.
    Patch embedding and merging should be done outside this class.
    """

    def __init__(
        self,
        args,
        dtype,
        state_dict,
        weight_cache_path,
    ):
        """
        Initialize the Vision Transformer model.

        Args:
            args (VisionModelArgs): Model arguments
            dtype (ttnn.dtype): Data type for computations
            mesh_device (ttnn.mesh_device): Mesh device for the model
            state_dict (dict): State dictionary containing model weights
            weight_cache_path (str): Path to weight cache
        """
        super().__init__()
        self.args = args
        self.dtype = dtype
        self.weight_cache_path = weight_cache_path
        self.fullatt_block_indexes = args.hf_config.vision_config.fullatt_block_indexes

        # Create transformation matrix for RoPE QK prefill
        transformation_mat_torch = get_rot_transformation_mat(
            args.head_dim
        )  # todo)) args.head_dim is ignored inside the function
        self.transformation_mats = {
            "prefill": ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=args.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(args.mesh_device),
            )
        }

        # Create vision blocks
        self.blocks = []
        for i in range(args.hf_config.vision_config.depth):
            block = VisionBlock(
                mesh_device=args.mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                transformation_mats=self.transformation_mats,
                args=args,
            )
            self.blocks.append(block)

        self.patch_merger = PatchMerger(
            mesh_device=args.mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    def prepare_input(self, patch_input, window_index, seq_len=None):
        """Convert a patchified torch input to a ttnn tensor
        Args:
            patch_input (torch.Tensor): Patchified input tensor
            window_index (torch.Tensor): Window index tensor

        Returns:
            ttnn.Tensor: Prepared input tensor
        """
        patch_seq_len, _ = patch_input.shape
        spatial_merge_unit = self.args.hf_config.vision_config.spatial_merge_size**2
        x = patch_input.reshape(patch_seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(patch_seq_len, -1)
        seq_len = ((patch_seq_len // 128) + 1) * 128 if seq_len is None else seq_len
        x = torch.nn.functional.pad(x, (0, 0, 0, seq_len - patch_seq_len)).unsqueeze(0)
        x = self.args.prepare_residual_tensor_prefill(
            x,
            force_replicated=False if self.args.is_galaxy else True,
        )
        return x

    def forward(
        self,
        x,
        unpadded_seq_len,
        rot_mats,
        cu_seqlens,
        cu_window_seqlens,
        profiler=None,
    ):
        """
        Forward pass through the Vision Transformer blocks.

        Args:
            x (ttnn.Tensor): Input tensor [batch_size, 1, seq_len, hidden_dim]
            cu_seqlens (torch.Tensor): Cumulative sequence lengths
            cu_window_seqlens (torch.Tensor): Cumulative window sequence lengths
            rot_mats (list): Rotation matrices for positional embeddings

        Returns:
            ttnn.Tensor: Output tensor
        """
        # Forward through each block
        for i, block in enumerate(self.blocks):
            # Determine which attention type to use (full or windowed)
            if i in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            # Forward through block
            x = block(
                x,
                cu_seqlens=cu_seqlens_now,
                rot_mats=rot_mats,
            )

        # Merge patches - first remove any sequence length padding
        x = x[:, :, :unpadded_seq_len, :]
        x = self.patch_merger(x)

        return x


class DropInVisionTransformer(torch.nn.Module):
    """Wraps VisionTransformer to be a drop-in replacement for
    Qwen2_5_VisionTransformerPretrainedModel. It uses the reference model
    for certain preprocessing steps like patch embedding and index calculation.
    """

    def __init__(
        self,
        reference_model,
        model_args: VisionModelArgs,
        dtype=ttnn.bfloat8_b,
        debug=False,
    ):
        """
        Initialize the TorchVisionTransformer wrapper.

        Args:
            tt_model (VisionTransformer): Initialized TT VisionTransformer instance.
            reference_model (Qwen2_5_VisionTransformerPretrainedModel): Initialized reference HF model instance.
            model_args (VisionModelArgs): Model configuration arguments.
            mesh_device (ttnn.MeshDevice): The mesh device used by the TT model.
        """
        super().__init__()
        self.reference_model = reference_model
        self.model_args = model_args
        self.debug = debug

        # FIXME: state_dict = model_args.load_state_dict()
        state_dict = standardize_hf_keys(reference_model.state_dict())
        state_dict = convert_hf_to_meta(state_dict, model_args.head_dim)
        state_dict_prefix = model_args.get_state_dict_prefix("VisionTransformer")
        state_dict = {f"{state_dict_prefix}.{k}": v for k, v in state_dict.items()}

        # Initialize TT model
        self.tt_model = VisionTransformer(
            args=model_args,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            dtype=dtype,
        )

    @property
    def dtype(self):
        return self.reference_model.dtype

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, profiler=None) -> torch.Tensor:
        """
        Forward pass mimicking the Qwen2_5_VisionTransformerPretrainedModel interface.

        Args:
            pixel_values (torch.Tensor): Input pixel values tensor (equivalent to hidden_states for the ref model start).
                                         Shape typically [num_patches, hidden_size] or similar before patch_embed.
            grid_thw (torch.Tensor): Tensor describing the grid dimensions (time, height, width) for each image/video.
                                     Shape [num_images_or_videos, 3].

        Returns:
            torch.Tensor: Output tensor matching the reference model's output shape [total_seq_len, out_hidden_size].
        """
        # process pixel_values for each image/video separately
        all_pixel_values = pixel_values
        all_grid_thw = grid_thw
        final_outputs = []
        # todo)) refactor this code to leverage tt-mesh's ttnn.ShardTensorToMesh(mesh_device, dim=batch_size_dim) for data parallelism
        # two main ideas for perf opt:
        # - [x] hoist the attention mask creation out of the loop using 43008 (2048*21) as seq_len --> does saved dynamic graph compilation time
        # - [ ] understand the cu_window_seqlens deeper --> maybe every image has the same cu_window_seqlens now with 300 DPI target image
        # - [x] understand the attention mask usage deeper
        # - [ ] make a sdpa kernel for qwen2.5 vl --> pass in cu_window_seqlens/cu_seqlens instead of attention_mask --> 300 DPI takes too long to copy attention_mask to device
        # - [ ] tensor parallel the vision model (seq_len dimension) --> >10 seconds to run tt_model.forward() for 300 DPI

        # [INFO] 300 DPI scanned doc with Letter paper (8.5x11 inches) has resolution around 2550x3300
        # Calculate padded sequence length (divisible by 2048) required by models/tt_transformers/tt/attention.py::forward_prefill
        target_seq_len = (
            (2550 // self.model_args.hf_config.vision_config.patch_size)
            * (3300 // self.model_args.hf_config.vision_config.patch_size)
            // 2048
            + 1
        ) * 2048
        num_iters = 0
        signpost("dropin_vision_transformer_forward", "start")
        for grid_thw in all_grid_thw:
            if profiler is not None:
                profiler.start(f"vision_model_loop_preprocess", iteration=num_iters)
            # --- pick out the pixel_values for this users' images (grid_thw.prod() pixels) ---
            pixel_values = all_pixel_values[: grid_thw.prod(), :]
            all_pixel_values = all_pixel_values[grid_thw.prod() :, :]
            # --- Preprocessing ---
            # 1. Calculate total unpadded sequence length
            grid_thw = grid_thw.unsqueeze(0)
            unpadded_seq_len = (grid_thw[:, 1] * grid_thw[:, 2]).sum().item()
            assert (
                unpadded_seq_len <= target_seq_len
            ), f"Not supported: unpadded_seq_len {unpadded_seq_len} is greater than target_seq_len {target_seq_len}"

            # 2. Use preprocessing function from reference/functional to get indices and embeddings
            cu_seqlens, cu_window_seqlens, position_embeddings, window_index = qwen2_5_vision_transformer_preprocess(
                seq_len=unpadded_seq_len,
                grid_thw=grid_thw,
                head_dim=self.model_args.head_dim,
                spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
                window_size=self.model_args.hf_config.vision_config.window_size,
                patch_size=self.model_args.hf_config.vision_config.patch_size,
            )

            # 3. Use reference model's patch embedding
            patch_input = self.reference_model.patch_embed(pixel_values)

            # 4. Prepare rotational embeddings (cos, sin) -> pad -> convert to TT tensors
            cos_orig, sin_orig = position_embeddings
            cos_orig, sin_orig = convert_rope_style_hf_to_meta(cos_orig, sin_orig)
            # pad sequence length with cos = 1, sin = 0 (identity rotation)
            cos_padded = (
                torch.nn.functional.pad(cos_orig, (0, 0, 0, target_seq_len - unpadded_seq_len), value=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            sin_padded = (
                torch.nn.functional.pad(sin_orig, (0, 0, 0, target_seq_len - unpadded_seq_len), value=0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            # Convert to TT tensors on the mesh device
            cos = ttnn.from_torch(
                cos_padded,
                dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                # mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
                # todo)) refactor this code to make the intent clear, which is data parallelism
                mesh_mapper=ttnn.ShardTensorToMesh(self.model_args.mesh_device, dim=0),
            )
            sin = ttnn.from_torch(
                sin_padded,
                dtype=ttnn.bfloat16,  # Use bfloat16 for RoPE
                layout=ttnn.TILE_LAYOUT,
                device=self.model_args.mesh_device,
                # mesh_mapper=ttnn.ReplicateTensorToMesh(self.model_args.mesh_device),
                # todo)) refactor this code to make the intent clear, which is data parallelism
                mesh_mapper=ttnn.ShardTensorToMesh(self.model_args.mesh_device, dim=0),
            )
            rot_mats = [cos, sin]

            # 5. Prepare input tensor for the TT model using window_index
            tt_input = self.tt_model.prepare_input(patch_input, window_index, target_seq_len)

            # --- TT Model Execution ---
            if profiler is not None:
                profiler.end(f"vision_model_loop_preprocess", iteration=num_iters)
                profiler.start(f"vision_model_loop_tt_model", iteration=num_iters)
            tt_out = self.tt_model(
                tt_input,
                unpadded_seq_len=unpadded_seq_len,
                rot_mats=rot_mats,  # Use rot_mats generated in this forward pass
                cu_seqlens=ttnn.from_torch(
                    cu_seqlens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.model_args.mesh_device
                ),
                cu_window_seqlens=ttnn.from_torch(
                    cu_window_seqlens,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.model_args.mesh_device,
                ),
                profiler=profiler,
            )
            if profiler is not None:
                profiler.end(f"vision_model_loop_tt_model", iteration=num_iters)
                profiler.start(f"vision_model_loop_postprocess", iteration=num_iters)

            # deallocate device tensors that are not needed by decode
            ttnn.deallocate(tt_input)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            ttnn.deallocate(rot_mats[0])
            ttnn.deallocate(rot_mats[1])

            # --- Postprocessing ---
            # 1. Convert TT output back to torch tensor
            tt_output_torch = ttnn.to_torch(
                tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.model_args.mesh_device, dim=1)
            )

            # deallocate TT output
            ttnn.deallocate(tt_out)

            # 2. Extract the relevant output part and adjust shape (matching test logic)
            out_hidden_size = self.model_args.hf_config.vision_config.out_hidden_size
            # Output shape from TT is [1, B=1, S, H_out_padded], slice H and squeeze B, batch dims
            tt_output_torch = tt_output_torch[:, 0:1, :, :out_hidden_size].squeeze(0).squeeze(0)

            # 3. Apply reverse window indexing to match reference model output order
            reverse_indices = torch.argsort(window_index)
            final_output = tt_output_torch[reverse_indices, :]

            if self.debug:
                logger.info(f"DropInVisionTransformer: Debug enabled, running reference model...")
                reference_output = self.reference_model.forward(pixel_values, grid_thw)
                _, pcc = comp_pcc(reference_output, final_output)
                logger.info(f"DropInVisionTransformer: PCC to reference model: {pcc}")

            final_outputs.append(final_output)

            if profiler is not None:
                profiler.end(f"vision_model_loop_postprocess", iteration=num_iters)
                num_iters += 1

        signpost("dropin_vision_transformer_forward", "end")

        if profiler is not None:
            # print the total time for each iteration
            for i in range(num_iters):
                logger.info(
                    f"vision_model_loop_preprocess at {i}: {profiler.get_duration('vision_model_loop_preprocess', iteration=i)}"
                )
                logger.info(
                    f"vision_model_loop_tt_model at {i}: {profiler.get_duration('vision_model_loop_tt_model', iteration=i)}"
                )
                logger.info(
                    f"vision_model_loop_postprocess at {i}: {profiler.get_duration('vision_model_loop_postprocess', iteration=i)}"
                )

        # concatenate all the outputs
        return torch.cat(final_outputs, dim=0)


class Transformer(TTTransformer):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        # Call parent constructor with vision-specific classes
        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            attention_class=QwenVLAttentionModule,
            rope_setup_class=RotarySetup,
        )

    @override
    def prepare_inputs_prefill(self, tokens, start_pos=0, page_table=None, chunk_page_table=None):
        # tokens is actually embeddings
        assert tokens.dim() == 3, "tokens should be a 3D tensor"
        S = tokens.shape[-2]
        tokens_embd = ttnn.from_torch(
            tokens.unsqueeze(1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device, dims=(None, 3), mesh_shape=self.args.cluster_shape
            ),
        )

        # Slice the rot mats to the prefill seqlen
        assert (
            self.rope_setup.cos_matrix.shape[2] >= start_pos + S
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix.shape[2]}"
        tt_rot_mats_prefill = [
            self.rope_setup.cos_matrix[:, :, start_pos : start_pos + S, :],
            self.rope_setup.sin_matrix[:, :, start_pos : start_pos + S, :],
        ]

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats_prefill, tt_page_table, tt_chunk_page_table
