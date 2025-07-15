# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.qwen3_moe.tt.ccl_1d import CCL1D
from models.demos.qwen3_moe.tt.rms_norm import RMSNorm
from models.demos.qwen3_moe.utils.abstract_module import AbstractModule
from models.demos.qwen3_moe.utils.config_dataclass import (
    AllGatherAsyncConfig,
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
    ReduceScatterAsyncConfig,
    ReshardConfig,
)
from models.demos.qwen3_moe.utils.config_helpers import save_and_get_path
from models.demos.qwen3_moe.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.utility_functions import nearest_y


class Attention1D(AbstractModule):
    """
    Multi-Latent Attention Module for 1D tensor parallelism.
    """

    MAX_BATCH_SIZE = ttnn.TILE_SIZE
    TG_GRID = (8, 4)

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        """Convert PyTorch weights to TTNN format for 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to their TTNN weight file paths
        """
        logger.info(f"Converting weights for Attention1D...state_dict keys: {list(state_dict.keys())}")
        weight_config = {}

        dim = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        head_dim = hf_config.head_dim

        def add_weight_config(
            torch_weight,
            our_name,
            kwarg_name,
            dtype,
            mem_config,
            layout,
            mesh_mapper,
        ):
            """Helper function to convert and save weights, updating weight_config."""
            ttnn_weight = ttnn.as_tensor(
                torch_weight,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=layout,
                memory_config=mem_config,
            )
            ttnn_weight = ttnn.unsqueeze_to_4D(ttnn_weight)
            logger.info(
                f"Converted {our_name} torch_weight.shape: {torch_weight.shape} -> ttnn_weight.shape: {ttnn_weight.shape}"
            )
            # Add to weight config
            weight_file_path = output_path / f"{our_name}.{kwarg_name}.weight"
            weight_config[our_name] = {kwarg_name: save_and_get_path(weight_file_path, ttnn_weight)}

        hf_ttnn_name_mapping = {
            "q_proj": "wq",
            "k_proj": "wk",
            "v_proj": "wv",
            "o_proj": "wo",
        }

        # wq
        hf_name = "q_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{hf_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[-2, None],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # wk
        hf_name = "k_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{hf_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[-2, None],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # wv
        hf_name = "v_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{hf_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[-2, None],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # wo
        hf_name = "o_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{hf_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[-1, None],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # Norm weights
        hf_name = "q_norm"
        our_name = "q_norm"
        q_norm_state_dict = {"weight": state_dict[f"{hf_name}.weight"]}
        weight_config["q_norm"] = RMSNorm.convert_weights(
            hf_config, q_norm_state_dict, output_path, mesh_device, norm_category="q_norm"
        )

        hf_name = "k_norm"
        our_name = "k_norm"
        k_norm_state_dict = {"weight": state_dict[f"{hf_name}.weight"]}
        weight_config["k_norm"] = RMSNorm.convert_weights(
            hf_config, k_norm_state_dict, output_path, mesh_device, norm_category="k_norm"
        )

        return weight_config

    @classmethod
    def prefill_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL1D
    ) -> ModelPrefillConfig:
        """Prefill model config for an MLP with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device
            ccl: CCL1D object for communication configuration

        Returns:
            Dict containing operator configurations for prefill mode
        """

        grid_size = mesh_device.compute_with_storage_grid_size()

        # Extract dimensions from HF config
        dim = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        num_key_value_heads = hf_config.num_key_value_heads
        head_dim = hf_config.head_dim
        num_key_value_groups = num_heads // num_key_value_heads

        max_seq_len = hf_config.max_seq_len

        mesh_shape = list(mesh_device.shape)
        num_heads_local = num_heads // mesh_shape[0]

        config: ModelPrefillConfig = {}

        config["hf_config"] = hf_config
        config["mesh_shape"] = mesh_shape

        config["wq"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wk"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wv"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wo"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        # Resharding for q_rope
        q_rope_shape = (1, Attention1D.MAX_BATCH_SIZE, num_heads_local, head_dim)
        q_rope_shard_height = nearest_y(q_rope_shape[2], ttnn.TILE_SIZE)
        q_rope_shard_width = q_rope_shape[3]
        q_rope_num_cores = q_rope_shape[1]
        q_rope_core_grid = ttnn.num_cores_to_corerangeset(q_rope_num_cores, grid_size, row_wise=True)
        q_rope_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(q_rope_shard_height, q_rope_shard_width),
            core_grid=q_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        config["q_rope_reshard"] = ReshardConfig(
            memory_config=q_rope_mem_cfg,
        )
        config["q_rope_out_reshard"] = ReshardConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Resharding for k_rope
        k_rope_shape = (1, Attention1D.MAX_BATCH_SIZE, 1, head_dim)
        k_rope_shard_height = nearest_y(k_rope_shape[2], ttnn.TILE_SIZE)
        k_rope_shard_width = k_rope_shape[3]
        k_rope_num_cores = k_rope_shape[1]
        k_rope_core_grid = ttnn.num_cores_to_corerangeset(k_rope_num_cores, grid_size, row_wise=True)
        k_rope_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(k_rope_shard_height, k_rope_shard_width),
            core_grid=k_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        config["k_rope_reshard"] = ReshardConfig(
            memory_config=k_rope_mem_cfg,
        )
        config["k_rope_out_reshard"] = ReshardConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Resharding for kv
        kv_shape = (1, Attention1D.MAX_BATCH_SIZE, 1, head_dim)
        kv_shard_height = nearest_y(kv_shape[2], ttnn.TILE_SIZE)
        kv_shard_width = kv_shape[3]
        kv_num_cores = kv_shape[1]
        kv_core_grid = ttnn.num_cores_to_corerangeset(kv_num_cores, grid_size, row_wise=True)
        kv_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(kv_shard_height, kv_shard_width),
            core_grid=kv_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        config["kv_reshard"] = ReshardConfig(
            memory_config=kv_mem_cfg,
        )

        # SDPA
        q_chunk_size = 0  # TODO: Make dynamic?
        k_chunk_size = 0  # TODO: Make dynamic?

        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        scale = head_dim**-0.5

        config["sdpa"] = {
            "scale": scale,
            "program_config": sdpa_program_config,
            "compute_kernel_config": sdpa_compute_kernel_config,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "attn_mask": None,
            "is_causal": True,
        }

        # Norms
        config["q_norm"] = RMSNorm.prefill_model_config(hf_config, mesh_device, norm_category="q_norm")
        config["k_norm"] = RMSNorm.prefill_model_config(hf_config, mesh_device, norm_category="k_norm")

        return config

    @classmethod
    def decode_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL1D
    ) -> ModelDecodeConfig:
        """Generate decode operator configuration for this MLP layer.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for decode mode
        """

        grid_size = mesh_device.compute_with_storage_grid_size()

        # Extract dimensions from HF config
        dim = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        num_key_value_heads = hf_config.num_key_value_heads
        head_dim = hf_config.head_dim
        num_key_value_groups = num_heads // num_key_value_heads

        max_seq_len = hf_config.max_seq_len

        mesh_shape = list(mesh_device.shape)
        num_heads_local = num_heads // mesh_shape[0]

        config: ModelDecodeConfig = {}

        config["hf_config"] = hf_config
        config["mesh_shape"] = mesh_shape

        config["wq"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wk"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wv"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wo"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        # Resharding for q_rope
        q_rope_shape = (1, Attention1D.MAX_BATCH_SIZE, num_heads_local, head_dim)
        q_rope_shard_height = nearest_y(q_rope_shape[2], ttnn.TILE_SIZE)
        q_rope_shard_width = q_rope_shape[3]
        q_rope_num_cores = q_rope_shape[1]
        q_rope_core_grid = ttnn.num_cores_to_corerangeset(q_rope_num_cores, grid_size, row_wise=True)
        q_rope_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(q_rope_shard_height, q_rope_shard_width),
            core_grid=q_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        config["q_rope_reshard"] = ReshardConfig(
            memory_config=q_rope_mem_cfg,
        )
        config["q_rope_out_reshard"] = ReshardConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Resharding for k_rope
        k_rope_shape = (1, Attention1D.MAX_BATCH_SIZE, 1, head_dim)
        k_rope_shard_height = nearest_y(k_rope_shape[2], ttnn.TILE_SIZE)
        k_rope_shard_width = k_rope_shape[3]
        k_rope_num_cores = k_rope_shape[1]
        k_rope_core_grid = ttnn.num_cores_to_corerangeset(k_rope_num_cores, grid_size, row_wise=True)
        k_rope_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(k_rope_shard_height, k_rope_shard_width),
            core_grid=k_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        config["k_rope_reshard"] = ReshardConfig(
            memory_config=k_rope_mem_cfg,
        )
        config["k_rope_out_reshard"] = ReshardConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Resharding for kv
        kv_shape = (1, Attention1D.MAX_BATCH_SIZE, 1, head_dim)
        kv_shard_height = nearest_y(kv_shape[2], ttnn.TILE_SIZE)
        kv_shard_width = kv_shape[3]
        kv_num_cores = kv_shape[1]
        kv_core_grid = ttnn.num_cores_to_corerangeset(kv_num_cores, grid_size, row_wise=True)
        kv_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(kv_shard_height, kv_shard_width),
            core_grid=kv_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        config["kv_reshard"] = ReshardConfig(
            memory_config=kv_mem_cfg,
        )

        # SDPA
        q_chunk_size = 0  # Unused in decode mode
        k_chunk_size = 0  # TODO: Make dynamic?

        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        q_num_cores = (
            Attention1D.MAX_BATCH_SIZE
        )  # TODO: How to use non-padded batch size here? (might need to be dynamic)
        block_height = nearest_y((Attention1D.MAX_BATCH_SIZE * num_heads_local) // q_num_cores, ttnn.TILE_SIZE)
        block_width = head_dim

        q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)
        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(block_height, block_width),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        sdpa_out_mem_config = ttnn.create_sharded_memory_config(
            shape=(block_height, head_dim),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        scale = head_dim**-0.5

        config["sdpa_reshard"] = ReshardConfig(
            memory_config=q_mem_config,
        )
        config["sdpa"] = {
            "scale": scale,
            "program_config": sdpa_program_config,
            "compute_kernel_config": sdpa_compute_kernel_config,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }
        config["sdpa_out_reshard"] = ReshardConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Norms
        config["q_norm"] = RMSNorm.decode_model_config(hf_config, mesh_device, norm_category="q_norm")
        config["k_norm"] = RMSNorm.decode_model_config(hf_config, mesh_device, norm_category="k_norm")

        # Set up CCLs
        # **Must be in order of execution**

        # Q
        config["wq_rs"] = ReduceScatterAsyncConfig(
            mesh_device=MeshDeviceStub(list(mesh_device.shape)),
            cluster_axis=1,
            dim=3,
            from_remote_multi_device_global_semaphore=ccl.get_semaphore(0),
            to_remote_multi_device_global_semaphore=ccl.get_semaphore(0),
            math_op=ttnn.ReduceType.Sum,
            num_links=ccl.get_max_links(0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        config["wq_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(list(mesh_device.shape)),
            cluster_axis=1,
            dim=3,
            multi_device_global_semaphore=ccl.get_semaphore(0),
            num_links=ccl.get_max_links(0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # K
        config["wk_rs"] = ReduceScatterAsyncConfig(
            mesh_device=MeshDeviceStub(list(mesh_device.shape)),
            cluster_axis=1,
            dim=3,
            from_remote_multi_device_global_semaphore=ccl.get_semaphore(0),
            to_remote_multi_device_global_semaphore=ccl.get_semaphore(0),
            math_op=ttnn.ReduceType.Sum,
            num_links=ccl.get_max_links(0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        config["wk_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(list(mesh_device.shape)),
            cluster_axis=1,
            dim=3,
            multi_device_global_semaphore=ccl.get_semaphore(0),
            num_links=ccl.get_max_links(0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # K
        config["wv_rs"] = ReduceScatterAsyncConfig(
            mesh_device=MeshDeviceStub(list(mesh_device.shape)),
            cluster_axis=1,
            dim=3,
            from_remote_multi_device_global_semaphore=ccl.get_semaphore(0),
            to_remote_multi_device_global_semaphore=ccl.get_semaphore(0),
            math_op=ttnn.ReduceType.Sum,
            num_links=ccl.get_max_links(0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        config["wv_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(list(mesh_device.shape)),
            cluster_axis=1,
            dim=3,
            multi_device_global_semaphore=ccl.get_semaphore(0),
            num_links=ccl.get_max_links(0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # KV
        config["wkv_a_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(list(mesh_device.shape)),
            cluster_axis=1,
            dim=1,
            multi_device_global_semaphore=ccl.get_semaphore(0),
            num_links=ccl.get_max_links(0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # WO
        config["wo_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(list(mesh_device.shape)),
            cluster_axis=1,
            dim=1,
            multi_device_global_semaphore=ccl.get_semaphore(0),
            num_links=ccl.get_max_links(0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        return config

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> Any:
        num_key_value_heads = hf_config.num_key_value_heads
        head_dim = hf_config.head_dim
        max_seq_len = hf_config.max_seq_len

        kv_dim = head_dim
        kv_cache_dtype = ttnn.bfloat8_b
        kv_cache_layout = ttnn.TILE_LAYOUT
        kv_cache_mem_config = ttnn.DRAM_MEMORY_CONFIG

        k_cache = torch.zeros(
            (
                Attention1D.MAX_BATCH_SIZE,  # TODO: Split batch when addign DP support
                num_key_value_heads,  # kv heads
                max_seq_len,
                head_dim,
            )
        )

        v_cache = torch.zeros(
            (
                Attention1D.MAX_BATCH_SIZE,  # TODO: Split batch when addign DP support
                num_key_value_heads,  # 1 kv heads
                max_seq_len,
                head_dim,
            )
        )

        tt_k_cache = ttnn.as_tensor(
            k_cache,
            dtype=kv_cache_dtype,
            layout=kv_cache_layout,
            device=mesh_device,
            memory_config=kv_cache_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            # TODO: Add caching
        )
        tt_v_cache = ttnn.as_tensor(
            v_cache,
            dtype=kv_cache_dtype,
            layout=kv_cache_layout,
            device=mesh_device,
            memory_config=kv_cache_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        return {"k_cache": tt_k_cache, "v_cache": tt_v_cache, MESH_DEVICE_STATE_DICT_KEY: mesh_device}

    @classmethod
    def forward_decode(
        self, x: ttnn.Tensor, position_idxs: [int], rope_tensors: dict, cfg: RunPrefillConfig
    ) -> ttnn.Tensor:
        """Forward pass of Attention1D in decode mode.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            position_idxs: List of position indices for the current batch
            rope_tensors: Dictionary containing RoPE tensors
            cfg: RunConfig containing weights and op configurations
        Returns:
            Output tensor after Attention1D computation

        """

        hf_config = cfg["hf_config"]
        num_heads = hf_config.num_attention_heads
        num_key_value_heads = hf_config.num_key_value_heads
        num_heads_local = num_heads // cfg["mesh_shape"][0]
        head_dim = hf_config.head_dim
        num_key_value_groups = num_heads // num_key_value_heads

        k_cache = cfg["k_cache"]
        v_cache = cfg["v_cache"]

        seq_len = x.shape[1]  # seq_len = 1
        bsz = x.shape[2]
        bsz_local = bsz // 1  # TODO: Use this when adding DP support (Attention1D.TG_GRID[0])

        logger.info(f"Input x shape: {x.shape}")
        # x : [1, seq_len, bsz, dim]
        # wq
        tt_q = ttnn.linear(x, **cfg["wq"])
        logger.info(f"tt_q shape: {tt_q.shape}, num_heads_local: {num_heads_local}, head_dim: {head_dim}")
        # tt_q = ttnn.experimental.reduce_scatter_async(tt_q, **cfg["wq_rs"])
        # tt_q = ttnn.experimental.all_gather_async(tt_q, **cfg["wq_ag"])
        tt_q = ttnn.reshape(tt_q, (seq_len, bsz, num_heads_local, head_dim))
        logger.info(f"tt_q reshaped1: {tt_q.shape}")
        tt_q = RMSNorm.forward_decode(tt_q, cfg["q_norm"])

        # q rope
        tt_q = ttnn.to_memory_config(tt_q, **cfg["q_rope_reshard"])
        tt_q = ttnn.experimental.rotary_embedding_llama(
            tt_q,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )
        tt_q = ttnn.to_memory_config(tt_q, **cfg["q_rope_out_reshard"])

        # wk
        tt_k = ttnn.linear(x, **cfg["wk"])
        logger.info(f"tt_k shape: {tt_k.shape}, num_key_value_heads: {num_key_value_heads}, head_dim: {head_dim}")
        # tt_k = ttnn.experimental.reduce_scatter_async(tt_k, **cfg["wk_rs"])
        # tt_k = ttnn.experimental.all_gather_async(tt_k, **cfg["wk_rs"])
        tt_k = ttnn.reshape(tt_k, (seq_len, bsz, num_key_value_heads, head_dim))
        logger.info(f"tt_k reshaped1: {tt_k.shape}")
        tt_k = RMSNorm.forward_decode(tt_k, cfg["k_norm"])
        # k rope
        tt_k = ttnn.to_memory_config(tt_k, **cfg["k_rope_reshard"])
        tt_k = ttnn.experimental.rotary_embedding_llama(
            tt_k,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )
        tt_k = ttnn.to_memory_config(tt_k, **cfg["k_rope_out_reshard"])
        logger.info(f"tt_k reshaped2: {tt_k.shape}")

        # wv
        tt_v = ttnn.linear(x, **cfg["wv"])
        logger.info(f"tt_v shape: {tt_v.shape}, num_key_value_heads: {num_key_value_heads}, head_dim: {head_dim}")
        # tt_v = ttnn.experimental.reduce_scatter_async(tt_v, **cfg["wv_rs"])
        # tt_v = ttnn.experimental.all_gather_async(tt_v, **cfg["wv_ag"])
        tt_v = ttnn.reshape(tt_v, (seq_len, bsz, num_key_value_heads, head_dim))
        logger.info(f"tt_v reshaped: {tt_v.shape}")

        # tt_k = ttnn.to_memory_config(tt_k, **cfg["kv_reshard"])
        # tt_v = ttnn.to_memory_config(tt_v, **cfg["kv_reshard"])
        logger.info(f"tt_q is sharded : {tt_q.is_sharded()}")
        logger.info(f"tt_k is sharded : {tt_k.is_sharded()}")
        logger.info(f"tt_v is sharded : {tt_v.is_sharded()}")

        # TODO: Update KV Cache
        # ttnn.experimental.paged_update_cache(
        #     k_cache,
        #     tt_k,
        #     update_idxs_tensor=position_idxs,
        # )
        # ttnn.experimental.paged_update_cache(
        #     v_cache,
        #     tt_v,
        #     update_idxs_tensor=position_idxs,
        # )

        # tt_q shape # [seq_len, bsz, num_heads_local, head_dim]
        # tt_k shape # [seq_len, bsz, num_key_value_heads, head_dim]
        # tt_v shape # [seq_len, bsz, num_key_value_heads, head_dim]
        tt_k = ttnn.repeat_interleave(tt_k, num_key_value_groups, 2)
        tt_v = ttnn.repeat_interleave(tt_v, num_key_value_groups, 2)
        tt_q = ttnn.permute(tt_q, (1, 2, 0, 3))
        tt_k = ttnn.permute(tt_k, (1, 2, 3, 0))
        tt_v = ttnn.permute(tt_v, (1, 2, 0, 3))
        attn_out = ttnn.matmul(tt_q, tt_k) * cfg["sdpa"]["scale"]
        attn_out = ttnn.softmax(attn_out, dim=-1)
        attn_out = ttnn.matmul(attn_out, tt_v)
        logger.info(f"attn_out shape after v matmul: {attn_out.shape}")
        attn_out = ttnn.transpose(attn_out, 1, 2)
        attn_out = ttnn.reshape(attn_out, (1, 1, bsz, num_heads_local * head_dim))

        # TODO: SDPA
        # tt_q = ttnn.to_memory_config(tt_q, **cfg["sdpa_reshard"])
        # attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
        #     tt_q,
        #     k_cache,
        #     v_cache,
        #     cur_pos_tensor=position_idxs,
        #     **cfg["sdpa"],
        # )  #  [1, bsz, num_heads_local, head_dim]
        # ttnn.deallocate(tt_q)
        # logger.info(f"attn_out shape: {attn_out.shape}")
        # attn_out = ttnn.to_memory_config(attn_out, **cfg["sdpa_out_reshard"])
        # # attn_out = ttnn.experimental.all_gather_async(attn_out, **cfg["wo_ag"])  # [1, num_heads, bsz, head_dim]
        # logger.info(f"attn_out sdpa_out_reshard shape: {attn_out.shape}")
        # attn_out = ttnn.reshape(attn_out, (1, 1, bsz, num_heads * head_dim))
        logger.info(f"attn_out shape: {attn_out.shape}")

        # wo
        out = ttnn.linear(attn_out, **cfg["wo"])  # [1, 1, bsz, dim]
        logger.info(f"out shape: {out.shape}")
        return out

    @classmethod
    def forward_prefill(self, x: ttnn.Tensor, user_id: int, rope_tensors: dict, cfg: RunPrefillConfig) -> ttnn.Tensor:
        """Forward pass of the MLP.

        Prefill mode we reshape to respect cfg["max_rows"] and generate program configs from the seq-len lambda.

        Args:
            x: Input tensor
            user_id: Batch index for cache updates
            rope_tensors: Dictionary containing RoPE tensors
            cfg: RunConfig containing weights and op configurations

        Returns:
            Output tensor after MLP computation
        """

        hf_config = cfg["hf_config"]
        num_heads = hf_config.num_attention_heads
        num_key_value_heads = hf_config.num_key_value_heads
        num_heads_local = num_heads // cfg["mesh_shape"][0]
        head_dim = hf_config.head_dim
        num_key_value_groups = num_heads // num_key_value_heads

        k_cache = cfg["k_cache"]
        v_cache = cfg["v_cache"]

        bsz = x.shape[1]  # bsz = 1
        seq_len = x.shape[2]

        logger.info(f"Input x shape: {x.shape}")
        # x : [1, bsz, seq_len, dim]
        # wq
        tt_q = ttnn.linear(x, **cfg["wq"])
        logger.info(f"tt_q shape: {tt_q.shape}")
        # tt_q = ttnn.experimental.reduce_scatter_async(tt_q, **cfg["wq_a_rs"])
        # tt_q = ttnn.experimental.all_gather_async(tt_q, **cfg["wq_a_ag"])
        tt_q = ttnn.reshape(tt_q, (seq_len, bsz, num_heads_local, head_dim))
        logger.info(f"tt_q reshaped1: {tt_q.shape}")
        tt_q = RMSNorm.forward_prefill(tt_q, cfg["q_norm"])
        logger.info(f"tt_q reshaped2: {tt_q.shape}")

        # q rope
        # tt_q = ttnn.to_memory_config(tt_q, **cfg["q_rope_reshard"])
        # tt_q = ttnn.experimental.rotary_embedding_llama(
        #     tt_q,
        #     rope_tensors["cos_matrix"],
        #     rope_tensors["sin_matrix"],
        #     rope_tensors["trans_matrix"],
        #     is_decode_mode=True,
        # )
        # tt_q = ttnn.to_memory_config(tt_q, **cfg["q_rope_out_reshard"])

        # wk
        tt_k = ttnn.linear(x, **cfg["wk"])
        # tt_k = ttnn.experimental.reduce_scatter_async(tt_k, **cfg["wk_rs"])
        # tt_k = ttnn.experimental.all_gather_async(tt_k, **cfg["wk_rs"])
        tt_k = ttnn.reshape(tt_k, (seq_len, bsz, num_key_value_heads, head_dim))
        tt_k = RMSNorm.forward_prefill(tt_k, cfg["k_norm"])
        # tt_k = ttnn.to_memory_config(tt_k, **cfg["k_rope_reshard"])
        # tt_k = ttnn.experimental.rotary_embedding_llama(
        #     tt_k,
        #     rope_tensors["cos_matrix"],
        #     rope_tensors["sin_matrix"],
        #     rope_tensors["trans_matrix"],
        #     is_decode_mode=True,
        # )
        # tt_k = ttnn.to_memory_config(tt_k, **cfg["k_rope_out_reshard"])

        # wv
        tt_v = ttnn.linear(x, **cfg["wv"])
        # tt_v = ttnn.experimental.reduce_scatter_async(tt_v, **cfg["wv_rs"])
        # tt_v = ttnn.experimental.all_gather_async(tt_v, **cfg["wv_ag"])
        tt_v = ttnn.reshape(tt_v, (seq_len, bsz, num_key_value_heads, head_dim))

        # TODO: Update KV Cache
        # ttnn.fill_cache(
        #     kv_cache,
        #     tt_kv,
        #     batch_idx=user_id,
        # )

        tt_k = ttnn.repeat_interleave(tt_k, num_key_value_groups, 2)
        tt_v = ttnn.repeat_interleave(tt_v, num_key_value_groups, 2)
        tt_q = ttnn.permute(tt_q, (1, 2, 0, 3))
        tt_k = ttnn.permute(tt_k, (1, 2, 3, 0))
        tt_v = ttnn.permute(tt_v, (1, 2, 0, 3))
        attn_out = ttnn.matmul(tt_q, tt_k) * cfg["sdpa"]["scale"]
        attn_out = ttnn.softmax(attn_out, dim=-1)
        attn_out = ttnn.matmul(attn_out, tt_v)
        logger.info(f"attn_out shape after v matmul: {attn_out.shape}")
        attn_out = ttnn.transpose(attn_out, 1, 2)
        logger.info(f"attn_out shape after transpose: {attn_out.shape}")
        attn_out = ttnn.reshape(attn_out, (1, 1, seq_len, num_heads_local * head_dim))
        logger.info(f"attn_out shape after reshape: {attn_out.shape}")

        # TODO: SDPA
        # attn_out = ttnn.transformer.scaled_dot_product_attention(
        #     tt_q,
        #     tt_k,
        #     tt_v,
        #     **cfg["sdpa"],
        # )  # [1, num_heads, seq_len, head_dim]
        # ttnn.deallocate(tt_q)

        # wo
        # attn_out = ttnn.experimental.all_gather_async(v_out, **cfg["wo_ag"])  # [1, num_heads, bsz, head_dim]
        # attn_out = ttnn.permute(v_out, (0, 2, 1, 3))  # [1, seq_len, num_heads, head_dim]

        out = ttnn.linear(attn_out, **cfg["wo"])  # [1, 1, seq_len, dim]
        logger.info(f"out shape: {out.shape}")

        return out
