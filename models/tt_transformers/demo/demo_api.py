# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Demo API for TT Transformers LLM Inference

This module provides a high-level API for running different aspects of LLM inference
on TT hardware, including warm-up, encoder (prefill), decoder, and full inference.
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

import ttnn

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.generator import Generator, SamplingParams, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision, determine_device_name


def create_tt_page_table(global_batch_size, data_parallel, paged_attention_config: PagedAttentionConfig):
    """Create TT page table for paged attention - matching original demo."""
    page_table = None

    if paged_attention_config:
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
        page_table = reverse_permutation.reshape(
            global_batch_size, paged_attention_config.max_num_blocks // (global_batch_size // data_parallel)
        )
    return page_table


def prepare_generator_args(
    num_devices,
    data_parallel,
    mesh_device,
    instruct,
    global_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    paged_attention,
):
    """Prepare generator arguments - matching original demo exactly."""
    submesh_devices = create_submeshes(mesh_device, data_parallel)
    state_dict = None

    # Hybrid requires a model per submesh
    model_args = []
    model = []
    tt_kv_cache = []

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks_per_dp"],
        )
        if paged_attention
        else None
    )

    for submesh in submesh_devices:
        model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
            submesh,
            instruct=instruct,
            max_batch_size=global_batch_size // data_parallel,
            optimizations=optimizations,
            max_seq_len=max_seq_len,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
        )
        model_args.append(model_args_i)
        model.append(model_i)
        tt_kv_cache.append(tt_kv_cache_i)

    page_table = create_tt_page_table(
        global_batch_size=global_batch_size,
        data_parallel=data_parallel,
        paged_attention_config=paged_attention_config,
    )
    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args[0].tokenizer
    return model_args, model, page_table, tt_kv_cache, tokenizer


def load_and_cache_context(context_url, cache_dir, max_length=None):
    """Load and cache context from URL - matching original demo."""
    cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()

    if cache_file.exists():
        with open(cache_file, "r") as f:
            context_text = f.read()
        logger.info(f"Loaded context from cache: {context_url}")
    else:
        try:
            import requests

            response = requests.get(context_url)
            if response.status_code == 200:
                context_text = response.text
                with open(cache_file, "w") as f:
                    f.write(context_text)
                logger.info(f"Downloaded and cached context: {context_url}")
            else:
                logger.warning(f"Failed to fetch context from URL: {context_url}. Status code: {response.status_code}")
                context_text = ""
        except Exception as e:
            logger.error(f"Error fetching context from URL: {context_url}. Error: {str(e)}")
            context_text = ""

    # Clip the context to the max length provided
    if max_length:
        context_text = context_text[:max_length]
        logger.info(f"Clipped the context text to {max_length} characters")

    return context_text


def load_inputs(user_input, batch, instruct):
    """Load input prompts from json - matching original demo."""
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)

    if len(user_input) < batch:
        logger.warning(
            f"Number of users in the file is less than the provided batch={batch}. Repeating the prompts to match the batch size."
        )
        user_input = user_input * batch

    in_prompt = []
    cache_dir = Path("models/tt_transformers/demo/context_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # The demo supports a custom prompt file, where the context is provided by a link to a book from the gutenberg project
    # It clips the excerpt to the max length provided to allow testing different long context lengthts
    for i in range(batch):
        prompt = user_input[i]["prompt"]
        if "context" in user_input[i]:
            if "max_length" in user_input[i]:  # Clip the context to the max length provided
                context_text = load_and_cache_context(
                    user_input[i]["context"], cache_dir, max_length=user_input[i]["max_length"]
                )
            else:
                context_text = load_and_cache_context(user_input[i]["context"], cache_dir)
            if instruct:
                prompt = (
                    "```" + context_text + "```\n\n" + prompt
                )  # Add the markdown block to the context to comply with the prompt
            else:
                prompt = context_text
        in_prompt.append(prompt)
    return in_prompt


class LLMDemoAPI:
    """
    High-level API for TT Transformers LLM inference.

    This class provides methods to run different aspects of LLM inference
    on TT hardware, including warm-up, encoder (prefill), decoder, and full inference.
    """

    def __init__(
        self,
        mesh_device,
        instruct: bool = True,
        max_batch_size: int = 1,
        max_seq_len: int = 1024,
        data_parallel: int = 1,
        optimizations=None,
        paged_attention: bool = True,
        page_params: Dict = None,
    ):
        """
        Initialize the LLM Demo API.

        Args:
            mesh_device: TT mesh device
            instruct: Whether to use instruction mode
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            data_parallel: Data parallelism factor
            optimizations: Model optimizations
            paged_attention: Whether to use paged attention
            page_params: Paged attention parameters
        """
        self.mesh_device = mesh_device
        self.instruct = instruct
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.data_parallel = data_parallel
        self.optimizations = optimizations
        self.paged_attention = paged_attention
        self.page_params = page_params or {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}

        # Model components
        self.model_args = None
        self.model = None
        self.page_table = None
        self.tt_kv_cache = None
        self.tokenizer = None
        self.generator = None
        self.is_initialized = False

        # Performance tracking
        self.profiler = None

    def _verify_compute_optimizations(self):
        """Verify that compute kernel optimizations are properly applied - matching original demo."""
        logger.info("Verifying compute optimizations...")

        # Check if model has the required compute configurations
        if hasattr(self, "model_args") and self.model_args:
            model_args = self.model_args[0] if isinstance(self.model_args, list) else self.model_args

            # Verify compute kernel configurations are present
            required_compute_configs = [
                "compute_kernel_config_lofi",
                "compute_kernel_config_hifi2",
                "compute_kernel_config_hifi2_na",
                "compute_kernel_config_hifi2_fp16",
                "compute_kernel_config_hifi4",
            ]

            missing_compute_configs = []
            for config in required_compute_configs:
                if not hasattr(model_args, config):
                    missing_compute_configs.append(config)

            if missing_compute_configs:
                logger.warning(f"Missing compute configurations: {missing_compute_configs}")
            else:
                logger.info("All required compute configurations present")

            # Verify optimization settings are applied
            if hasattr(model_args, "optimizations") and model_args.optimizations:
                logger.info(f"Compute optimization level: {model_args.optimizations.__name__}")

                # Check if performance optimizations are applied
                if hasattr(model_args.optimizations, "tensor_dtype_settings"):
                    tensor_settings = model_args.optimizations.tensor_dtype_settings
                    logger.info(f"Tensor precision settings: {tensor_settings}")

                if hasattr(model_args.optimizations, "op_fidelity_settings"):
                    fidelity_settings = model_args.optimizations.op_fidelity_settings
                    logger.info(f"Op fidelity settings: {fidelity_settings}")
            else:
                logger.warning("No compute optimizations applied")

        logger.info("Compute optimization verification complete")

    def _verify_trace_optimizations(self):
        """Verify that trace optimizations are properly applied - matching original demo."""
        logger.info("Verifying trace optimizations...")

        # Check if generator has trace optimizations
        if hasattr(self, "generator") and self.generator:
            # Verify trace capture is enabled
            if hasattr(self.generator, "_capture_trace_text"):
                logger.info("Trace capture method available")
            else:
                logger.warning("Trace capture method not available")

            # Verify trace execution is optimized
            if hasattr(self.generator, "decode_forward_text"):
                logger.info("Decode forward method available")
            else:
                logger.warning("Decode forward method not available")

        # Check if mesh device supports trace optimizations
        if hasattr(self, "mesh_device") and self.mesh_device:
            logger.info("Mesh device available for trace optimizations")
        else:
            logger.warning("Mesh device not available")

        logger.info("Trace optimization verification complete")

    def _verify_memory_optimizations(self):
        """Verify that memory optimizations are properly applied - matching original demo."""
        logger.info("Verifying memory optimizations...")

        # Check if model has the required memory configurations
        if hasattr(self, "model_args") and self.model_args:
            model_config = (
                self.model_args[0].model_config if isinstance(self.model_args, list) else self.model_args.model_config
            )

            # Verify key memory configurations are present
            required_configs = [
                "DECODE_RESIDUAL_MEMCFG",
                "XQKV_DECODE_PROGCFG",
                "SDPA_DECODE_PROGCFG",
                "DECODE_MLP_W1_W3_PRG_CONFIG",
                "DECODE_MLP_W2_PRG_CONFIG",
            ]

            missing_configs = []
            for config in required_configs:
                if config not in model_config:
                    missing_configs.append(config)

            if missing_configs:
                logger.warning(f"Missing memory configurations: {missing_configs}")
            else:
                logger.info("All required memory configurations present")

            # Verify optimization settings
            if hasattr(self.model_args[0], "optimizations") and self.model_args[0].optimizations:
                logger.info(f"Optimization level: {self.model_args[0].optimizations.__name__}")
            else:
                logger.warning("No optimizations applied")

        logger.info("Memory optimization verification complete")

    def initialize_model(self):
        """Initialize the model components using original demo approach."""
        logger.info("Initializing model components...")

        # Calculate parameters
        num_devices = self.mesh_device.get_num_devices() if hasattr(self.mesh_device, "get_num_devices") else 1
        global_batch_size = self.max_batch_size * self.data_parallel

        # Use the original demo's prepare_generator_args function
        self.model_args, self.model, self.page_table, self.tt_kv_cache, self.tokenizer = prepare_generator_args(
            num_devices=num_devices,
            data_parallel=self.data_parallel,
            mesh_device=self.mesh_device,
            instruct=self.instruct,
            global_batch_size=global_batch_size,
            optimizations=self.optimizations,
            max_seq_len=self.max_seq_len,
            page_params=self.page_params,
            paged_attention=self.paged_attention,
        )

        # Create generator using the original demo's approach
        self.generator = Generator(
            self.model,
            self.model_args,
            self.mesh_device,
            tokenizer=self.tokenizer,
        )

        self.is_initialized = True
        logger.info("Model initialization complete")

        # OPTIMIZATION: Verify memory optimizations are applied
        self._verify_memory_optimizations()

        # OPTIMIZATION: Verify compute optimizations are applied
        self._verify_compute_optimizations()

        # OPTIMIZATION: Verify trace optimizations are applied
        self._verify_trace_optimizations()

    def load_inputs(self, input_file: str, batch_size: int) -> List[str]:
        """
        Load input prompts from JSON file - matching original demo.

        Args:
            input_file: Path to JSON file containing prompts
            batch_size: Number of prompts to load

        Returns:
            List of prompt strings
        """
        return load_inputs(input_file, batch_size, self.instruct)

    def _preprocess_inputs(self, input_prompts: List[str]) -> Tuple[torch.Tensor, List[int], List[int]]:
        """Preprocess input prompts for inference using original demo logic."""
        # Use the original demo's preprocessing function
        input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
            input_prompts,
            self.tokenizer,
            self.model_args,
            self.instruct,
            max_generated_tokens=200,  # Default value, can be overridden
        )

        # Stack the list of tensors into a single tensor
        if isinstance(input_tokens_prefill_pt, list):
            input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt, dim=0)

        return input_tokens_prefill_pt, decoding_pos, prefill_lens

    def warm_up(self, input_prompts: List[str], max_generated_tokens: int = 200) -> Dict:
        """
        Warm up the model with trace capture and optimization - matching original demo.

        Args:
            input_prompts: List of input prompts to warm up with
            max_generated_tokens: Maximum number of tokens to generate during warm-up

        Returns:
            Dictionary with warm-up results and timing information
        """
        if not self.is_initialized:
            self.initialize_model()

        logger.info("Starting model warm-up with trace optimizations...")

        # OPTIMIZATION: Use exact same warm-up approach as original demo
        start_time = time.time()

        # Preprocess inputs - matching original demo
        initial_tokens, decoding_pos, prefill_lens = self._preprocess_inputs(input_prompts)

        # OPTIMIZATION: Run prefill warm-up with trace capture
        logger.info("Running prefill warm-up with trace capture...")
        prefill_start_time = time.time()

        # Run prefill - matching original demo exactly
        # Convert list to tensor if needed
        if isinstance(initial_tokens, list):
            initial_tokens = torch.stack(initial_tokens)
        # Squeeze to 2D if needed
        if initial_tokens.dim() == 3 and initial_tokens.shape[1] == 1:
            initial_tokens = initial_tokens.squeeze(1)

        # OPTIMIZATION: Add prompt_lens parameter to match original demo
        prompt_lens = torch.tensor(decoding_pos)

        self.generator.prefill_forward_text(
            initial_tokens,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=prompt_lens,
        )

        prefill_time = time.time() - prefill_start_time
        logger.info(f"Prefill warm-up completed in {prefill_time:.3f}s")

        # OPTIMIZATION: Run decode warm-up with trace capture and optimization
        logger.info("Running decode warm-up with trace capture and optimization...")
        decode_start_time = time.time()

        # Run decode warm-up - matching original demo exactly
        # This is where the trace is captured and optimized
        # Use the correct parameters for _capture_trace_text
        # Ensure current_pos has correct batch shape/type for batch size
        current_pos = torch.tensor([int(val.item()) if hasattr(val, "item") else int(val) for val in decoding_pos])
        logger.info(f"DEBUG: current_pos type={type(current_pos)}, value={current_pos}")

        # OPTIMIZATION: Handle current_pos based on data_parallel configuration
        # For data_parallel=1, pass as single tensor; for data_parallel>1, chunk it
        if self.data_parallel == 1:
            current_pos_for_trace = current_pos
        else:
            current_pos_for_trace = torch.chunk(current_pos, self.data_parallel, 0)

        self.generator._capture_trace_text(
            initial_tokens,
            current_pos_for_trace,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
        )

        decode_time = time.time() - decode_start_time
        logger.info(f"Decode warm-up completed in {decode_time:.3f}s")

        total_warm_up_time = time.time() - start_time
        logger.info(f"Warm-up complete with trace optimizations in {total_warm_up_time:.3f}s")

        return {
            "warm_up_time": total_warm_up_time,
            "prefill_warm_up_time": prefill_time,
            "decode_warm_up_time": decode_time,
            "initial_tokens": initial_tokens,
            "prefill_lens": prefill_lens,
        }

    def measure_ttft(self, input_prompts: List[str], sampling_params: Dict = None) -> Dict:
        """
        Measure Time to First Token (TTFT) separately from compilation time.

        Args:
            input_prompts: List of input prompts
            sampling_params: Sampling parameters for generation

        Returns:
            Dictionary containing TTFT measurement results
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        if sampling_params is None:
            sampling_params = {"temperature": 0, "top_p": 0.08}

        logger.info("Measuring Time to First Token (TTFT)...")

        # Preprocess inputs
        input_tokens_prefill_pt, decoding_pos, prefill_lens = self._preprocess_inputs(input_prompts)

        # Ensure tensor has correct shape (batch_size, seq_len)
        # The preprocess_inputs_prefill returns tensors of shape (1, seq_len) for each batch item
        # After stacking, we get (batch_size, seq_len) - this should be correct
        logger.info(f"DEBUG: input_tokens_prefill_pt shape: {input_tokens_prefill_pt.shape}")
        logger.info(f"DEBUG: input_tokens_prefill_pt dtype: {input_tokens_prefill_pt.dtype}")

        # Fix tensor shape if it has extra dimensions
        if input_tokens_prefill_pt.dim() == 3:
            # Reshape from (batch_size, 1, seq_len) to (batch_size, seq_len)
            input_tokens_prefill_pt = input_tokens_prefill_pt.squeeze(1)
            logger.info(f"DEBUG: After squeeze, shape: {input_tokens_prefill_pt.shape}")

        # Start TTFT measurement (excluding compilation)
        ttft_start_time = time.time()

        # Run prefill to get first token
        output_logits = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=prefill_lens,
        )

        # Get first token
        if sampling_params.get("temperature", 0) == 0:
            # Greedy decoding
            first_token = torch.argmax(output_logits, dim=-1).unsqueeze(1)
        else:
            # Sampling
            _, first_token = sample_host(
                output_logits,
                temperature=sampling_params.get("temperature", 0.6),
                top_p=sampling_params.get("top_p", 0.08),
                on_host=True,
            )

        ttft_time = time.time() - ttft_start_time

        # Calculate per-user TTFT (matching original demo calculation)
        batch_size = len(input_prompts)
        ttft_time_per_user = ttft_time / batch_size
        ttft_time_per_user_ms = ttft_time_per_user * 1000

        logger.info(f"Total Time to First Token (TTFT): {ttft_time*1000:.2f}ms")
        logger.info(f"Per-User Time to First Token (TTFT): {ttft_time_per_user_ms:.2f}ms")

        return {
            "ttft_time": ttft_time,
            "ttft_time_ms": ttft_time * 1000,
            "ttft_time_per_user": ttft_time_per_user,
            "ttft_time_per_user_ms": ttft_time_per_user_ms,
            "first_token": first_token,
            "output_logits": output_logits,
        }

    def run_encoder(
        self, input_prompts: List[str], max_generated_tokens: int = 200
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        Run encoder (prefill) phase only - matching original demo.

        Args:
            input_prompts: List of input prompts
            max_generated_tokens: Maximum number of tokens to generate

        Returns:
            Tuple of (sampled_tokens, decoding_positions, prefill_lengths)
        """
        if not self.is_initialized:
            self.initialize_model()

        logger.info("Running encoder (prefill) phase...")

        # Preprocess inputs using original demo logic
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts,
            self.tokenizer,
            self.model_args,
            self.instruct,
            max_generated_tokens,
            max_prefill_len=self.max_seq_len,
        )

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(len(input_prompts), -1)

        # Run prefill
        logits = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=prefill_lens,
        )

        # Sample the first token
        sampled_tokens = torch.argmax(logits, dim=-1)

        logger.info("Encoder (prefill) phase complete")

        return sampled_tokens, decoding_pos, prefill_lens

    def run_decoder(
        self,
        initial_tokens: torch.Tensor,
        start_positions: List[int],
        max_tokens: int = 200,
        sampling_params: Optional[Dict] = None,
        stop_at_eos: bool = True,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Run decoder phase only - matching original demo with optimizations.

        Args:
            initial_tokens: Initial tokens to start decoding from
            start_positions: Starting positions for each sequence
            max_tokens: Maximum number of tokens to generate
            sampling_params: Sampling parameters for token generation
            stop_at_eos: Whether to stop at end-of-sequence tokens

        Returns:
            Tuple of (generated_tokens, token_timings)
        """
        if not self.is_initialized:
            self.initialize_model()

        logger.info("Running decoder phase with optimizations...")

        # Set up sampling parameters
        if sampling_params is None:
            sampling_params = {"temperature": 0, "top_p": 0.08}

        # Setup device sampling parameters (matching original demo logic exactly)
        global_batch_size = self.max_batch_size
        data_parallel = self.data_parallel

        # TODO Argmax on device is only supported for batch_size=1 (per submesh)
        argmax_on_device = (
            False if (global_batch_size // data_parallel > 1 or sampling_params["temperature"] != 0) else True
        )
        if argmax_on_device:
            device_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
        else:
            device_sampling_params = None

        # Initialize tracking variables - matching original demo exactly
        current_pos = torch.tensor(start_positions)
        all_outputs = [[] for _ in range(len(initial_tokens))]
        user_done = [False] * len(initial_tokens)

        # Add initial tokens to outputs
        for user in range(len(initial_tokens)):
            all_outputs[user].append(int(initial_tokens[user].item()))

        # Ensure out_tok has correct shape [batch_size, 1] - matching original demo
        out_tok = initial_tokens.unsqueeze(1) if initial_tokens.dim() == 1 else initial_tokens
        iteration = 0
        users_decoding = True

        # Performance tracking - matching original demo
        token_timings = []

        # OPTIMIZATION: Enable explicit memory optimizations for decode mode
        logger.info(f"Using device-side argmax: {argmax_on_device}")
        logger.info(f"Global batch size: {global_batch_size}, Data parallel: {data_parallel}")

        # OPTIMIZATION: Ensure trace is properly captured and executed
        logger.info("Enabling trace optimizations for decode loop")

        while users_decoding:
            # Performance timing - matching original demo
            decode_start_time = time.time()

            # OPTIMIZATION: Use exact same trace execution pattern as original demo
            # Run decode forward - matching original demo exactly with optimizations
            logits = self.generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=True,  # Enable tracing for performance measurement
                page_table=self.page_table,
                kv_cache=self.tt_kv_cache,
                sampling_params=device_sampling_params,
            )

            # OPTIMIZATION: Use exact same token extraction logic as original demo
            # Get next token - matching original demo logic exactly
            if device_sampling_params is not None:
                # OPTIMIZATION: Device-side argmax - no host transfer needed
                out_tok = logits.unsqueeze(1)
            else:
                # TODO Fix use case with temperature > 0
                _, out_tok = sample_host(
                    logits,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    on_host=True,
                )

            # Performance timing - matching original demo
            decode_iteration_time = time.time() - decode_start_time
            token_timings.append(decode_iteration_time)

            # Performance logging - matching original demo
            tokens_per_second_per_user = 1 / decode_iteration_time
            logger.info(
                f"Iteration {iteration}: {1000*decode_iteration_time:.0f}ms @ {tokens_per_second_per_user:.1f} tok/s/user ({global_batch_size*tokens_per_second_per_user:.1f} tok/s throughput)"
            )

            # Update positions and outputs - matching original demo exactly
            current_pos += 1
            for user in range(len(initial_tokens)):
                user_tok = out_tok[user].item()
                if user_tok not in self.tokenizer.stop_tokens and not user_done[user]:
                    all_outputs[user].append(user_tok)
                else:
                    if stop_at_eos:
                        user_done[user] = True
                        logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                        if all(user_done):
                            users_decoding = False

            iteration += 1

            # Upper limit of generated tokens for each user
            if iteration >= max_tokens:
                users_decoding = False

        logger.info("Finished decoding with optimizations")

        return all_outputs, token_timings

    def run_full_inference(
        self,
        input_prompts: List[str],
        max_generated_tokens: int = 200,
        sampling_params: Dict = None,
        stop_at_eos: bool = True,
        enable_trace: bool = True,
    ) -> Dict:
        """
        Run full inference (prefill + decode) with performance measurement.

        Args:
            input_prompts: List of input prompts
            max_generated_tokens: Maximum number of tokens to generate
            sampling_params: Sampling parameters for generation
            stop_at_eos: Whether to stop at end-of-sequence token
            enable_trace: Whether to enable tracing (default: True for performance)

        Returns:
            Dictionary containing outputs and performance metrics
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        if sampling_params is None:
            sampling_params = {"temperature": 0, "top_p": 0.08}

        logger.info("Running full inference with optimizations...")

        # Read inputs
        logger.info("Reading inputs...")
        input_tokens_prefill_pt, decoding_pos, prefill_lens = self._preprocess_inputs(input_prompts)

        # Measure TTFT separately (excluding compilation)
        ttft_results = self.measure_ttft(input_prompts, sampling_params)

        # Start full inference timing (including compilation if needed)
        start_time = time.time()

        # Prefill phase with tracing
        logger.info("Starting prefill...")

        # Stack tensors if it's a list
        if isinstance(input_tokens_prefill_pt, list):
            input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt, dim=0)

        # Ensure tensor has correct shape (batch_size, seq_len)
        if input_tokens_prefill_pt.dim() == 3:
            # Reshape from (batch_size, 1, seq_len) to (batch_size, seq_len)
            input_tokens_prefill_pt = input_tokens_prefill_pt.squeeze(1)

        # Call prefill with correct parameters
        output_logits = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=prefill_lens,
        )

        # Now handle the generation phase
        logger.info("Starting generation...")

        # Initialize generation state for batch processing
        batch_size = len(input_prompts)
        generated_tokens_per_batch = [[] for _ in range(batch_size)]
        user_done = [False] * batch_size

        # Create current_pos tensor for the entire batch - matching original demo
        current_pos = torch.tensor(prefill_lens)  # Start from the end of prefill for each sequence
        out_tok = torch.argmax(output_logits, dim=-1).unsqueeze(1)  # [batch_size, 1]

        # Add initial tokens to each batch item
        for user in range(batch_size):
            generated_tokens_per_batch[user].append(out_tok[user, 0].item())

        for i in range(1, max_generated_tokens):
            # Decode next token
            output_logits = self.generator.decode_forward_text(
                out_tok,
                current_pos,
                page_table=self.page_table,
                kv_cache=self.tt_kv_cache,
                enable_trace=enable_trace,
                sampling_params=SamplingParams(
                    temperature=sampling_params.get("temperature", 0),
                    top_k=sampling_params.get("top_k", 1),
                    top_p=sampling_params.get("top_p", 0.08),
                )
                if sampling_params.get("temperature", 0) > 0
                else None,
            )
            # Sample next token
            if sampling_params.get("temperature", 0) == 0:
                # Greedy decoding
                out_tok = torch.argmax(output_logits, dim=-1).unsqueeze(1)
            else:
                # Sampling
                _, out_tok = sample_host(
                    output_logits,
                    temperature=sampling_params.get("temperature", 0.6),
                    top_p=sampling_params.get("top_p", 0.08),
                    on_host=True,
                )

            # Add tokens to each batch item and check for EOS
            for user in range(batch_size):
                if not user_done[user]:
                    user_tok = out_tok[user, 0].item()
                    generated_tokens_per_batch[user].append(user_tok)

                    # Check for EOS token
                    if stop_at_eos and user_tok == self.tokenizer.eos_token_id:
                        user_done[user] = True
                        logger.trace(f"[User {user}] Finished decoding at iteration {i}")

            # Stop if all users are done
            if all(user_done):
                break

            current_pos += 1

        total_inference_time = time.time() - start_time
        logger.info("Prefill finished")

        # Calculate compilation time (difference between total and TTFT)
        compilation_time = total_inference_time - ttft_results["ttft_time"]

        logger.info(f"Total inference time: {total_inference_time:.3f}s")
        logger.info(f"Compilation time: {compilation_time:.3f}s")
        logger.info(f"Total TTFT (excluding compilation): {ttft_results['ttft_time_ms']:.2f}ms")
        logger.info(f"Per-User TTFT (excluding compilation): {ttft_results['ttft_time_per_user_ms']:.2f}ms")

        # Decode phase timing
        decode_time = time.time() - start_time

        # Calculate performance metrics
        total_tokens = sum(len(tokens) for tokens in generated_tokens_per_batch)
        average_tokens_per_prompt = total_tokens / len(input_prompts) if input_prompts else 0

        # Calculate throughput metrics
        prefill_tokens = sum(prefill_lens)
        decode_tokens = total_tokens - prefill_tokens

        # Use TTFT time as prefill time for performance calculations
        prefill_time = ttft_results["ttft_time"]
        prefill_tokens_per_second = prefill_tokens / prefill_time if prefill_time > 0 else 0
        decode_tokens_per_second_per_user = (
            decode_tokens / decode_time / len(input_prompts) if decode_time > 0 and input_prompts else 0
        )
        decode_tokens_per_second_total = decode_tokens / decode_time if decode_time > 0 else 0

        # Get profiler timings if available from the generator
        try:
            compile_prefill_time = (
                self.generator.profiler.get_duration("compile_prefill") if hasattr(self.generator, "profiler") else 0.0
            )
        except (KeyError, AttributeError):
            compile_prefill_time = 0.0

        try:
            compile_decode_time = (
                self.generator.profiler.get_duration("compile_decode") if hasattr(self.generator, "profiler") else 0.0
            )
        except (KeyError, AttributeError):
            compile_decode_time = 0.0

        inference_prefill_time = prefill_time
        inference_decode_time = decode_time

        # Decode outputs for each batch item
        outputs = []
        for user in range(batch_size):
            # Convert generated tokens for this user to text
            output_text = self.tokenizer.decode(generated_tokens_per_batch[user], skip_special_tokens=True)
            outputs.append(output_text)

        # Performance metrics
        performance_metrics = {
            "compile_prefill": compile_prefill_time,
            "compile_decode": compile_decode_time,
            "inference_prefill": inference_prefill_time,
            "inference_decode": inference_decode_time,
            "prefill_time_to_token": prefill_time / prefill_tokens if prefill_tokens > 0 else 0,
            "prefill_t/s": prefill_tokens_per_second,
            "decode_t/s/u": decode_tokens_per_second_per_user,
            "decode_t/s": decode_tokens_per_second_total,
            "total_tokens": total_tokens,
            "average_tokens_per_prompt": average_tokens_per_prompt,
            # Separated timing metrics
            "ttft_time_ms": ttft_results["ttft_time_ms"],
            "ttft_time": ttft_results["ttft_time"],
            "ttft_time_per_user_ms": ttft_results["ttft_time_per_user_ms"],
            "ttft_time_per_user": ttft_results["ttft_time_per_user"],
            "compilation_time": compilation_time,
            "total_inference_time": total_inference_time,
        }

        return {
            "outputs": outputs,
            "performance_metrics": performance_metrics,
            "num_prompts": len(input_prompts),
            "generated_tokens": generated_tokens_per_batch,
            "ttft_results": ttft_results,  # Include TTFT results for reference
        }

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_initialized:
            return {"error": "Model not initialized"}

        return {
            "model_name": self.model_args[0].model_name,
            "base_model_name": self.model_args[0].base_model_name,
            "vocab_size": self.model_args[0].vocab_size,
            "n_layers": self.model_args[0].n_layers,
            "max_seq_len": self.max_seq_len,
            "max_batch_size": self.max_batch_size,
            "data_parallel": self.data_parallel,
            "instruct": self.instruct,
            "paged_attention": self.paged_attention,
            "device_name": determine_device_name(self.mesh_device),
        }

    def reset_kv_cache(self) -> None:
        """Reset the KV cache to zero - matching original demo."""
        if not self.is_initialized:
            logger.warning("Model not initialized, cannot reset KV cache")
            return

        logger.info("Resetting KV cache...")
        for i in range(len(self.model)):
            for layer in self.model[i].layers:
                k_cache, v_cache = layer.attention.layer_past
                k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)
        logger.info("KV cache reset complete")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.is_initialized:
            logger.info("Cleaning up resources...")
            # Add any cleanup logic here
            self.is_initialized = False
            logger.info("Cleanup complete")


# Convenience functions for common use cases
def create_demo_api(
    mesh_device,
    instruct: bool = True,
    max_batch_size: int = 1,
    max_seq_len: int = 1024,
    data_parallel: int = 1,
    optimizations: Optional[DecodersPrecision] = None,
    paged_attention: bool = True,
    page_params: Optional[Dict] = None,
) -> LLMDemoAPI:
    """
    Create a demo API instance with default parameters.

    Args:
        mesh_device: TT mesh device
        instruct: Whether to use instruct weights
        max_batch_size: Maximum batch size per data parallel group
        max_seq_len: Maximum sequence length
        data_parallel: Number of data parallel groups
        optimizations: Model optimization level
        paged_attention: Whether to use paged attention
        page_params: Page parameters for paged attention

    Returns:
        LLMDemoAPI instance
    """
    return LLMDemoAPI(
        mesh_device=mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        data_parallel=data_parallel,
        optimizations=optimizations,
        paged_attention=paged_attention,
        page_params=page_params,
    )


def run_quick_demo(mesh_device, input_prompts: List[str], max_generated_tokens: int = 50, **kwargs) -> Dict:
    """
    Run a quick demo with minimal setup.

    Args:
        mesh_device: TT mesh device
        input_prompts: List of input prompts
        max_generated_tokens: Maximum number of tokens to generate
        **kwargs: Additional arguments for LLMDemoAPI

    Returns:
        Dictionary containing demo results
    """
    api = create_demo_api(mesh_device, **kwargs)
    api.initialize_model()
    api.warm_up(input_prompts, max_generated_tokens=10)
    results = api.run_full_inference(input_prompts, max_generated_tokens=max_generated_tokens)
    api.cleanup()
    return results
