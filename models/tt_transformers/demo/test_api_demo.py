#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test file for the LLM Demo API

This file tests the API functionality and performance against the original demo.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from loguru import logger

import ttnn

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from demo_api import create_demo_api

from models.tt_transformers.tt.model_config import DecodersPrecision


def load_test_inputs(input_file: str, batch_size: int) -> list:
    """Load test inputs from JSON file."""
    with open(input_file, "r") as f:
        user_input = json.load(f)

    if len(user_input) < batch_size:
        logger.warning(
            f"Number of users in the file is less than the provided batch={batch_size}. "
            f"Repeating the prompts to match the batch size."
        )
        user_input = user_input * batch_size

    in_prompt = []
    for i in range(batch_size):
        prompt = user_input[i]["prompt"]
        in_prompt.append(prompt)

    return in_prompt


@pytest.mark.parametrize(
    "batch_size, max_generated_tokens, sampling_params, stop_at_eos",
    [
        (1, 200, {"temperature": 0, "top_p": 0.08}, True),  # Batch-1 test
        (32, 200, {"temperature": 0, "top_p": 0.08}, True),  # Batch-32 test
    ],
    ids=["batch-1", "batch-32"],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_api_performance(
    batch_size,
    max_generated_tokens,
    sampling_params,
    stop_at_eos,
    mesh_device,
    request,
):
    """
    Test the API performance against the original demo.
    """
    test_id = request.node.callspec.id

    # Skip if not in CI environment for batch-32
    if "batch-32" in test_id and not os.environ.get("CI"):
        pytest.skip("Batch-32 test only runs in CI environment")

    logger.info(f"Running API performance test: {test_id}")

    # Test parameters
    input_file = "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json"
    max_seq_len = 1024
    data_parallel = 1
    paged_attention = True
    page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
    instruct = True

    # Create API instance
    api = create_demo_api(
        mesh_device=mesh_device,
        instruct=instruct,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        data_parallel=data_parallel,
        optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        paged_attention=paged_attention,
        page_params=page_params,
    )

    try:
        # Initialize model
        logger.info("Initializing model...")
        api.initialize_model()

        # Load inputs
        logger.info("Loading inputs...")
        input_prompts = load_test_inputs(input_file, batch_size)

        # Warm up
        logger.info("Running warm-up...")
        warm_up_result = api.warm_up(input_prompts, max_generated_tokens=10)
        assert warm_up_result["warm_up_complete"] == True

        # Run full inference
        logger.info("Running full inference...")
        results = api.run_full_inference(
            input_prompts=input_prompts,
            max_generated_tokens=max_generated_tokens,
            sampling_params=sampling_params,
            stop_at_eos=stop_at_eos,
            enable_trace=True,  # Explicitly enable tracing for performance measurement
        )

        # Verify results
        assert "outputs" in results
        assert "performance_metrics" in results
        assert len(results["outputs"]) == batch_size
        assert results["num_prompts"] == batch_size

        # Print performance metrics
        metrics = results["performance_metrics"]
        logger.info("=== API Performance Metrics ===")
        logger.info(f"Compile prefill time: {metrics['compile_prefill']:.3f}s")
        logger.info(f"Compile decode time: {metrics['compile_decode']:.3f}s")
        logger.info(f"Inference prefill time: {metrics['inference_prefill']:.3f}s")
        logger.info(f"Inference decode time: {metrics['inference_decode']:.3f}s")
        logger.info(f"Prefill time to token: {metrics['prefill_time_to_token']*1000:.2f}ms")
        logger.info(f"Prefill tokens/s: {metrics['prefill_t/s']:.2f}")
        logger.info(f"Decode tokens/s/user: {metrics['decode_t/s/u']:.2f}")
        logger.info(f"Decode tokens/s: {metrics['decode_t/s']:.2f}")
        logger.info(f"Total tokens generated: {metrics['total_tokens']}")
        logger.info(f"Average tokens per prompt: {metrics['average_tokens_per_prompt']:.2f}")

        # Print some outputs
        logger.info("=== Sample Outputs ===")
        for i, output in enumerate(results["outputs"][:3]):  # Show first 3 outputs
            logger.info(f"User {i}: {output[:200]}...")

        # Performance assertions
        assert metrics["total_tokens"] > 0
        assert metrics["decode_t/s/u"] > 0
        assert metrics["decode_t/s"] > 0

        logger.info(f"API test {test_id} completed successfully")

    finally:
        # Cleanup
        api.cleanup()


@pytest.mark.parametrize(
    "batch_size, max_generated_tokens",
    [
        (1, 50),  # Quick batch-1 test
        (32, 50),  # Quick batch-32 test
    ],
    ids=["quick-batch-1", "quick-batch-32"],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_api_functionality(
    batch_size,
    max_generated_tokens,
    mesh_device,
    request,
):
    """
    Test basic API functionality.
    """
    test_id = request.node.callspec.id

    # Skip if not in CI environment for batch-32
    if "batch-32" in test_id and not os.environ.get("CI"):
        pytest.skip("Batch-32 test only runs in CI environment")

    logger.info(f"Running API functionality test: {test_id}")

    # Test parameters
    input_file = "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json"
    max_seq_len = 1024
    data_parallel = 1
    paged_attention = True
    page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
    instruct = True

    # Create API instance
    api = create_demo_api(
        mesh_device=mesh_device,
        instruct=instruct,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        data_parallel=data_parallel,
        optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        paged_attention=paged_attention,
        page_params=page_params,
    )

    try:
        # Initialize model
        logger.info("Initializing model...")
        api.initialize_model()

        # Get model info
        model_info = api.get_model_info()
        logger.info(f"Model info: {model_info}")
        assert "model_name" in model_info
        assert "base_model_name" in model_info

        # Load inputs
        logger.info("Loading inputs...")
        input_prompts = load_test_inputs(input_file, batch_size)

        # Test encoder only
        logger.info("Testing encoder...")
        sampled_tokens, decoding_pos, prefill_lens = api.run_encoder(
            input_prompts, max_generated_tokens=max_generated_tokens
        )
        assert sampled_tokens.shape[0] == batch_size
        assert len(decoding_pos) == batch_size
        assert len(prefill_lens) == batch_size

        # Test decoder only
        logger.info("Testing decoder...")
        generated_tokens, token_timings = api.run_decoder(
            initial_tokens=sampled_tokens,
            start_positions=decoding_pos,
            max_tokens=max_generated_tokens,
            sampling_params={"temperature": 0, "top_p": 0.08},
            stop_at_eos=True,
        )
        assert len(generated_tokens) == batch_size

        # Test full inference
        logger.info("Testing full inference...")
        results = api.run_full_inference(
            input_prompts=input_prompts,
            max_generated_tokens=max_generated_tokens,
            sampling_params={"temperature": 0, "top_p": 0.08},
            stop_at_eos=True,
            enable_trace=True,
        )

        # Verify results
        assert "outputs" in results
        assert "performance_metrics" in results
        assert len(results["outputs"]) == batch_size

        # Test KV cache reset
        logger.info("Testing KV cache reset...")
        api.reset_kv_cache()

        logger.info(f"API functionality test {test_id} completed successfully")

    finally:
        # Cleanup
        api.cleanup()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_api_model_info(mesh_device):
    """Test getting model information."""
    logger.info("Testing model info...")

    api = create_demo_api(
        mesh_device=mesh_device,
        instruct=True,
        max_batch_size=1,
        max_seq_len=1024,
        data_parallel=1,
        optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        paged_attention=True,
        page_params={"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},
    )

    try:
        # Test before initialization
        model_info = api.get_model_info()
        assert "error" in model_info

        # Initialize and test again
        api.initialize_model()
        model_info = api.get_model_info()
        assert "model_name" in model_info
        assert "base_model_name" in model_info
        assert "vocab_size" in model_info
        assert "n_layers" in model_info
        assert "max_seq_len" in model_info
        assert "max_batch_size" in model_info
        assert "data_parallel" in model_info
        assert "instruct" in model_info
        assert "paged_attention" in model_info
        assert "device_name" in model_info

        logger.info(f"Model info: {model_info}")

    finally:
        api.cleanup()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_api_input_loading(mesh_device):
    """Test input loading functionality."""
    logger.info("Testing input loading...")

    api = create_demo_api(
        mesh_device=mesh_device,
        instruct=True,
        max_batch_size=1,
        max_seq_len=1024,
        data_parallel=1,
        optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        paged_attention=True,
        page_params={"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},
    )

    try:
        # Test loading inputs
        input_file = "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json"
        inputs = api.load_inputs(input_file, batch_size=2)

        assert len(inputs) == 2
        assert all(isinstance(prompt, str) for prompt in inputs)
        assert all(len(prompt) > 0 for prompt in inputs)

        logger.info(f"Loaded {len(inputs)} inputs successfully")

    finally:
        api.cleanup()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_api_ttft_measurement(mesh_device):
    """Test TTFT measurement functionality."""
    logger.info("Testing TTFT measurement...")

    api = create_demo_api(
        mesh_device=mesh_device,
        instruct=True,
        max_batch_size=1,
        max_seq_len=1024,
        data_parallel=1,
        optimizations=lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        paged_attention=True,
        page_params={"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},
    )

    try:
        # Initialize model
        api.initialize_model()

        # Test prompts
        test_prompts = ["What is your favorite condiment?"]

        # Warm up the model first
        logger.info("Warming up model...")
        warm_up_results = api.warm_up(test_prompts, max_generated_tokens=50)

        # Now measure TTFT separately
        logger.info("Measuring TTFT...")
        ttft_results = api.measure_ttft(test_prompts)

        # Verify TTFT results
        assert "ttft_time" in ttft_results
        assert "ttft_time_ms" in ttft_results
        assert "first_token" in ttft_results
        assert "output_logits" in ttft_results

        # Verify reasonable TTFT values
        assert ttft_results["ttft_time"] > 0
        assert ttft_results["ttft_time_ms"] > 0
        assert ttft_results["ttft_time_ms"] < 10000  # Should be less than 10 seconds

        logger.info(f"TTFT measurement successful: {ttft_results['ttft_time_ms']:.2f}ms")

        # Test with different sampling parameters
        sampling_params = {"temperature": 0.6, "top_p": 0.08}
        ttft_results_with_sampling = api.measure_ttft(test_prompts, sampling_params)

        assert "ttft_time" in ttft_results_with_sampling
        assert "ttft_time_ms" in ttft_results_with_sampling

        logger.info(f"TTFT with sampling: {ttft_results_with_sampling['ttft_time_ms']:.2f}ms")

    finally:
        api.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
