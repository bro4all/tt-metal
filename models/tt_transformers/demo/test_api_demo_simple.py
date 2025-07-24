#!/usr/bin/env python3

import logging
import os

import ttnn
from models.tt_transformers.demo.demo_api import create_demo_api
from models.tt_transformers.tt.model_config import DecodersPrecision

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup environment variables for TT-Metal."""
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    os.environ["ARCH_NAME"] = "wormhole_b0"
    os.environ["TT_METAL_HOME"] = os.getcwd()
    os.environ["PYTHONPATH"] = os.getcwd()
    os.environ["TMPDIR"] = "/tmp"
    os.environ["HF_MODEL"] = "meta-llama/Llama-3.2-3B"


def get_dispatch_core_config():
    """Get dispatch core configuration like in YOLOv10."""
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
    if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        dispatch_core_type = ttnn.device.DispatchCoreType.ETH
    dispatch_core_axis = ttnn.DispatchCoreAxis.ROW
    dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)
    return dispatch_core_config


def create_custom_device():
    """Create device with custom configuration like YOLOv10."""
    device_id = 0
    device = ttnn.CreateDevice(
        device_id,
        dispatch_core_config=get_dispatch_core_config(),
        l1_small_size=10 * 1024,
        trace_region_size=30 * 1024 * 1024,  # Match test_api_demo.py: 30MB
        num_command_queues=1,  # Match test_api_demo.py: 1 command queue
    )
    return device


def create_test_prompts(batch_size: int = 1):
    """Create test prompts for the benchmark."""
    base_prompts = [
        "Explain the benefits of speculative decoding in language models.",
        "What are the key performance metrics for LLM inference?",
        "How does speculative decoding improve time to first token?",
        "Describe the architecture of transformer models.",
        "What is the difference between prefill and decode phases?",
    ]

    # Repeat prompts to match batch size
    if len(base_prompts) < batch_size:
        repeated_prompts = []
        while len(repeated_prompts) < batch_size:
            repeated_prompts.extend(base_prompts)
        base_prompts = repeated_prompts

    # Take exactly batch_size prompts
    return base_prompts[:batch_size]


def test_api_performance_simple():
    """Test API performance with meta-llama/Llama-3.2-3B using test_api_demo.py patterns."""
    logger.info("Setting up environment...")
    setup_environment()

    # Test parameters (same as test_api_demo.py)
    batch_size = 1
    max_generated_tokens = 50
    sampling_params = {"temperature": 0, "top_p": 0.08}
    stop_at_eos = True
    max_seq_len = 1024
    data_parallel = 1
    paged_attention = True
    page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
    instruct = True

    logger.info("Creating custom device...")
    device = create_custom_device()
    logger.info(f"✓ Device created: {device}")

    # Create API instance (same as test_api_demo.py)
    logger.info("Creating API instance...")
    api = create_demo_api(
        mesh_device=device,
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
        input_prompts = create_test_prompts(batch_size)
        logger.info(f"Input prompts: {input_prompts}")

        # Skip warm-up to avoid tuple index error and go directly to inference
        logger.info("Skipping warm-up (to avoid tuple index error) and running full inference directly...")

        # Run full inference (same as test_api_demo.py)
        logger.info("Running full inference...")
        results = api.run_full_inference(
            input_prompts=input_prompts,
            max_generated_tokens=max_generated_tokens,
            sampling_params=sampling_params,
            stop_at_eos=stop_at_eos,
            enable_trace=True,  # Explicitly enable tracing for performance measurement
        )

        # Verify results (same as test_api_demo.py)
        assert "outputs" in results
        assert "performance_metrics" in results
        assert len(results["outputs"]) == batch_size
        assert results["num_prompts"] == batch_size

        # Print performance metrics (same as test_api_demo.py)
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

        # Print some outputs (same as test_api_demo.py)
        logger.info("=== Sample Outputs ===")
        for i, output in enumerate(results["outputs"][:3]):  # Show first 3 outputs
            logger.info(f"User {i}: {output[:200]}...")

        # Performance assertions (same as test_api_demo.py)
        assert metrics["total_tokens"] > 0
        assert metrics["decode_t/s/u"] > 0
        assert metrics["decode_t/s"] > 0

        logger.info("✓ API performance test completed successfully")

    finally:
        # Cleanup
        api.cleanup()


if __name__ == "__main__":
    test_api_performance_simple()
