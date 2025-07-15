# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.utility_functions import comp_ulp


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ([512, 512], torch.bfloat16),
    ],
)
def test_bench_pow(shape, dtype, device):
    # Create 1D tensors with single elements: a=9, b=2
    torch_a = torch.rand(shape, dtype=dtype)
    torch_b = torch.rand(shape, dtype=dtype)

    # Perform torch power operation
    torch_result = torch.pow(torch_a, torch_b)

    # Convert to ttnn tensors with TILE layout and L1 memory config
    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.Layout.TILE, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.Layout.TILE, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    print(f"Calling ttnn.pow")
    # Perform ttnn power operation

    for _ in range(0, 4):
        ttnn_result = ttnn.pow(ttnn_a, ttnn_b)

    # Convert ttnn result back to torch for comparison
    ttnn_result_torch = ttnn.to_torch(ttnn_result)

    # Check if results match
    ulp_threshold = 10
    match_status = comp_ulp(torch_result, ttnn_result_torch, ulp_threshold)

    # match_status = torch.allclose(torch_result, ttnn_result_torch, rtol=1e-3, atol=1e-3)


def benchmark_pow(device, dtype):
    # Create 1D tensors with single elements: a=9, b=2
    # torch_a = torch.tensor([9.0, 100000.0, 5.0, 10.0, -1, -1, -1, torch.nan, 0.0, 0.0], dtype=dtype)
    # torch_b = torch.tensor([2.0, 1.7984, 3.0, -1.0, 3.5, 2, 3, 1, 1.8828, -9.2578e-01], dtype=dtype)
    torch_a = torch.tensor([9.0], dtype=dtype)
    torch_b = torch.tensor([2.0], dtype=dtype)

    # Perform torch power operation
    torch_result = torch.pow(torch_a, torch_b)

    # Convert to ttnn tensors with TILE layout and L1 memory config
    ttnn_a = ttnn.from_torch(torch_a, layout=ttnn.Layout.TILE, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_b = ttnn.from_torch(torch_b, layout=ttnn.Layout.TILE, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Perform ttnn power operation
    ttnn_result = ttnn.pow(ttnn_a, ttnn_b)

    # Convert ttnn result back to torch for comparison
    ttnn_result_torch = ttnn.to_torch(ttnn_result)

    # Check if results match
    ulp_threshold = 10
    match_status = comp_ulp(torch_result, ttnn_result_torch, ulp_threshold)

    return torch_result, ttnn_result_torch, match_status


def main():
    """
    Main function to run the benchmark and display results
    """
    print("Testing ttnn.pow vs torch.pow comparison")
    print("=" * 50)

    # Initialize device
    device = ttnn.open_device(device_id=0)
    torch.set_printoptions(precision=20)

    try:
        # Run the benchmark
        torch_result, ttnn_result, match_status = benchmark_pow(device, torch.bfloat16)
        torch_result_f32, ttnn_result_f32, match_status = benchmark_pow(device, torch.float32)

        # Display results
        print(f"Input values: a=9, b=2")
        print(f"Computing a**b (9**2):")
        print(f"")
        print(f"-----")
        print(f"[bfloat16] Torch result:  {torch_result}")
        print(f"[bfloat16] TTNN result:   {ttnn_result}")
        print(f"-----")
        print(f"[float32 ] Torch result:  {torch_result_f32}")
        print(f"[float32 ] TTNN result:   {ttnn_result_f32}")
        print(f"")
        print(f"Results match: {match_status}")
        print(f"Expected value: 81.0")

        # Additional information
        print(f"")
        print(f"Data type: bfloat16")
        print(f"Tensor shape: [1] (1D with single element)")
        print(f"TTNN layout: TILE")

        if match_status:
            print(f"✓ Test PASSED: ttnn.pow matches torch.pow")
        else:
            print(f"✗ Test FAILED: ttnn.pow does not match torch.pow")
            print(f"Difference: {torch.abs(torch_result - ttnn_result)}")

    finally:
        # Clean up device
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
