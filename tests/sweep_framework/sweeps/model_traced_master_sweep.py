# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import random
import torch
import ttnn

from tests.sweep_framework.master_config_loader import (
    MasterConfigLoader,
    unpack_traced_config,
    unpack_binary_traced_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)

# Load master config loader for all traced model configurations
loader = MasterConfigLoader()

# Get all operations that have traced configurations
operations_with_configs = loader.get_operations_with_configs()

# Create parameters for all operations in a single suite
parameters = {"model_traced_all": {}}

# Collect all operation + config combinations using the same logic as distributed approach
all_configs = []
processed_counts = {}

for operation_name in operations_with_configs.keys():
    # Use get_suite_parameters to get properly processed configs (with deduplication)
    op_params = loader.get_suite_parameters(operation_name)

    if op_params and "traced_config_name" in op_params:
        traced_config_names = op_params["traced_config_name"]
        processed_counts[operation_name] = len(traced_config_names)
        for config_name in traced_config_names:
            all_configs.append({"operation_name": operation_name, "traced_config_name": config_name})

# Add all combinations to the parameters
parameters["model_traced_all"] = {"operation_config": all_configs}

print(f"✅ Created centralized sweep with {len(all_configs)} total operation-config combinations:")
for op_name in sorted(processed_counts.keys()):
    processed_count = processed_counts[op_name]
    raw_count = operations_with_configs.get(op_name, 0)
    print(f"  {op_name}: {processed_count} processed configs (from {raw_count} raw)")
print(f"Total combinations: {len(all_configs)}")


def run_model_traced_operation(
    operation_config,
    *,
    device,
) -> list:
    """
    Run a single traced configuration for any operation.
    This centralized function handles all operations that have traced configs.
    """

    operation_name = operation_config["operation_name"]
    traced_config_name = operation_config["traced_config_name"]

    # Get the traced config details
    if traced_config_name is None:
        return [False, 0]  # No config to test

    # Determine if this is a unary or binary operation
    configs = loader.get_operation_configs(operation_name)
    if not configs:
        return [False, 0]  # No configs available

    is_binary = loader._is_binary_operation(configs)

    try:
        if is_binary:
            # Handle binary operations
            (
                input_shape_a,
                input_shape_b,
                input_a_dtype,
                input_b_dtype,
                input_a_layout,
                input_b_layout,
                input_a_memory_config,
                input_b_memory_config,
                output_memory_config,
            ) = unpack_binary_traced_config(traced_config_name)

            # Generate test data
            data_seed = random.randint(0, 20000000)
            torch.manual_seed(data_seed)

            torch_input_tensor_a = gen_func_with_cast_tt(
                partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
            )(input_shape_a)

            if isinstance(input_shape_b, list):
                torch_input_tensor_b = gen_func_with_cast_tt(
                    partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
                )(input_shape_b)
            else:
                torch_input_tensor_b = torch.tensor(input_shape_b, dtype=torch.float32)

            # Get golden function and compute expected output
            golden_function = ttnn.get_golden_function(getattr(ttnn, operation_name))
            torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

            # Create ttnn tensors
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )

            input_tensor_b = ttnn.from_torch(
                torch_input_tensor_b,
                dtype=input_b_dtype,
                layout=input_b_layout,
                device=device,
                memory_config=input_b_memory_config,
            )

            # Run the operation
            start_time = start_measuring_time()
            result = getattr(ttnn, operation_name)(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
            output_tensor = ttnn.to_torch(result)
            e2e_perf = stop_measuring_time(start_time)

        else:
            # Handle unary operations
            (
                input_shape,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                output_memory_config,
            ) = unpack_traced_config(traced_config_name)

            # Generate test data
            data_seed = random.randint(0, 20000000)
            torch.manual_seed(data_seed)

            torch_input_tensor_a = gen_func_with_cast_tt(
                partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
            )(input_shape)

            # Get golden function and compute expected output
            golden_function = ttnn.get_golden_function(getattr(ttnn, operation_name))
            torch_output_tensor = golden_function(torch_input_tensor_a)

            # Create ttnn tensor
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )

            # Run the operation
            start_time = start_measuring_time()
            result = getattr(ttnn, operation_name)(input_tensor_a, memory_config=output_memory_config)
            output_tensor = ttnn.to_torch(result)
            e2e_perf = stop_measuring_time(start_time)

        # Compare results
        return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]

    except Exception as e:
        print(f"❌ Error running {operation_name} with config {traced_config_name}: {e}")
        return [False, 0]


# Entry point for the sweep framework
def run(
    operation_config,
    *,
    device,
) -> list:
    """
    Entry point for the sweep framework.
    Runs any operation with its traced configuration.
    """
    return run_model_traced_operation(operation_config, device=device)
