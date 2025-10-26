# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pandas as pd
import torch
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename

import pytest
import ttnn
from models.demos.gemma3.tt.gemma_vision_model import TtGemmaTransformerVision
from models.demos.gemma3.tt.model_config import ModelArgs
from models.demos.llama3_70b_galaxy.tests.test_prefill_device_perf import (
    average_per_instance_dict,
    build_duration_dict,
    build_duration_per_instance_dict,
    merge_device_rows,
)
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf
from models.tt_transformers.tt.ccl import TT_CCL

THRESHOLD_PERCENT = 5

SAVE_NEW_PERF_TARGETS = False
TARGETS_JSON_FILENAME = (
    "models/demos/gemma3/tests/perf_targets/targets_test_perf_vision_cross_attention_transformer.json"
)


@pytest.mark.parametrize("device_params", [{"fabric_config": True, "l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), (1, 8))],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("nr_forward_iterations", [15])
def test_perf_gemma_vision(mesh_device, batch_size, nr_forward_iterations):
    profiler = BenchmarkProfiler()

    logger.info("Started profiling")
    profiler.start("total_run")
    run_model(
        mesh_device=mesh_device,
        batch_size=batch_size,
        profiler=profiler,
        nr_forward_iterations=nr_forward_iterations,
    )
    profiler.end("total_run")
    logger.info("Ended profiling")

    inference_measurements = [profiler.get_duration("model_forward_inference", i) for i in range(nr_forward_iterations)]
    inference_mean = sum(inference_measurements) / len(inference_measurements)

    measurement_keys = {k for _, k in profiler.start_times.keys()}

    measurements = dict()
    for k in measurement_keys:
        measurements[k] = profiler.get_duration(k) if k != "model_forward_inference" else inference_mean
        logger.info(f"measurement {k}: {measurements[k]}")

    targets = load_targets(
        TARGETS_JSON_FILENAME,
        device_type=determine_device_name(mesh_device),
    )

    if SAVE_NEW_PERF_TARGETS:
        helper_write_to_json(
            determine_device_name(mesh_device),
            measurements["model_forward_inference"],
            output_filename=TARGETS_JSON_FILENAME,
        )

    upper_threshold = targets["model_forward_inference"] * (1 + THRESHOLD_PERCENT / 100)
    lower_threshold = targets["model_forward_inference"] * (1 - THRESHOLD_PERCENT / 100)
    assert lower_threshold < inference_mean < upper_threshold


def helper_write_to_json(device_type, measurements, output_filename):
    """
    This function reads the file /output_filename/ and updates it with the new measurements. For example if the file has measurements for N150 it will overwrite them with the new measurements.
    """

    with open(output_filename, "r") as f:
        file_dict = json.load(f)

    file_dict[device_type] = {"model_forward_inference": measurements}

    with open(output_filename, "w") as f:
        json.dump(file_dict, f, indent=4)


def run_model(mesh_device, batch_size, profiler, nr_forward_iterations):
    dtype = ttnn.bfloat16
    model_args = ModelArgs(mesh_device)
    profiler.start("weight_loading")
    state_dict = model_args.load_state_dict()
    profiler.end("weight_loading")

    image_size = model_args.vision_chunk_size
    in_channels = model_args.vision_in_channels

    input_tensor = torch.rand((batch_size, in_channels, image_size, image_size))

    profiler.start("weight_transfer_to_device_and_model_initialization")
    model = TtGemmaTransformerVision(
        mesh_device,
        tt_ccl=TT_CCL(mesh_device),
        state_dict=state_dict,
        state_dict_prefix="model.vision_tower.vision_model.",
        dtype=dtype,
        configuration=model_args,
        return_intermediate=False,
    )
    profiler.end("weight_transfer_to_device_and_model_initialization")

    ttnn.synchronize_device(mesh_device)
    profiler.start("model_forward_compile")
    test_output = model(input_tensor)
    ttnn.synchronize_device(mesh_device)
    profiler.end("model_forward_compile")

    for cur_inference_iteration in range(nr_forward_iterations):
        profiler.start("model_forward_inference", cur_inference_iteration)
        test_output = model(input_tensor)
        ttnn.synchronize_device(mesh_device)
        profiler.end("model_forward_inference", cur_inference_iteration)

    profiler.start("postprocessing_and_transfer")
    out = ttnn.from_device(test_output)

    tt_output_torch = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[0, :, :, :]
    profiler.end("postprocessing_and_transfer")

    return tt_output_torch


def load_targets(filename, device_type):
    if not os.path.exists(filename):
        logger.warning(f"Expected outputs file {filename} does not exist. Skipping loading targets.")
        return []

    with open(filename, "r") as f:
        try:
            targets = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filename}: {e}. Returning empty list.")
            return []

    dict_targets = targets[device_type]

    return dict_targets


def compare_with_target(kernel_duration_per_instance_averaged_dict, perf_targets, profiler, margins):
    # benchmark_data = BenchmarkData()
    step_name = "gemma_vision_cross_attention_transformer_op_to_op_perf"
    passing = True
    for op_index, op_code_with_id in enumerate(kernel_duration_per_instance_averaged_dict.keys()):
        if op_code_with_id in perf_targets:
            op_name = op_code_with_id

            avg_kernel_duration = kernel_duration_per_instance_averaged_dict[op_code_with_id]

            # # average
            # benchmark_data.add_measurement(profiler, 0, step_name, op_name + "-model-kernel-avg", avg_kernel_duration)

            # Verify kernel duration is within tolerance
            upper_limit = perf_targets[op_code_with_id] + margins[op_code_with_id] * perf_targets[op_code_with_id]
            lower_limit = perf_targets[op_code_with_id] - margins[op_code_with_id] * perf_targets[op_code_with_id]

            if avg_kernel_duration > upper_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} kernel: {avg_kernel_duration} ns is larger than target "
                    f"({perf_targets[op_code_with_id]}) ns, difference: "
                    f"{abs(avg_kernel_duration - upper_limit)} ns, margin: "
                    f"{margins[op_code_with_id]}, "
                    f"relative margin to pass would be: "
                    f"{abs(perf_targets[op_code_with_id] - avg_kernel_duration) / perf_targets[op_code_with_id]}"
                )
            elif avg_kernel_duration < lower_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} kernel: {avg_kernel_duration} ns is smaller than target "
                    f"({perf_targets[op_code_with_id]}) ns, difference: "
                    f"{abs(lower_limit - avg_kernel_duration)} ns, margin: "
                    f"{margins[op_code_with_id]}, "
                    f"relative margin to pass would be: "
                    f"{abs(perf_targets[op_code_with_id] - avg_kernel_duration) / perf_targets[op_code_with_id]}"
                )
        else:
            passing = False
            logger.info(f"Warning: {op_code_with_id} not found in perf_targets")

    assert passing, "One or more ops did not meet performance targets. Check logs for details."


@pytest.mark.models_device_performance_bare_metal
def test_op_to_op_perf_gemma_vision():
    profiler = BenchmarkProfiler()
    batch_size = 1
    subdir = f"ttnn_gemma_cross_attention_perf"
    num_iterations = 1
    command = f"pytest models/demos/gemma3/tests/test_vision_cross_attention_transformer.py::test_gemma_vision"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    profiler.start("run")
    profiler.start("PROFILLING OP TO OP")
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, has_signposts=False)
    profiler.end("PROFILLING OP TO OP")
    profiler.end("run")

    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)
    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    df = merge_device_rows(df)

    ops_raw_dict = df[["OP CODE", "DEVICE KERNEL DURATION [ns]"]].to_dict(orient="records")
    kernel_duration_dict = build_duration_dict(ops_raw_dict, "DEVICE KERNEL DURATION [ns]")
    kernel_duration_per_instance_dict = build_duration_per_instance_dict(kernel_duration_dict, 1)

    # Average over all iterations of each op instance (in this specific case it is the same)
    kernel_duration_per_instance_averaged_dict = average_per_instance_dict(kernel_duration_per_instance_dict)

    expected_perf_cols = {}
    margins = {}
    with open(
        f"models/demos/gemma3/tests/perf_targets/targets_test_perf_vision_cross_attention_op_to_op.json", "r"
    ) as f:
        expected_perf_cols = json.load(f)
    with open(
        f"models/demos/gemma3/tests/perf_targets/targets_margins_test_perf_vision_cross_attention_op_to_op.json", "r"
    ) as f:
        margins = json.load(f)
    compare_with_target(kernel_duration_per_instance_averaged_dict, expected_perf_cols, profiler, margins)
