# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import csv
import os
import urllib
from loguru import logger
import statistics
import json

from models.experimental.stable_diffusion_xl_base.utils.clip_encoder import CLIPEncoder
from models.experimental.stable_diffusion_35_large.tests.utils.fid_score import calculate_fid_score
from models.experimental.stable_diffusion_35_large.tt.fun_pipeline import TtStableDiffusion3Pipeline
from models.experimental.stable_diffusion_35_large.tt.parallel_config import (
    StableDiffusionParallelManager,
    EncoderParallelManager,
    create_vae_parallel_config,
)
import ttnn
from models.utility_functions import profiler
from loguru import logger


# currently same as sdxl
COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"
OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "sd35_test_results.json"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 25000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        [(2, 4), (2, 1), (2, 0), (2, 1), ttnn.Topology.Linear, 1],
        # [(4, 8), (2, 1), (4, 0), (4, 1), ttnn.Topology.Linear, 3],
    ],
    ids=[
        "t3k_cfg2_sp2_tp2",
        # "tg_cfg2_sp4_tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps",
    [
        ("large", 1024, 1024, 3.5, 28),
    ],
)
@pytest.mark.parametrize("captions_path", ["models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_sd35(
    mesh_device,
    model_name,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    cfg,
    sp,
    tp,
    topology,
    num_links,
    captions_path,
    coco_statistics_path,
    evaluation_range,
    model_location_generator,
):
    start_from, num_prompts = evaluation_range
    prompts = sd35_get_prompts(captions_path, start_from, num_prompts)
    logger.info(f"start inference from prompt index: {start_from} to {start_from + num_prompts}")

    cfg_factor, cfg_axis = cfg
    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp
    parallel_manager = StableDiffusionParallelManager(
        mesh_device,
        cfg_factor,
        sp_factor,
        tp_factor,
        sp_factor,
        tp_factor,
        topology,
        cfg_axis=cfg_axis,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
    )

    guidance_cond = 2 if (guidance_scale > 1 and cfg_factor == 1) else 1

    # temp reshape submesh 0 to 1x4 for clip
    encoder_device = parallel_manager.submesh_devices[0]
    if parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape[1] != 4:
        # if reshaping, keep vae on submesh 0 and disable t5
        vae_device = parallel_manager.submesh_devices[0]
        enable_t5_text_encoder = False
        cfg_shape = parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape
        assert cfg_shape[0] * cfg_shape[1] == 4, f"cannot reshape {cfg_shape} to a 1x4 mesh"
        logger.info(f"reshaping submesh device 0 from {cfg_shape} to (1, 4) for clip")
        encoder_device.reshape(ttnn.MeshShape(1, 4))
    else:
        # if no reshape, vae on submesh 1 and we can enable t5
        vae_device = parallel_manager.submesh_devices[1]
        enable_t5_text_encoder = True

    encoder_parallel_manager = EncoderParallelManager(
        encoder_device,
        topology,
        mesh_axis=1,  # 1x4 submesh, parallel on axis 1
        num_links=num_links,
    )
    vae_parallel_manager = create_vae_parallel_config(vae_device, parallel_manager)

    # restore submesh 0 shape so dit runs with intended mesh
    parallel_manager.submesh_devices[0].reshape(
        ttnn.MeshShape(*parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape)
    )

    pipeline = TtStableDiffusion3Pipeline(
        checkpoint_name=f"stabilityai/stable-diffusion-3.5-{model_name}",
        mesh_device=mesh_device,
        enable_t5_text_encoder=enable_t5_text_encoder,
        guidance_cond=guidance_cond,
        parallel_manager=parallel_manager,
        encoder_parallel_manager=encoder_parallel_manager,
        vae_parallel_manager=vae_parallel_manager,
        height=image_h,
        width=image_w,
        model_location_generator=model_location_generator,
    )

    pipeline.prepare(
        batch_size=1,
        width=image_w,
        height=image_h,
        guidance_scale=guidance_scale,
        prompt_sequence_length=333,
        spatial_sequence_length=4096,
    )

    images = []
    for prompt in prompts:
        negative_prompt = ""

        profiler.start("denoising_loop")
        profiler.start("vae_decode")

        generated_images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            prompt_3=[prompt],
            negative_prompt_1=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            negative_prompt_3=[negative_prompt],
            num_inference_steps=num_inference_steps,
            seed=0,
            traced=True,
        )

        profiler.end("denoising_loop")
        profiler.end("vae_decode")

        images.append(generated_images[0])

    clip = CLIPEncoder()
    clip_scores = [100 * clip.get_clip_score(prompts[i], img).item() for i, img in enumerate(images)]
    average_clip_score = sum(clip_scores) / len(clip_scores)

    deviation_clip_score = "N/A"
    fid_score = "N/A"
    if num_prompts >= 2 and os.path.isfile(coco_statistics_path):
        deviation_clip_score = statistics.stdev(clip_scores)
        fid_score = calculate_fid_score(images, coco_statistics_path)
    elif num_prompts >= 2 and not os.path.isfile(coco_statistics_path):
        logger.warning(f"fid skipped: stats file not found at {coco_statistics_path}")

    print(f"FID score: {fid_score}")
    print(f"Average CLIP Score: {average_clip_score}")
    print(f"Standard Deviation of CLIP Scores: {deviation_clip_score}")

    data = {
        "model": "sd35",
        "metadata": {
            "device": "T3K",
            "model_name": model_name,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "image_width": image_w,
            "image_height": image_h,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "cfg_factor": cfg_factor,
            "sp_factor": sp_factor,
            "tp_factor": tp_factor,
        },
        "benchmarks_summary": [
            {
                "device": "T3K",
                "model": "sd35",
                "average_denoising_time": profiler.get("denoising_loop"),
                "average_vae_time": profiler.get("vae_decode"),
                "average_inference_time": profiler.get("denoising_loop") + profiler.get("vae_decode"),
                "min_inference_time": min(
                    i + j for i, j in zip(profiler.times["denoising_loop"], profiler.times["vae_decode"])
                ),
                "max_inference_time": max(
                    i + j for i, j in zip(profiler.times["denoising_loop"], profiler.times["vae_decode"])
                ),
                "average_clip": average_clip_score,
                "deviation_clip": deviation_clip_score,
                "fid_score": fid_score,
            }
        ],
    }

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")

    for submesh_device in parallel_manager.submesh_devices:
        ttnn.synchronize_device(submesh_device)


def sd35_get_prompts(captions_path, start_from, num_prompts):
    assert (
        0 <= start_from < 5000 and start_from + num_prompts <= 5000
    ), "start_from must be between 0 and 4999, and start_from + num_prompts must not exceed 5000."

    prompts = []
    if not os.path.isfile(captions_path):
        logger.info(f"file {captions_path} not found. downloading...")
        os.makedirs(os.path.dirname(captions_path), exist_ok=True)
        urllib.request.urlretrieve(COCO_CAPTIONS_DOWNLOAD_PATH, captions_path)
        logger.info("download complete.")

    with open(captions_path, "r") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        next(reader)  # skip header
        for index, row in enumerate(reader):
            if index < start_from:
                continue
            if index >= start_from + num_prompts:
                break
            prompts.append(row[2])
    return prompts
