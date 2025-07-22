#!/usr/bin/env python3

import subprocess
import argparse
import math
import os
import logging
from datetime import datetime, timedelta

# Config
TEST_FILE = "models/experimental/stable_diffusion_xl_base/tests/test_sdxl_stability.py"
ENV_VAR = "SDXL_NUM_IMAGES_PER_DEVICE"
LOG_FILE = "sdxl_stress_test.log"


def format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_test(num_images_per_device, chunk_idx, total_chunks):
    logging.info(f"Starting chunk {chunk_idx + 1}/{total_chunks} with {num_images_per_device} images per device...")

    start_time = datetime.now()

    env = os.environ.copy()
    env[ENV_VAR] = str(num_images_per_device)

    result = subprocess.run(["pytest", TEST_FILE, "-k", "test_sdxl_stress"], env=env)

    end_time = datetime.now()
    duration = end_time - start_time

    if result.returncode != 0:
        logging.error(f"Chunk {chunk_idx + 1} FAILED after {format_timedelta(duration)}.")
        exit(result.returncode)
    else:
        logging.info(f"Chunk {chunk_idx + 1} completed successfully in {format_timedelta(duration)}.")


def main():
    parser = argparse.ArgumentParser(description="Run SDXL stress test in chunks.")
    parser.add_argument("--total-images-per-device", type=int, required=True, help="Total images per device to run")
    args = parser.parse_args()

    # Delete old log file BEFORE configuring logging
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # Setup logging AFTER log file deletion
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )

    total = args.total_images_per_device
    chunk_size = total
    num_chunks = math.ceil(total / chunk_size)

    logging.info(f"=== Starting SDXL stress test with total {total} images per device in {num_chunks} chunk(s) ===")

    overall_start = datetime.now()

    for i in range(num_chunks):
        this_chunk = chunk_size if (i + 1) * chunk_size <= total else total - i * chunk_size
        run_test(this_chunk, i, num_chunks)

    overall_end = datetime.now()
    total_duration = overall_end - overall_start

    logging.info(f"=== All chunks completed successfully in total time {format_timedelta(total_duration)} ===")


if __name__ == "__main__":
    main()
