#!/usr/bin/env python3
"""
Lightweight performance harness for DPT-Large TTNN.
Prints a perf header line and CSV row for the perf sheet.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image

import ttnn  # type: ignore
from models.demos.wormhole.dpt.tt.dpt_ttnn import DPTTTNN, DPTConfig


def prepare_input(image_path: Path, size: int) -> torch.Tensor:
    if not image_path.exists():
        alt = next((p for p in image_path.parent.glob("*.jpg")), None)
        if alt is None:
            raise FileNotFoundError(f"No image found at {image_path} or in {image_path.parent}")
        image_path = alt
    image = Image.open(image_path).convert("RGB").resize((size, size))
    arr = (torch.from_numpy(np.array(image).astype(np.float32)) / 127.5) - 1.0  # [-1, 1]
    return arr.unsqueeze(0)  # NHWC with batch dim


def summarize(times_ms: Iterable[float]) -> dict:
    arr = np.array(list(times_ms), dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def run_benchmark(model: DPTTTNN, tt_input, device, warmup: int, iters: int) -> List[float]:
    timings: List[float] = []
    for _ in range(max(warmup, 0)):
        _ = model(tt_input)
        ttnn.synchronize_device(device)
    for _ in range(iters):
        start = time.perf_counter()
        _ = model(tt_input)
        ttnn.synchronize_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed_ms)
    return timings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=Path("models/demos/wormhole/dpt/assets/dog.jpg"))
    ap.add_argument("--weights", type=Path, default=Path("models/demos/wormhole/dpt/weights"))
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--size", type=int, default=384, help="Square resize for perf runs")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV file to append perf row")
    args = ap.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    try:
        host_input = prepare_input(args.image, args.size)
        host_input = host_input.contiguous()  # already NHWC
        tt_input = ttnn.from_torch(
            host_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        model = DPTTTNN(DPTConfig(image_size=args.size), args.weights)

        times = run_benchmark(model, tt_input, device, args.warmup, args.iters)
        stats = summarize(times)

        header = "PERF_HEADER,model,batch,height,width,iters,mean_ms,p50_ms,p90_ms,p99_ms,min_ms,max_ms"
        row = (
            "PERF_DPT_LARGE_TTNN",
            1,
            args.size,
            args.size,
            args.iters,
            stats["mean_ms"],
            stats["p50_ms"],
            stats["p90_ms"],
            stats["p99_ms"],
            stats["min_ms"],
            stats["max_ms"],
        )
        print(header)
        print("PERF,{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(*row))

        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            file_exists = args.csv.exists()
            with open(args.csv, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header.split(",")[1:])
                writer.writerow(row)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
