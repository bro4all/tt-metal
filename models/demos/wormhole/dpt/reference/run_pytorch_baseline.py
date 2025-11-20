#!/usr/bin/env python3
"""
Run PyTorch DPT-Large (Intel/dpt-large) on a folder of images, save depth maps and basic metrics.
Designed as the ground-truth generator for TTNN PCC checks.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
import torch
from transformers import pipeline


def save_depth_png(depth: np.ndarray, path: Path) -> None:
    """Save depth to 8-bit PNG for quick visual checks."""
    d = depth - depth.min()
    d = d / (d.max() + 1e-8)
    img = (d * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient for flattened arrays."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def run(
    image_dir: Path,
    output_dir: Path,
    compare_dir: Optional[Path],
    device: str,
) -> Dict[str, Dict[str, float]]:
    pipe = pipeline(
        task="depth-estimation",
        model="Intel/dpt-large",
        device=0 if device.startswith("cuda") else -1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: Dict[str, Dict[str, float]] = {}

    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        result = pipe(Image.open(img_path))
        depth = np.array(result["depth"])

        stem = img_path.stem
        npz_path = output_dir / f"{stem}.npz"
        png_path = output_dir / f"{stem}.png"

        np.savez_compressed(npz_path, depth=depth)
        save_depth_png(depth, png_path)

        stats = {
            "min": float(depth.min()),
            "max": float(depth.max()),
            "mean": float(depth.mean()),
            "std": float(depth.std()),
        }

        if compare_dir is not None:
            ref_file = compare_dir / f"{stem}.npz"
            if ref_file.exists():
                ref_depth = np.load(ref_file)["depth"]
                stats["pcc_vs_ref"] = pcc(depth, ref_depth)

        metrics[stem] = stats

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Folder of input images",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder for depth maps/npz/metrics",
    )
    p.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional folder of reference npz files to compute PCC",
    )
    p.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HuggingFace token (export HF_TOKEN env is also fine)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cuda or cpu",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")

    metrics = run(args.images, args.output, args.compare, args.device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
