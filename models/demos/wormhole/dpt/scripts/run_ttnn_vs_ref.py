#!/usr/bin/env python3
"""
Run TTNN DPT-Large on sample assets and compare against PyTorch reference depth maps.

Expected usage (on a TT device host with ttnn installed):
    python models/demos/wormhole/dpt/scripts/run_ttnn_vs_ref.py \
        --images models/demos/wormhole/dpt/assets \
        --reference models/demos/wormhole/dpt/out_reference \
        --weights models/demos/wormhole/dpt/weights \
        --device-id 0

The script prints per-image PCC and saves optional outputs for debugging.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import DPTImageProcessor

import ttnn  # type: ignore
from models.demos.wormhole.dpt.tt.dpt_ttnn import DPTTTNN, DPTConfig


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient for flattened arrays."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def preprocess(image_path: Path, processor: DPTImageProcessor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    # pixel_values: [1, 3, H, W] already normalized to [-1, 1]
    pixel_values = inputs["pixel_values"].permute(0, 2, 3, 1).contiguous()  # NHWC
    return pixel_values, (image.height, image.width)


def upsample_to_original(depth: torch.Tensor, target_hw: Tuple[int, int]) -> np.ndarray:
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth.permute(0, 3, 1, 2)  # NHWC -> NCHW
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)  # assume N H W
    elif depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected depth tensor shape {tuple(depth.shape)}")

    depth = depth.float()
    depth = torch.nn.functional.interpolate(
        depth, size=target_hw, mode="bilinear", align_corners=False
    )
    return depth.squeeze(0).squeeze(0).cpu().numpy()


def run(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    weights_dir = Path(args.weights)
    ref_dir = Path(args.reference)
    images_dir = Path(args.images)
    save_dir = Path(args.save_dir) if args.save_dir else None

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    device = ttnn.open_device(device_id=args.device_id)
    try:
        processor = DPTImageProcessor.from_pretrained(args.hf_model)
        model = DPTTTNN(DPTConfig(), weights_dir)

        results: Dict[str, Dict[str, float]] = {}
        for img_path in sorted(images_dir.glob("*")):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue

            pixel_values, orig_hw = preprocess(img_path, processor)

            tt_input = ttnn.from_torch(
                pixel_values,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            tt_depth = model(tt_input)
            depth_torch = tt_depth if isinstance(tt_depth, torch.Tensor) else ttnn.to_torch(tt_depth)
            depth_np = upsample_to_original(depth_torch, orig_hw)

            stem = img_path.stem
            if save_dir:
                np.savez_compressed(save_dir / f"{stem}.npz", depth=depth_np)

            ref_path = ref_dir / f"{stem}.npz"
            ref_depth = np.load(ref_path)["depth"] if ref_path.exists() else None

            stats = {
                "min": float(depth_np.min()),
                "max": float(depth_np.max()),
                "mean": float(depth_np.mean()),
                "std": float(depth_np.std()),
            }
            if ref_depth is not None:
                # ensure shapes align for PCC
                if ref_depth.shape != depth_np.shape:
                    ref_depth = torch.from_numpy(ref_depth).unsqueeze(0).unsqueeze(0)
                    ref_depth = torch.nn.functional.interpolate(
                        ref_depth, size=orig_hw, mode="bilinear", align_corners=False
                    ).squeeze().numpy()
                stats["pcc_vs_ref"] = pcc(depth_np, ref_depth)

            if args.compare_hidden:
                hidden_file = ref_dir / f"{stem}_hidden.npz"
                if hidden_file.exists() and getattr(model, "debug_taps", None):
                    hf_hidden = np.load(hidden_file)
                    tap_pccs = []
                    for i, tap in enumerate(model.debug_taps):
                        # to_torch returns bfloat16; cast to float32 before NumPy to avoid unsupported dtype
                        tap_np = ttnn.to_torch(tap).float().cpu().numpy().ravel()
                        hf = hf_hidden.get(f"tap{i}")
                        if hf is None:
                            continue
                        tap_pccs.append(pcc(tap_np, hf.ravel()))
                    if tap_pccs:
                        stats["pcc_taps_mean"] = float(np.mean(tap_pccs))
                        stats["pcc_taps_min"] = float(np.min(tap_pccs))
                        stats["pcc_taps_max"] = float(np.max(tap_pccs))
                # Neck-level PCC (TT neck vs torch neck debug)
                if os.getenv("DPT_DEBUG_NECK", "0") == "1":
                    feats_tt = getattr(model, "debug_neck_feats_ttnn", None)
                    feats_th = getattr(model, "debug_neck_feats_torch", None)
                    fused_tt = getattr(model, "debug_fused_ttnn", None)
                    fused_th = getattr(model, "debug_fused_torch", None)
                    if feats_tt and feats_th:
                        neck_pccs = []
                        for tt, th in zip(feats_tt, feats_th):
                            tt_t = tt if isinstance(tt, torch.Tensor) else ttnn.to_torch(tt)
                            tt_np = tt_t.float().cpu().numpy().ravel()
                            th_np = th.cpu().numpy().ravel()
                            neck_pccs.append(pcc(tt_np, th_np))
                        stats["pcc_neck_mean"] = float(np.mean(neck_pccs))
                        stats["pcc_neck_min"] = float(np.min(neck_pccs))
                        stats["pcc_neck_max"] = float(np.max(neck_pccs))
                    if fused_tt and fused_th:
                        fused_pccs = []
                        for tt, th in zip(fused_tt, fused_th):
                            tt_t = tt if isinstance(tt, torch.Tensor) else ttnn.to_torch(tt)
                            tt_np = tt_t.float().cpu().numpy().ravel()
                            th_np = th.cpu().numpy().ravel()
                            fused_pccs.append(pcc(tt_np, th_np))
                        stats["pcc_fused_mean"] = float(np.mean(fused_pccs))
                        stats["pcc_fused_min"] = float(np.min(fused_pccs))
                        stats["pcc_fused_max"] = float(np.max(fused_pccs))
            results[stem] = stats

            pcc_str = f"{stats.get('pcc_vs_ref', 'N/A')}"
            if args.compare_hidden and "pcc_taps_mean" in stats:
                pcc_str += f" taps_mean={stats['pcc_taps_mean']:.4f}"
            if os.getenv("DPT_DEBUG_NECK", "0") == "1" and "pcc_neck_mean" in stats:
                pcc_str += f" neck_mean={stats['pcc_neck_mean']:.4f}"
            print(f"{stem}: pcc={pcc_str} shape={depth_np.shape} range=({stats['min']:.2f},{stats['max']:.2f})")

        if args.metrics:
            with open(args.metrics, "w") as f:
                json.dump(results, f, indent=2)
        return results
    finally:
        ttnn.close_device(device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=Path("models/demos/wormhole/dpt/assets"), help="Input image folder")
    p.add_argument(
        "--reference",
        type=Path,
        default=Path("models/demos/wormhole/dpt/out_reference"),
        help="Folder with PyTorch reference .npz files",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("models/demos/wormhole/dpt/weights"),
        help="Folder containing fused_qkv_state_dict.npz",
    )
    p.add_argument("--device-id", type=int, default=0, help="TT device id")
    p.add_argument(
        "--hf-model",
        type=str,
        default="Intel/dpt-large",
        help="HF model id for preprocessing metadata",
    )
    p.add_argument("--save-dir", type=Path, default=None, help="Optional folder to dump TTNN depth npz files")
    p.add_argument("--metrics", type=Path, default=None, help="Optional path to write JSON metrics")
    p.add_argument("--compare-hidden", action="store_true", help="Compare encoder taps against saved HF hidden taps")
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
