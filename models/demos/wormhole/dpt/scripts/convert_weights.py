#!/usr/bin/env python3
"""
Convert HuggingFace Intel/dpt-large weights into TTNN-friendly numpy blobs.

Outputs:
- raw_state_dict.npz            : full FP32 state dict (mirrors HF keys)
- fused_qkv_state_dict.npz      : same but with attention qkv fused into single matrices
- manifest.json                 : metadata for downstream loaders

Notes:
- Intended to run on a host with torch/transformers installed (e.g., the Koyeb worker).
- Fused QKV are packed as:
    {layer_prefix}/attn/qkv.weight shape (3*hidden, hidden)
    {layer_prefix}/attn/qkv.bias   shape (3*hidden)
  and the original query/key/value entries are dropped from fused dict.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from transformers import DPTForDepthEstimation


def fuse_qkv(sd: Dict[str, torch.Tensor], num_layers: int) -> Dict[str, torch.Tensor]:
    fused = {}
    for k, v in sd.items():
        fused[k] = v

    for i in range(num_layers):
        prefix = f"backbone.encoder.layer.{i}.attention.attention."
        q_w = fused.pop(prefix + "query.weight")
        k_w = fused.pop(prefix + "key.weight")
        v_w = fused.pop(prefix + "value.weight")
        q_b = fused.pop(prefix + "query.bias")
        k_b = fused.pop(prefix + "key.bias")
        v_b = fused.pop(prefix + "value.bias")
        fused[prefix + "qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        fused[prefix + "qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)
    return fused


def save_npz(tensors: Dict[str, torch.Tensor], path: Path) -> None:
    np.savez_compressed(path, **{k: v.cpu().numpy() for k, v in tensors.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Intel/dpt-large")
    ap.add_argument("--outdir", type=Path, default=Path("weights/converted"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    model = DPTForDepthEstimation.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to(args.device)
    sd = model.state_dict()
    num_layers = model.config.num_hidden_layers

    # Raw dump
    raw_path = args.outdir / "raw_state_dict.npz"
    save_npz(sd, raw_path)

    # Fused dump
    fused_sd = fuse_qkv(sd, num_layers)
    fused_path = args.outdir / "fused_qkv_state_dict.npz"
    save_npz(fused_sd, fused_path)

    manifest = {
        "model": args.model,
        "num_layers": num_layers,
        "hidden_size": model.config.hidden_size,
        "num_heads": model.config.num_attention_heads,
        "patch_size": model.config.patch_size,
        "image_size": model.config.image_size,
        "files": {
            "raw": str(raw_path),
            "fused_qkv": str(fused_path),
        },
    }
    with open(args.outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Saved:")
    print(f"  raw_state_dict   -> {raw_path}")
    print(f"  fused_qkv_state  -> {fused_path}")
    print(f"  manifest         -> {args.outdir/'manifest.json'}")


if __name__ == "__main__":
    main()
