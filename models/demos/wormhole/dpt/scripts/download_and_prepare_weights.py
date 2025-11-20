#!/usr/bin/env python3
"""
Download HF Intel/dpt-large weights and stage them for TTNN conversion.
Stage 1: snapshot repo locally (safetensors + config).
Stage 2 (TODO): emit numpy weights with fused QKV and TT-friendly layouts (bf16/bf8).
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="Intel/dpt-large",
        help="HuggingFace repo id",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("weights/hf_dpt_large"),
        help="Destination directory",
    )
    ap.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token if needed",
    )
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=args.model,
        ignore_patterns=["*.onnx", "*.msgpack", "*.json", "*.bin"],  # keep safetensors + config
        allow_patterns=["*"],
        local_dir=args.output,
        token=args.token,
    )
    print(f"Snapshot downloaded to {snapshot_path}")
    print("Next: convert safetensors to TTNN fused formats (QKV merge, bf8/bf16).")


if __name__ == "__main__":
    main()
