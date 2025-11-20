"""
TTNN implementation scaffold for DPT-Large.
Fill in: patch embedding, 24-layer ViT-L/16 encoder, reassembly blocks, fusion head, and run() helper.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import ttnn  # type: ignore
import numpy as np
from .vit_layer import ViTLayerConfig, ViTLayerTTNN


@dataclass
class DPTConfig:
    image_size: int = 384
    patch_size: int = 16
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    features: int = 256
    taps: tuple[int, int, int, int] = (6, 12, 18, 24)
    non_negative: bool = True
    dtype = ttnn.bfloat16


class DPTTTNN:
    def __init__(self, cfg: DPTConfig, weights_dir: Path):
        self.cfg = cfg
        self.weights_dir = weights_dir
        # load converted weights, set up sharding and fused ops (later)
        self.weights = load_weights(weights_dir)
        self.vit_cfg = ViTLayerConfig(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            seq_len=(cfg.image_size // cfg.patch_size) ** 2,
            dtype=cfg.dtype,
        )
        self.layers = [
            ViTLayerTTNN(self.vit_cfg, self.weights, layer_idx=i) for i in range(cfg.num_layers)
        ]

    def __call__(self, images: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            images: NHWC float tensor normalized to [-1,1], shape (N, H, W, 3)
        Returns:
            depth tensor (N, H, W, 1) in tile layout
        """
        # TODO:
        # 1) Patch embed (fold -> linear) + pos embed add
        # 2) Encoder over 24 layers, collect taps (6,12,18,24)
        # 3) Reassembly + fusion + refinement head
        raise NotImplementedError


def load_weights(weights_dir: Path) -> Dict[str, Any]:
    """Placeholder for weight loading/format conversion."""
    fused_path = weights_dir / "fused_qkv_state_dict.npz"
    if not fused_path.exists():
        raise FileNotFoundError(f"Missing fused weights at {fused_path}")
    npz = np.load(fused_path)
    return {k: npz[k] for k in npz.files}
