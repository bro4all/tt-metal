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
from .patch_embed import PatchEmbedConfig, patch_embed
from .reassembly import ReassemblyBlock, ReassemblyConfig
from .fusion_head import FusionHead, FusionHeadConfig


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
        # stash common tensors
        self.patch_w = self.weights["dpt.embeddings.patch_embeddings.projection.weight"]
        self.patch_b = self.weights["dpt.embeddings.patch_embeddings.projection.bias"]
        self.pos_embed = self.weights["dpt.embeddings.position_embeddings"]

        self.vit_cfg = ViTLayerConfig(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            seq_len=(cfg.image_size // cfg.patch_size) ** 2,
            dtype=cfg.dtype,
        )
        self.layers = [
            ViTLayerTTNN(self.vit_cfg, self.weights, layer_idx=i) for i in range(cfg.num_layers)
        ]
        # Reassembly blocks (channels from DPT paper: 256,512,1024,1024 -> 256)
        self.reassembly = [
            ReassemblyBlock(ReassemblyConfig(in_channels=256, out_channels=cfg.features, upsample=True),
                            self.weights["dpt.scratch.layer1_rn.weight"],
                            self.weights["dpt.scratch.layer1_rn.bias"]),
            ReassemblyBlock(ReassemblyConfig(in_channels=512, out_channels=cfg.features, upsample=True),
                            self.weights["dpt.scratch.layer2_rn.weight"],
                            self.weights["dpt.scratch.layer2_rn.bias"]),
            ReassemblyBlock(ReassemblyConfig(in_channels=1024, out_channels=cfg.features, upsample=True),
                            self.weights["dpt.scratch.layer3_rn.weight"],
                            self.weights["dpt.scratch.layer3_rn.bias"]),
            ReassemblyBlock(ReassemblyConfig(in_channels=1024, out_channels=cfg.features, upsample=False),
                            self.weights["dpt.scratch.layer4_rn.weight"],
                            self.weights["dpt.scratch.layer4_rn.bias"]),
        ]
        # Fusion head weights
        self.fusion_head = FusionHead(
            FusionHeadConfig(features=cfg.features, dtype=cfg.dtype),
            self.weights["dpt.scratch.output_conv.0.weight"],
            self.weights["dpt.scratch.output_conv.0.bias"],
            self.weights["dpt.scratch.output_conv.2.weight"],
            self.weights["dpt.scratch.output_conv.2.bias"],
            self.weights["dpt.scratch.output_conv.4.weight"],
            self.weights["dpt.scratch.output_conv.4.bias"],
        )
        self.patch_cfg = PatchEmbedConfig(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            hidden_size=cfg.hidden_size,
            dtype=cfg.dtype,
        )

    def __call__(self, images: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            images: NHWC float tensor normalized to [-1,1], shape (N, H, W, 3)
        Returns:
            depth tensor (N, H, W, 1) in tile layout
        """
        # Patch embed + pos add
        x = patch_embed(images, self.patch_w, self.patch_b, self.pos_embed, self.patch_cfg)

        # Encoder + taps
        taps = {}
        for i, layer in enumerate(self.layers, 1):
            x = layer(x)
            if i in self.cfg.taps:
                taps[i] = x

        # Reassembly expects taps in order 6,12,18,24 (mapped to 0..3)
        feats = [
            self.reassembly[0](taps[self.cfg.taps[0]]),
            self.reassembly[1](taps[self.cfg.taps[1]]),
            self.reassembly[2](taps[self.cfg.taps[2]]),
            self.reassembly[3](taps[self.cfg.taps[3]]),
        ]

        # Simple top-down fusion (DPT uses FeatureFusionBlock_custom; placeholder here)
        fused = feats[-1]
        for f in reversed(feats[:-1]):
            fused = ttnn.add(fused, f)
        depth = self.fusion_head(fused)
        return depth


def load_weights(weights_dir: Path) -> Dict[str, Any]:
    """Placeholder for weight loading/format conversion."""
    fused_path = weights_dir / "fused_qkv_state_dict.npz"
    if not fused_path.exists():
        raise FileNotFoundError(f"Missing fused weights at {fused_path}")
    npz = np.load(fused_path)
    return {k: npz[k] for k in npz.files}
