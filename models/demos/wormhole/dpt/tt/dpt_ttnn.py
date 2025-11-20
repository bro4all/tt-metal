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
from .reassembly import ReassemblyStage, ReassemblyConfig
from .fusion_head import FusionHead, FusionHeadConfig, FeatureFusionStage, FusionBlockConfig


@dataclass
class DPTConfig:
    image_size: int = 384
    patch_size: int = 16
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    features: int = 256
    neck_hidden_sizes: tuple[int, int, int, int] = (256, 512, 1024, 1024)
    reassemble_factors: tuple[float, float, float, float] = (4.0, 2.0, 1.0, 0.5)
    taps: tuple[int, int, int, int] = (6, 12, 18, 24)
    non_negative: bool = True
    head_in_index: int = -1
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
        self.cls_token = self.weights["dpt.embeddings.cls_token"]

        self.vit_cfg = ViTLayerConfig(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            seq_len=(cfg.image_size // cfg.patch_size) ** 2,
            dtype=cfg.dtype,
        )
        self.layers = [
            ViTLayerTTNN(self.vit_cfg, self.weights, layer_idx=i) for i in range(cfg.num_layers)
        ]

        self.reassembly = ReassemblyStage(
            ReassemblyConfig(
                hidden_size=cfg.hidden_size,
                neck_hidden_sizes=cfg.neck_hidden_sizes,
                reassemble_factors=cfg.reassemble_factors,
                fusion_hidden_size=cfg.features,
                dtype=cfg.dtype,
            ),
            self.weights,
        )
        self.fusion_stage = FeatureFusionStage(FusionBlockConfig(hidden_size=cfg.features, dtype=cfg.dtype), self.weights)

        # Fusion head weights (DPTDepthEstimationHead)
        self.fusion_head = FusionHead(
            FusionHeadConfig(features=cfg.features, dtype=cfg.dtype, head_in_index=cfg.head_in_index),
            self.weights["head.head.0.weight"],
            self.weights["head.head.0.bias"],
            self.weights["head.head.2.weight"],
            self.weights["head.head.2.bias"],
            self.weights["head.head.4.weight"],
            self.weights["head.head.4.bias"],
            None,
            None,
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
        x = patch_embed(images, self.patch_w, self.patch_b, self.pos_embed, self.cls_token, self.patch_cfg)

        # Encoder + taps
        taps = []
        for i, layer in enumerate(self.layers, 1):
            x = layer(x)
            if i in self.cfg.taps:
                taps.append(x)

        patch_hw = self.cfg.image_size // self.cfg.patch_size
        feats = self.reassembly(taps, patch_hw)
        fused_pyramid = self.fusion_stage(feats)
        depth = self.fusion_head(fused_pyramid)
        return depth


def load_weights(weights_dir: Path) -> Dict[str, Any]:
    """Placeholder for weight loading/format conversion."""
    fused_path = weights_dir / "fused_qkv_state_dict.npz"
    if not fused_path.exists():
        raise FileNotFoundError(f"Missing fused weights at {fused_path}")
    npz = np.load(fused_path)
    return {k: npz[k] for k in npz.files}
