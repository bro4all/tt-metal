"""
Patch embedding for DPT (ViT-L/16).
Takes NHWC input normalized to [-1,1], folds into patches, applies linear proj, and adds pos embeddings.
"""
from dataclasses import dataclass
import ttnn  # type: ignore
from typing import Tuple


@dataclass
class PatchEmbedConfig:
    image_size: int = 384
    patch_size: int = 16
    hidden_size: int = 1024
    dtype = ttnn.bfloat16


def patch_embed(x: ttnn.Tensor, proj_w, proj_b, pos_embed, cfg: PatchEmbedConfig) -> ttnn.Tensor:
    # x: NHWC
    patch_size = cfg.patch_size
    # fold spatial into sequence (similar to tt fold used in ViT demo)
    stride_h = patch_size
    stride_w = 1
    folded = ttnn.fold(x, stride_h, stride_w)
    folded = ttnn.to_layout(folded, layout=ttnn.TILE_LAYOUT, dtype=cfg.dtype)
    # linear projection
    out = ttnn.linear(folded, proj_w, bias=proj_b, dtype=cfg.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
    # add pos embedding (broadcast batch)
    pos = ttnn.from_torch(pos_embed, dtype=cfg.dtype, layout=ttnn.TILE_LAYOUT, device=out.device())
    out = ttnn.add(out, pos)
    return out
