"""
Patch embedding for DPT (ViT-L/16).
Takes NHWC input normalized to [-1,1], folds into patches, applies linear proj, and adds pos embeddings.
"""
import os
from dataclasses import dataclass
from typing import Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
import ttnn  # type: ignore


@dataclass
class PatchEmbedConfig:
    image_size: int = 384
    patch_size: int = 16
    hidden_size: int = 1024
    dtype: Any = ttnn.bfloat16


def patch_embed(x: ttnn.Tensor, proj_w, proj_b, pos_embed, cls_token, cfg: PatchEmbedConfig) -> ttnn.Tensor:
    """
    Match the DPT embedding stack:
    - conv-style patch projection (implemented via fold + linear)
    - prepend CLS token
    - add positional embeddings (includes CLS position)
    """
    # Optional torch fallback for correctness (avoids ambiguity in ttnn.fold ordering).
    use_torch = os.getenv("DPT_FORCE_TORCH_PATCH", "1") == "1"
    device = x.device()
    if use_torch:
        # x comes in NHWC; convert to NCHW for torch conv2d
        x_t = ttnn.to_torch(x).permute(0, 3, 1, 2).float()
        w_t = proj_w if isinstance(proj_w, torch.Tensor) else torch.from_numpy(proj_w)
        b_t = proj_b if isinstance(proj_b, torch.Tensor) else torch.from_numpy(proj_b)
        # conv stride == patch size replicates ViT patch projection
        feat = F.conv2d(x_t, w_t, bias=b_t, stride=cfg.patch_size, padding=0)
        tokens_t = feat.flatten(2).transpose(1, 2)  # (B, seq, hidden)
        cls_t = torch.from_numpy(cls_token).repeat(tokens_t.shape[0], 1, 1).to(tokens_t)
        tokens_t = torch.cat([cls_t, tokens_t], dim=1)
        pos_t = torch.from_numpy(pos_embed).to(tokens_t)
        tokens_t = tokens_t + pos_t
        return ttnn.from_torch(tokens_t.contiguous(), dtype=cfg.dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Default TTNN path (uses fold + linear)
    patch_size = cfg.patch_size
    # fold spatial into sequence of flattened patches
    folded = ttnn.fold(x, stride_h=patch_size, stride_w=patch_size)
    folded = ttnn.to_layout(folded, layout=ttnn.TILE_LAYOUT, dtype=cfg.dtype)

    if not isinstance(proj_b, ttnn.Tensor):
        proj_b = ttnn.from_torch(
            torch.from_numpy(proj_b).contiguous(),
            dtype=cfg.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    # linear projection expects weight [in=3*patch*patch, out=hidden]
    if isinstance(proj_w, ttnn.Tensor):
        proj_w_tt = proj_w
    else:
        proj_w_flat = torch.from_numpy(proj_w).reshape(cfg.hidden_size, -1).t().contiguous()
        proj_w_tt = ttnn.from_torch(
            proj_w_flat, dtype=cfg.dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
    tokens = ttnn.linear(folded, proj_w_tt, bias=proj_b, dtype=cfg.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)

    # flatten spatial (H/ps, W/ps) -> sequence
    patch_grid = cfg.image_size // patch_size
    seq_len = patch_grid * patch_grid
    tokens = ttnn.reshape(tokens, (tokens.shape[0], seq_len, cfg.hidden_size))

    # prepend CLS token (broadcast across batch)
    batch = tokens.shape[0]
    cls = np.repeat(cls_token, repeats=batch, axis=0)  # (B, 1, hidden)
    cls_tt = ttnn.from_torch(
        torch.from_numpy(cls).contiguous(), dtype=cfg.dtype, layout=ttnn.TILE_LAYOUT, device=tokens.device()
    )
    tokens = ttnn.concat([cls_tt, tokens], dim=1)

    # add positional embeddings
    pos = ttnn.from_torch(
        torch.from_numpy(pos_embed).contiguous(),
        dtype=cfg.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=tokens.device(),
    )
    tokens = ttnn.add(tokens, pos)
    return tokens
