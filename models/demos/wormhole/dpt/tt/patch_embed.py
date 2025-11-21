"""
Patch embedding for DPT (ViT-L/16).
Takes NHWC input normalized to [-1,1], folds into patches, applies linear proj, and adds pos embeddings.
"""
from dataclasses import dataclass
from typing import Tuple, Any
import numpy as np
import torch
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
    - conv-style patch projection
    - prepend CLS token
    - add positional embeddings (includes CLS position)
    """
    # x: NHWC
    patch_size = cfg.patch_size

    # Ensure weights live on device as TTNN tensors
    device = x.device()
    if not isinstance(proj_w, ttnn.Tensor):
        proj_w = ttnn.from_torch(
            torch.from_numpy(proj_w).contiguous(),
            dtype=cfg.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
    if not isinstance(proj_b, ttnn.Tensor):
        proj_b = ttnn.from_torch(
            torch.from_numpy(proj_b).contiguous(),
            dtype=cfg.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    x_rm = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=cfg.dtype)
    tokens = ttnn.conv2d(
        input_tensor=x_rm,
        weight_tensor=proj_w,
        bias_tensor=proj_b,
        device=device,
        in_channels=proj_w.shape[1],
        out_channels=proj_w.shape[0],
        batch_size=x_rm.shape[0],
        input_height=x_rm.shape[1],
        input_width=x_rm.shape[2],
        kernel_size=(patch_size, patch_size),
        stride=(patch_size, patch_size),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        dtype=cfg.dtype,
    )

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
