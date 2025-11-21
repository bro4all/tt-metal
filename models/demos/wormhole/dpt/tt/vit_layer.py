"""
Minimal ViT encoder layer for DPT on TTNN.
Currently a skeleton: wire fused QKV + softmax + MLP using TTNN building blocks.
"""
from dataclasses import dataclass
from typing import Any
import torch
import ttnn  # type: ignore


@dataclass
class ViTLayerConfig:
    hidden_size: int = 1024
    num_heads: int = 16
    seq_len: int = (384 // 16) * (384 // 16) + 1  # include CLS token
    dtype: Any = ttnn.bfloat16
    # TODO: add per-op sharding configs when we tune perf


class ViTLayerTTNN:
    def __init__(self, cfg: ViTLayerConfig, weights, layer_idx: int):
        self.cfg = cfg
        self.w = weights
        self.idx = layer_idx
        # Prefetch tensor views for this layer
        base = f"dpt.encoder.layer.{layer_idx}."
        self.qkv_w = self.w[base + "attention.attention.qkv.weight"]
        self.qkv_b = self.w[base + "attention.attention.qkv.bias"]
        self.out_w = self.w[base + "attention.output.dense.weight"]
        self.out_b = self.w[base + "attention.output.dense.bias"]
        self.ff1_w = self.w[base + "intermediate.dense.weight"]
        self.ff1_b = self.w[base + "intermediate.dense.bias"]
        self.ff2_w = self.w[base + "output.dense.weight"]
        self.ff2_b = self.w[base + "output.dense.bias"]
        self.ln1_w = self.w[base + "layernorm_before.weight"]
        self.ln1_b = self.w[base + "layernorm_before.bias"]
        self.ln2_w = self.w[base + "layernorm_after.weight"]
        self.ln2_b = self.w[base + "layernorm_after.bias"]
        # TODO: prefetch sharding configs and tensors

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.cfg
        device = x.device()

        def to_tt(tensor, transpose: bool = False, layout=ttnn.ROW_MAJOR_LAYOUT):
            if isinstance(tensor, ttnn.Tensor):
                return tensor
            data = torch.from_numpy(tensor)
            if transpose and data.ndim == 2:
                data = data.t()
            return ttnn.from_torch(data.contiguous(), dtype=cfg.dtype, layout=layout, device=device)

        def ln_param(param):
            """
            Tile-layout gamma/beta padded to tile height (32) so LayerNorm kernel accepts it.
            Shape after expansion: [1, 1, 32, hidden].
            """
            if isinstance(param, ttnn.Tensor):
                return param
            data = torch.from_numpy(param)
            expanded = data.unsqueeze(0).repeat(32, 1).unsqueeze(0).unsqueeze(0)
            return ttnn.from_torch(
                expanded.contiguous(), dtype=cfg.dtype, layout=ttnn.TILE_LAYOUT, device=device
            )

        self.ln1_w = ln_param(self.ln1_w)
        self.ln1_b = ln_param(self.ln1_b)
        self.ln2_w = ln_param(self.ln2_w)
        self.ln2_b = ln_param(self.ln2_b)
        self.qkv_w = to_tt(self.qkv_w, transpose=True, layout=ttnn.TILE_LAYOUT)
        self.qkv_b = to_tt(self.qkv_b)
        self.out_w = to_tt(self.out_w, transpose=True, layout=ttnn.TILE_LAYOUT)
        self.out_b = to_tt(self.out_b)
        self.ff1_w = to_tt(self.ff1_w, transpose=True, layout=ttnn.TILE_LAYOUT)
        self.ff1_b = to_tt(self.ff1_b)
        self.ff2_w = to_tt(self.ff2_w, transpose=True, layout=ttnn.TILE_LAYOUT)
        self.ff2_b = to_tt(self.ff2_b)
        # LayerNorm before attention
        ln1 = ttnn.layer_norm(x, weight=self.ln1_w, bias=self.ln1_b, epsilon=1e-5)

        # Fused QKV linear
        qkv = ttnn.linear(ln1, self.qkv_w, bias=self.qkv_b, dtype=cfg.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
        # Sanity-check the fused projection produced the expected 3 * hidden width before splitting.
        expected_last = cfg.hidden_size * 3
        actual_last = qkv.shape[-1]
        if actual_last != expected_last:
            raise RuntimeError(
                f"[ViT layer {self.idx}] fused QKV has last_dim={actual_last}, expected {expected_last}; "
                f"full shape={qkv.shape}"
            )

        # reshape QKV: [B, seq, 3, heads, head_dim]
        head_dim = cfg.hidden_size // cfg.num_heads
        seq_len = qkv.shape[1]
        # Avoid ttnn.split (currently asserting inside split_with_slice_impl on WH) by reshaping
        # and slicing the stacked Q/K/V dimension manually.
        qkv = ttnn.reshape(qkv, (qkv.shape[0], seq_len, 3, cfg.hidden_size))
        q = ttnn.slice(
            qkv, (0, 0, 0, 0), (qkv.shape[0], seq_len, 1, cfg.hidden_size), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k = ttnn.slice(
            qkv, (0, 0, 1, 0), (qkv.shape[0], seq_len, 2, cfg.hidden_size), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        v = ttnn.slice(
            qkv, (0, 0, 2, 0), (qkv.shape[0], seq_len, 3, cfg.hidden_size), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # reshape to B, heads, seq, head_dim for proper attention computation
        q = ttnn.reshape(q, (qkv.shape[0], seq_len, cfg.num_heads, head_dim))
        k = ttnn.reshape(k, (qkv.shape[0], seq_len, cfg.num_heads, head_dim))
        v = ttnn.reshape(v, (qkv.shape[0], seq_len, cfg.num_heads, head_dim))
        q = ttnn.transpose(q, 1, 2)  # B, heads, seq, head_dim
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.transpose(v, 1, 2)
        # Use row-major layout for matmul stability
        q = ttnn.to_layout(q, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=cfg.dtype)
        k = ttnn.to_layout(k, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=cfg.dtype)
        v = ttnn.to_layout(v, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=cfg.dtype)

        # scaled dot-product attention over sequence dimension
        attn_scores = ttnn.matmul(q, ttnn.transpose(k, -1, -2))  # shape [B, heads, seq, seq]
        attn_scores = ttnn.multiply(attn_scores, 1.0 / (head_dim**0.5))
        attn_probs = ttnn.softmax(attn_scores, dim=-1)
        context = ttnn.matmul(attn_probs, v)  # [B, heads, seq, head_dim]

        # merge heads back to [B, seq, hidden]
        context = ttnn.transpose(context, 1, 2)  # B, seq, heads, head_dim
        context = ttnn.reshape(context, (-1, seq_len, cfg.hidden_size))
        attn_out = ttnn.linear(context, self.out_w, bias=self.out_b, dtype=cfg.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Residual 1
        x = ttnn.add(x, attn_out)

        # LayerNorm + MLP
        ln2 = ttnn.layer_norm(x, weight=self.ln2_w, bias=self.ln2_b, epsilon=1e-5)
        ff1 = ttnn.linear(ln2, self.ff1_w, bias=self.ff1_b, dtype=cfg.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
        ff1 = ttnn.gelu(ff1)
        ff2 = ttnn.linear(ff1, self.ff2_w, bias=self.ff2_b, dtype=cfg.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Residual 2
        y = ttnn.add(x, ff2)
        return y
