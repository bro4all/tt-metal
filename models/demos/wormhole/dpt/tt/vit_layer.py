"""
Minimal ViT encoder layer for DPT on TTNN.
Currently a skeleton: wire fused QKV + softmax + MLP using TTNN building blocks.
"""
from dataclasses import dataclass
import ttnn  # type: ignore


@dataclass
class ViTLayerConfig:
    hidden_size: int = 1024
    num_heads: int = 16
    seq_len: int = (384 // 16) * (384 // 16)  # 576
    dtype = ttnn.bfloat16
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
        # LayerNorm before attention
        ln1 = ttnn.layer_norm(x, self.ln1_w, self.ln1_b, epsilon=1e-5, dtype=cfg.dtype)

        # Fused QKV linear
        qkv = ttnn.linear(ln1, self.qkv_w, bias=self.qkv_b, dtype=cfg.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)

        # reshape QKV: [B, seq, 3, heads, head_dim]
        head_dim = cfg.hidden_size // cfg.num_heads
        q, k, v = ttnn.split(qkv, cfg.hidden_size, dim=-1)
        q = ttnn.reshape(q, (q.shape[0], cfg.seq_len, cfg.num_heads, head_dim))
        k = ttnn.reshape(k, (k.shape[0], cfg.seq_len, cfg.num_heads, head_dim))
        v = ttnn.reshape(v, (v.shape[0], cfg.seq_len, cfg.num_heads, head_dim))

        # scaled dot-product attention
        attn_scores = ttnn.matmul(q, ttnn.transpose(k, -1, -2))  # shape [B, seq, heads, seq]
        attn_scores = ttnn.scale(attn_scores, 1.0 / (head_dim ** 0.5))
        attn_probs = ttnn.softmax(attn_scores, dim=-1)
        context = ttnn.matmul(attn_probs, v)  # [B, seq, heads, head_dim]

        # merge heads
        context = ttnn.reshape(context, (-1, cfg.seq_len, cfg.hidden_size))
        attn_out = ttnn.linear(context, self.out_w, bias=self.out_b, dtype=cfg.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Residual 1
        x = ttnn.add(x, attn_out)

        # LayerNorm + MLP
        ln2 = ttnn.layer_norm(x, self.ln2_w, self.ln2_b, epsilon=1e-5, dtype=cfg.dtype)
        ff1 = ttnn.linear(ln2, self.ff1_w, bias=self.ff1_b, dtype=cfg.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
        ff1 = ttnn.gelu(ff1)
        ff2 = ttnn.linear(ff1, self.ff2_w, bias=self.ff2_b, dtype=cfg.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Residual 2
        y = ttnn.add(x, ff2)
        return y
