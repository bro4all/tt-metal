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
        # TODO: implement QKV fused linear, attention, MLP, residuals, norms.
        raise NotImplementedError
