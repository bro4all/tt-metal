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
        # TODO: prefetch sharding configs and tensors

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # TODO: implement QKV fused linear, attention, MLP, residuals, norms.
        raise NotImplementedError
