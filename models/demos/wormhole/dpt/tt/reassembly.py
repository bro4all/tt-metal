"""
Reassembly blocks for DPT.
Each block takes a ViT feature map, projects with 1x1 conv (linear) and upsamples.
"""
from dataclasses import dataclass
import ttnn  # type: ignore


@dataclass
class ReassemblyConfig:
    in_channels: int
    out_channels: int
    upsample: bool = True
    dtype = ttnn.bfloat16


class ReassemblyBlock:
    def __init__(self, cfg: ReassemblyConfig, proj_w, proj_b):
        self.cfg = cfg
        self.proj_w = proj_w
        self.proj_b = proj_b

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # 1x1 projection (conv)
        y = ttnn.conv2d(
            x,
            self.proj_w,
            bias=self.proj_b,
            dtype=self.cfg.dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if self.cfg.upsample:
            y = ttnn.upsample(y, scale_factor=(2.0, 2.0))
        return y
