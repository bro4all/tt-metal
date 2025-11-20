"""
Fusion + refinement head for DPT (simplified skeleton).
"""
from dataclasses import dataclass
import ttnn  # type: ignore


@dataclass
class FusionHeadConfig:
    features: int = 256
    dtype = ttnn.bfloat16


class FusionHead:
    def __init__(self, cfg: FusionHeadConfig, conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b):
        self.cfg = cfg
        self.conv1_w = conv1_w
        self.conv1_b = conv1_b
        self.conv2_w = conv2_w
        self.conv2_b = conv2_b
        self.conv3_w = conv3_w
        self.conv3_b = conv3_b

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        y = ttnn.conv2d(x, self.conv1_w, bias=self.conv1_b, dtype=self.cfg.dtype)
        y = ttnn.relu(y)
        y = ttnn.upsample(y, scale_factor=(2.0, 2.0))
        y = ttnn.conv2d(y, self.conv2_w, bias=self.conv2_b, dtype=self.cfg.dtype)
        y = ttnn.relu(y)
        y = ttnn.conv2d(y, self.conv3_w, bias=self.conv3_b, dtype=self.cfg.dtype)
        if not self.cfg.dtype == ttnn.bfloat16:
            y = ttnn.to_dtype(y, dtype=ttnn.bfloat16)
        return y
