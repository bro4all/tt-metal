"""
Fusion/refinement blocks for DPT neck + depth head.
"""
from dataclasses import dataclass
from typing import Any, Sequence
from typing import Sequence
import torch
import ttnn  # type: ignore


@dataclass
class FusionBlockConfig:
    hidden_size: int = 256
    dtype: Any = ttnn.bfloat16


class ResidualConvUnit:
    def __init__(self, cfg: FusionBlockConfig, w1, b1, w2, b2):
        self.cfg = cfg
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        device = x.device()
        if not isinstance(self.w1, ttnn.Tensor):
            self.w1 = ttnn.from_torch(
                torch.from_numpy(self.w1).contiguous(),
                dtype=self.cfg.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        if not isinstance(self.w2, ttnn.Tensor):
            self.w2 = ttnn.from_torch(
                torch.from_numpy(self.w2).contiguous(),
                dtype=self.cfg.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        if self.b1 is not None and not isinstance(self.b1, ttnn.Tensor):
            self.b1 = ttnn.from_torch(
                torch.from_numpy(self.b1).contiguous(),
                dtype=self.cfg.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        if self.b2 is not None and not isinstance(self.b2, ttnn.Tensor):
            self.b2 = ttnn.from_torch(
                torch.from_numpy(self.b2).contiguous(),
                dtype=self.cfg.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        # relu expects TILE layout; conv prefers ROW_MAJOR. Hop to TILE for relu, then back.
        x_rm = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)
        x_tile = ttnn.to_layout(x_rm, layout=ttnn.TILE_LAYOUT, dtype=self.cfg.dtype)
        y = ttnn.relu(x_tile)
        y = ttnn.to_layout(y, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)
        y = ttnn.conv2d(
            input_tensor=y,
            weight_tensor=self.w1,
            bias_tensor=self.b1,
            device=device,
            in_channels=self.w1.shape[1],
            out_channels=self.w1.shape[0],
            batch_size=y.shape[0],
            input_height=y.shape[1],
            input_width=y.shape[2],
            kernel_size=(self.w1.shape[2], self.w1.shape[3]),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            dtype=self.cfg.dtype,
        )
        y_tile = ttnn.to_layout(y, layout=ttnn.TILE_LAYOUT, dtype=self.cfg.dtype)
        y = ttnn.relu(y_tile)
        y = ttnn.to_layout(y, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)
        y = ttnn.conv2d(
            input_tensor=y,
            weight_tensor=self.w2,
            bias_tensor=self.b2,
            device=device,
            in_channels=self.w2.shape[1],
            out_channels=self.w2.shape[0],
            batch_size=y.shape[0],
            input_height=y.shape[1],
            input_width=y.shape[2],
            kernel_size=(self.w2.shape[2], self.w2.shape[3]),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            dtype=self.cfg.dtype,
        )
        return ttnn.add(y, x_rm)


class FeatureFusionBlock:
    """
    Two RCUs + optional residual skip + upsample + 1x1 projection.
    Mirrors DPTFeatureFusionLayer.
    """

    def __init__(self, cfg: FusionBlockConfig, proj_w, proj_b, r1_w1, r1_b1, r1_w2, r1_b2, r2_w1, r2_b1, r2_w2, r2_b2):
        self.cfg = cfg
        self.proj_w = proj_w
        self.proj_b = proj_b
        self.rcu1 = ResidualConvUnit(cfg, r1_w1, r1_b1, r1_w2, r1_b2)
        self.rcu2 = ResidualConvUnit(cfg, r2_w1, r2_b1, r2_w2, r2_b2)

    def __call__(self, x: ttnn.Tensor, residual: ttnn.Tensor | None) -> ttnn.Tensor:
        device = x.device()
        if not isinstance(self.proj_w, ttnn.Tensor):
            self.proj_w = ttnn.from_torch(
                torch.from_numpy(self.proj_w).contiguous(),
                dtype=self.cfg.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        if self.proj_b is not None and not isinstance(self.proj_b, ttnn.Tensor):
            self.proj_b = ttnn.from_torch(
                torch.from_numpy(self.proj_b).contiguous(),
                dtype=self.cfg.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        # Keep fusion activations in ROW_MAJOR; RCUs handle relu layout hops internally.
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)
        if residual is not None:
            # align spatial shapes if needed
            if hasattr(residual, "shape") and hasattr(x, "shape"):
                rh, rw = residual.shape[1], residual.shape[2]
                xh, xw = x.shape[1], x.shape[2]
                if (rh != xh) or (rw != xw):
                    residual = ttnn.upsample(residual, scale_factor=(xh / rh, xw / rw))
            residual = ttnn.to_layout(residual, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)
            x = ttnn.add(x, self.rcu1(residual))

        x = self.rcu2(x)
        x = ttnn.upsample(x, scale_factor=(2.0, 2.0))
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.proj_w,
            bias_tensor=self.proj_b,
            device=device,
            in_channels=self.proj_w.shape[1],
            out_channels=self.proj_w.shape[0],
            batch_size=x.shape[0],
            input_height=x.shape[1],
            input_width=x.shape[2],
            kernel_size=(self.proj_w.shape[2], self.proj_w.shape[3]),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            dtype=self.cfg.dtype,
        )
        return x


class FeatureFusionStage:
    """
    Top-down fusion over four feature maps (reverse order), returning the fused pyramid.
    """

    def __init__(self, cfg: FusionBlockConfig, weights) -> None:
        self.blocks: list[FeatureFusionBlock] = []
        for i in range(4):
            p_w = weights[f"neck.fusion_stage.layers.{i}.projection.weight"]
            p_b = weights[f"neck.fusion_stage.layers.{i}.projection.bias"]

            r1_w1 = weights[f"neck.fusion_stage.layers.{i}.residual_layer1.convolution1.weight"]
            r1_b1 = weights[f"neck.fusion_stage.layers.{i}.residual_layer1.convolution1.bias"]
            r1_w2 = weights[f"neck.fusion_stage.layers.{i}.residual_layer1.convolution2.weight"]
            r1_b2 = weights[f"neck.fusion_stage.layers.{i}.residual_layer1.convolution2.bias"]

            r2_w1 = weights[f"neck.fusion_stage.layers.{i}.residual_layer2.convolution1.weight"]
            r2_b1 = weights[f"neck.fusion_stage.layers.{i}.residual_layer2.convolution1.bias"]
            r2_w2 = weights[f"neck.fusion_stage.layers.{i}.residual_layer2.convolution2.weight"]
            r2_b2 = weights[f"neck.fusion_stage.layers.{i}.residual_layer2.convolution2.bias"]

            self.blocks.append(FeatureFusionBlock(cfg, p_w, p_b, r1_w1, r1_b1, r1_w2, r1_b2, r2_w1, r2_b1, r2_w2, r2_b2))

    def __call__(self, features: Sequence[ttnn.Tensor]) -> list[ttnn.Tensor]:
        fused_outputs = []
        fused = None
        for feat, block in zip(features[::-1], self.blocks):
            fused = block(feat if fused is None else fused, residual=None if fused is None else feat)
            fused_outputs.append(fused)
        return fused_outputs


@dataclass
class FusionHeadConfig:
    features: int = 256
    dtype: Any = ttnn.bfloat16
    add_projection: bool = False
    head_in_index: int = -1


class FusionHead:
    """
    Depth estimation head (three convs) matching DPTDepthEstimationHead.
    """

    def __init__(
        self,
        cfg: FusionHeadConfig,
        conv1_w,
        conv1_b,
        conv2_w,
        conv2_b,
        conv3_w,
        conv3_b,
        projection_w=None,
        projection_b=None,
    ):
        self.cfg = cfg
        self.conv1_w = conv1_w
        self.conv1_b = conv1_b
        self.conv2_w = conv2_w
        self.conv2_b = conv2_b
        self.conv3_w = conv3_w
        self.conv3_b = conv3_b
        self.projection_w = projection_w
        self.projection_b = projection_b

    def __call__(self, fused_pyramid: Sequence[ttnn.Tensor]) -> ttnn.Tensor:
        x = fused_pyramid[self.cfg.head_in_index]
        device = x.device()

        def to_tt(t):
            if t is None or isinstance(t, ttnn.Tensor):
                return t
            return ttnn.from_torch(
                torch.from_numpy(t).contiguous(),
                dtype=self.cfg.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )

        self.projection_w = to_tt(self.projection_w)
        self.projection_b = to_tt(self.projection_b)
        self.conv1_w = to_tt(self.conv1_w)
        self.conv1_b = to_tt(self.conv1_b)
        self.conv2_w = to_tt(self.conv2_w)
        self.conv2_b = to_tt(self.conv2_b)
        self.conv3_w = to_tt(self.conv3_w)
        self.conv3_b = to_tt(self.conv3_b)

        if self.projection_w is not None:
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)
            x = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.projection_w,
                bias_tensor=self.projection_b,
                device=device,
                in_channels=self.projection_w.shape[1],
                out_channels=self.projection_w.shape[0],
                batch_size=x.shape[0],
                input_height=x.shape[1],
                input_width=x.shape[2],
                kernel_size=(self.projection_w.shape[2], self.projection_w.shape[3]),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                dtype=self.cfg.dtype,
            )
            x_tile = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT, dtype=self.cfg.dtype)
            x = ttnn.relu(x_tile)
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)

        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)

        y = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_w,
            bias_tensor=self.conv1_b,
            device=device,
            in_channels=self.conv1_w.shape[1],
            out_channels=self.conv1_w.shape[0],
            batch_size=x.shape[0],
            input_height=x.shape[1],
            input_width=x.shape[2],
            kernel_size=(self.conv1_w.shape[2], self.conv1_w.shape[3]),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            dtype=self.cfg.dtype,
        )
        y = ttnn.upsample(y, scale_factor=(2.0, 2.0))
        y = ttnn.conv2d(
            input_tensor=y,
            weight_tensor=self.conv2_w,
            bias_tensor=self.conv2_b,
            device=device,
            in_channels=self.conv2_w.shape[1],
            out_channels=self.conv2_w.shape[0],
            batch_size=y.shape[0],
            input_height=y.shape[1],
            input_width=y.shape[2],
            kernel_size=(self.conv2_w.shape[2], self.conv2_w.shape[3]),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            dtype=self.cfg.dtype,
        )
        y_tile = ttnn.to_layout(y, layout=ttnn.TILE_LAYOUT, dtype=self.cfg.dtype)
        y = ttnn.relu(y_tile)
        y = ttnn.to_layout(y, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)
        y = ttnn.conv2d(
            input_tensor=y,
            weight_tensor=self.conv3_w,
            bias_tensor=self.conv3_b,
            device=device,
            in_channels=self.conv3_w.shape[1],
            out_channels=self.conv3_w.shape[0],
            batch_size=y.shape[0],
            input_height=y.shape[1],
            input_width=y.shape[2],
            kernel_size=(self.conv3_w.shape[2], self.conv3_w.shape[3]),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            dtype=self.cfg.dtype,
        )
        y_tile = ttnn.to_layout(y, layout=ttnn.TILE_LAYOUT, dtype=self.cfg.dtype)
        y = ttnn.relu(y_tile)  # non-negative depth
        y = ttnn.to_layout(y, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)
        return y
