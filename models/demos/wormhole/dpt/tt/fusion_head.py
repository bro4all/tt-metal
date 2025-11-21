"""
Fusion/refinement blocks for DPT neck + depth head.
"""
import os
from dataclasses import dataclass
from typing import Any, Sequence

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
        self.w1_prepared = None
        self.b1_prepared = None
        self.w2_prepared = None
        self.b2_prepared = None
        self.conv_config = ttnn.Conv2dConfig(
            config_tensors_in_dram=True,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG

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
        if self.w1_prepared is None or self.w1_prepared.device() != device:
            if isinstance(self.w1, ttnn.Tensor):
                w1_host = ttnn.from_torch(
                    ttnn.to_torch(self.w1).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                w1_host = ttnn.from_torch(
                    torch.from_numpy(self.w1).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            self.w1_prepared = ttnn.prepare_conv_weights(
                weight_tensor=w1_host,
                input_memory_config=self.memory_config,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=self.w1.shape[1],
                out_channels=self.w1.shape[0],
                batch_size=y.shape[0],
                input_height=y.shape[1],
                input_width=y.shape[2],
                kernel_size=(self.w1.shape[2], self.w1.shape[3]),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                has_bias=self.b1 is not None,
                groups=1,
                device=device,
                input_dtype=self.cfg.dtype,
                output_dtype=self.cfg.dtype,
                conv_config=self.conv_config,
            )
            if self.b1 is not None:
                if isinstance(self.b1, ttnn.Tensor):
                    b1_host = ttnn.from_torch(
                        ttnn.to_torch(self.b1).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    b1_host = ttnn.from_torch(
                        torch.from_numpy(self.b1).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                self.b1_prepared = ttnn.prepare_conv_bias(
                    bias_tensor=b1_host,
                    input_memory_config=self.memory_config,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW",
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
                    device=device,
                    input_dtype=self.cfg.dtype,
                    output_dtype=self.cfg.dtype,
                    conv_config=self.conv_config,
                )
        y = ttnn.conv2d(
            input_tensor=y,
            weight_tensor=self.w1_prepared,
            bias_tensor=self.b1_prepared,
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
            conv_config=self.conv_config,
        )
        y_tile = ttnn.to_layout(y, layout=ttnn.TILE_LAYOUT, dtype=self.cfg.dtype)
        y = ttnn.relu(y_tile)
        y = ttnn.to_layout(y, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)
        if self.w2_prepared is None or self.w2_prepared.device() != device:
            if isinstance(self.w2, ttnn.Tensor):
                w2_host = ttnn.from_torch(
                    ttnn.to_torch(self.w2).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                w2_host = ttnn.from_torch(
                    torch.from_numpy(self.w2).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            self.w2_prepared = ttnn.prepare_conv_weights(
                weight_tensor=w2_host,
                input_memory_config=self.memory_config,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=self.w2.shape[1],
                out_channels=self.w2.shape[0],
                batch_size=y.shape[0],
                input_height=y.shape[1],
                input_width=y.shape[2],
                kernel_size=(self.w2.shape[2], self.w2.shape[3]),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                has_bias=self.b2 is not None,
                groups=1,
                device=device,
                input_dtype=self.cfg.dtype,
                output_dtype=self.cfg.dtype,
                conv_config=self.conv_config,
            )
            if self.b2 is not None:
                if isinstance(self.b2, ttnn.Tensor):
                    b2_host = ttnn.from_torch(
                        ttnn.to_torch(self.b2).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    b2_host = ttnn.from_torch(
                        torch.from_numpy(self.b2).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                self.b2_prepared = ttnn.prepare_conv_bias(
                    bias_tensor=b2_host,
                    input_memory_config=self.memory_config,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW",
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
                    device=device,
                    input_dtype=self.cfg.dtype,
                    output_dtype=self.cfg.dtype,
                    conv_config=self.conv_config,
                )
        y = ttnn.conv2d(
            input_tensor=y,
            weight_tensor=self.w2_prepared,
            bias_tensor=self.b2_prepared,
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
            conv_config=self.conv_config,
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
        self.proj_w_prepared = None
        self.proj_b_prepared = None
        self.conv_config = ttnn.Conv2dConfig(
            config_tensors_in_dram=True,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG
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
        if self.proj_w_prepared is None or self.proj_w_prepared.device() != device:
            if isinstance(self.proj_w, ttnn.Tensor):
                pw_host = ttnn.from_torch(
                    ttnn.to_torch(self.proj_w).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                pw_host = ttnn.from_torch(
                    torch.from_numpy(self.proj_w).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            self.proj_w_prepared = ttnn.prepare_conv_weights(
                weight_tensor=pw_host,
                input_memory_config=self.memory_config,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=self.proj_w.shape[1],
                out_channels=self.proj_w.shape[0],
                batch_size=x.shape[0],
                input_height=x.shape[1],
                input_width=x.shape[2],
                kernel_size=(self.proj_w.shape[2], self.proj_w.shape[3]),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                has_bias=self.proj_b is not None,
                groups=1,
                device=device,
                input_dtype=self.cfg.dtype,
                output_dtype=self.cfg.dtype,
                conv_config=self.conv_config,
            )
            if self.proj_b is not None:
                if isinstance(self.proj_b, ttnn.Tensor):
                    pb_host = ttnn.from_torch(
                        ttnn.to_torch(self.proj_b).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    pb_host = ttnn.from_torch(
                        torch.from_numpy(self.proj_b).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                self.proj_b_prepared = ttnn.prepare_conv_bias(
                    bias_tensor=pb_host,
                    input_memory_config=self.memory_config,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW",
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
                    device=device,
                    input_dtype=self.cfg.dtype,
                    output_dtype=self.cfg.dtype,
                    conv_config=self.conv_config,
                )
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.proj_w_prepared,
            bias_tensor=self.proj_b_prepared,
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
            conv_config=self.conv_config,
        )
        return x


class FeatureFusionStage:
    """
    Top-down fusion over four feature maps (reverse order), returning the fused pyramid.
    """

    def __init__(self, cfg: FusionBlockConfig, weights) -> None:
        self.blocks: list[FeatureFusionBlock] = []
        self.use_torch_fusion = os.getenv("DPT_FORCE_TORCH_FUSION", "1") == "1"
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
        if self.use_torch_fusion:
            # Fallback fusion on host to avoid L1_SMALL conv failures.
            device0 = features[0].device()
            feats_t = [ttnn.to_torch(f).float() for f in features]  # NHWC

            def conv_nhwc(x, w_np, b_np=None, stride=1, padding=1):
                x_chw = x.permute(0, 3, 1, 2).contiguous()
                w_t = torch.from_numpy(w_np).float()
                b_t = torch.from_numpy(b_np).float() if b_np is not None else None
                y = torch.nn.functional.conv2d(x_chw, w_t, bias=b_t, stride=stride, padding=padding)
                return y.permute(0, 2, 3, 1).contiguous()

            def rcu(x, w1, b1, w2, b2):
                # Pre-activation residual block: ReLU -> conv -> ReLU -> conv -> add input
                residual = x
                x = torch.nn.functional.relu(x)
                x = conv_nhwc(x, w1, b1, padding=1)
                x = torch.nn.functional.relu(x)
                x = conv_nhwc(x, w2, b2, padding=1)
                return x + residual

            fused_outputs_t = []
            fused_t = None
            for feat_t, block in zip(feats_t[::-1], self.blocks):
                if fused_t is None:
                    x = feat_t
                else:
                    # align residual spatial to fused
                    res = feat_t
                    if res.shape[1] != fused_t.shape[1] or res.shape[2] != fused_t.shape[2]:
                        res = torch.nn.functional.interpolate(
                            res.permute(0, 3, 1, 2),
                            size=(fused_t.shape[1], fused_t.shape[2]),
                            mode="bilinear",
                            align_corners=False,
                        ).permute(0, 2, 3, 1)
                    x = fused_t + rcu(res, block.rcu1.w1, block.rcu1.b1, block.rcu1.w2, block.rcu1.b2)
                x = rcu(x, block.rcu2.w1, block.rcu2.b1, block.rcu2.w2, block.rcu2.b2)
                # upsample by factor 2
                x = torch.nn.functional.interpolate(
                    x.permute(0, 3, 1, 2), scale_factor=2.0, mode="bilinear", align_corners=True
                ).permute(0, 2, 3, 1)
                x = conv_nhwc(x, block.proj_w, block.proj_b, padding=0)
                fused_t = x
                fused_outputs_t.append(fused_t)

            # Return as torch NHWC tensors (fusion head will handle).
            return fused_outputs_t

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
        self.conv_config = ttnn.Conv2dConfig(
            config_tensors_in_dram=True,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.proj_w_prepared = None
        self.proj_b_prepared = None
        self.conv1_w_prepared = None
        self.conv1_b_prepared = None
        self.conv2_w_prepared = None
        self.conv2_b_prepared = None
        self.conv3_w_prepared = None
        self.conv3_b_prepared = None

    def __call__(self, fused_pyramid: Sequence[ttnn.Tensor]) -> ttnn.Tensor:
        # Torch fallback when fusion was run on host.
        if isinstance(fused_pyramid[0], torch.Tensor):
            x = fused_pyramid[self.cfg.head_in_index]  # NHWC torch
            # optional projection
            if self.projection_w is not None:
                x = torch.nn.functional.conv2d(
                    x.permute(0, 3, 1, 2),
                    torch.from_numpy(self.projection_w).float(),
                    bias=torch.from_numpy(self.projection_b) if self.projection_b is not None else None,
                    padding=1,
                ).permute(0, 2, 3, 1)
                x = torch.nn.functional.relu(x)

            y = torch.nn.functional.conv2d(
                x.permute(0, 3, 1, 2),
                torch.from_numpy(self.conv1_w).float(),
                bias=torch.from_numpy(self.conv1_b) if self.conv1_b is not None else None,
                padding=1,
            ).permute(0, 2, 3, 1)
            y = torch.nn.functional.interpolate(
                y.permute(0, 3, 1, 2), scale_factor=2.0, mode="bilinear", align_corners=True
            ).permute(0, 2, 3, 1)
            y = torch.nn.functional.conv2d(
                y.permute(0, 3, 1, 2),
                torch.from_numpy(self.conv2_w).float(),
                bias=torch.from_numpy(self.conv2_b) if self.conv2_b is not None else None,
                padding=1,
            ).permute(0, 2, 3, 1)
            y = torch.nn.functional.relu(y)
            y = torch.nn.functional.conv2d(
                y.permute(0, 3, 1, 2),
                torch.from_numpy(self.conv3_w).float(),
                bias=torch.from_numpy(self.conv3_b) if self.conv3_b is not None else None,
                padding=0,
            ).permute(0, 2, 3, 1)
            y = torch.nn.functional.relu(y)
            return y  # torch NHWC

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

        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)

        if self.projection_w is not None:
            if self.proj_w_prepared is None or self.proj_w_prepared.device() != device:
                if isinstance(self.projection_w, ttnn.Tensor):
                    pw_host = ttnn.from_torch(
                        ttnn.to_torch(self.projection_w).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    pw_host = ttnn.from_torch(
                        torch.from_numpy(self.projection_w).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                self.proj_w_prepared = ttnn.prepare_conv_weights(
                    weight_tensor=pw_host,
                    input_memory_config=self.memory_config,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW",
                    in_channels=self.projection_w.shape[1],
                    out_channels=self.projection_w.shape[0],
                    batch_size=x.shape[0],
                    input_height=x.shape[1],
                    input_width=x.shape[2],
                    kernel_size=(self.projection_w.shape[2], self.projection_w.shape[3]),
                    stride=(1, 1),
                    padding=(1, 1),
                    dilation=(1, 1),
                    has_bias=self.projection_b is not None,
                    groups=1,
                    device=device,
                    input_dtype=self.cfg.dtype,
                    output_dtype=self.cfg.dtype,
                    conv_config=self.conv_config,
                )
                if self.projection_b is not None:
                    if isinstance(self.projection_b, ttnn.Tensor):
                        pb_host = ttnn.from_torch(
                            ttnn.to_torch(self.projection_b).contiguous(),
                            dtype=self.cfg.dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                        )
                    else:
                        pb_host = ttnn.from_torch(
                            torch.from_numpy(self.projection_b).contiguous(),
                            dtype=self.cfg.dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                        )
                    self.proj_b_prepared = ttnn.prepare_conv_bias(
                        bias_tensor=pb_host,
                        input_memory_config=self.memory_config,
                        input_layout=ttnn.ROW_MAJOR_LAYOUT,
                        weights_format="OIHW",
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
                        device=device,
                        input_dtype=self.cfg.dtype,
                        output_dtype=self.cfg.dtype,
                        conv_config=self.conv_config,
                    )
            x = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.proj_w_prepared,
                bias_tensor=self.proj_b_prepared,
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
                conv_config=self.conv_config,
            )
            x_tile = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT, dtype=self.cfg.dtype)
            x = ttnn.relu(x_tile)
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)

        # Conv1
        if self.conv1_w_prepared is None or self.conv1_w_prepared.device() != device:
            if isinstance(self.conv1_w, ttnn.Tensor):
                c1_host = ttnn.from_torch(
                    ttnn.to_torch(self.conv1_w).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                c1_host = ttnn.from_torch(
                    torch.from_numpy(self.conv1_w).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            self.conv1_w_prepared = ttnn.prepare_conv_weights(
                weight_tensor=c1_host,
                input_memory_config=self.memory_config,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=self.conv1_w.shape[1],
                out_channels=self.conv1_w.shape[0],
                batch_size=x.shape[0],
                input_height=x.shape[1],
                input_width=x.shape[2],
                kernel_size=(self.conv1_w.shape[2], self.conv1_w.shape[3]),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                has_bias=self.conv1_b is not None,
                groups=1,
                device=device,
                input_dtype=self.cfg.dtype,
                output_dtype=self.cfg.dtype,
                conv_config=self.conv_config,
            )
            if self.conv1_b is not None:
                if isinstance(self.conv1_b, ttnn.Tensor):
                    c1b_host = ttnn.from_torch(
                        ttnn.to_torch(self.conv1_b).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    c1b_host = ttnn.from_torch(
                        torch.from_numpy(self.conv1_b).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                self.conv1_b_prepared = ttnn.prepare_conv_bias(
                    bias_tensor=c1b_host,
                    input_memory_config=self.memory_config,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW",
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
                    device=device,
                    input_dtype=self.cfg.dtype,
                    output_dtype=self.cfg.dtype,
                    conv_config=self.conv_config,
                )
        y = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_w_prepared,
            bias_tensor=self.conv1_b_prepared,
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
            conv_config=self.conv_config,
        )

        y = ttnn.upsample(y, scale_factor=(2.0, 2.0))

        # Conv2
        if self.conv2_w_prepared is None or self.conv2_w_prepared.device() != device:
            if isinstance(self.conv2_w, ttnn.Tensor):
                c2_host = ttnn.from_torch(
                    ttnn.to_torch(self.conv2_w).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                c2_host = ttnn.from_torch(
                    torch.from_numpy(self.conv2_w).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            self.conv2_w_prepared = ttnn.prepare_conv_weights(
                weight_tensor=c2_host,
                input_memory_config=self.memory_config,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=self.conv2_w.shape[1],
                out_channels=self.conv2_w.shape[0],
                batch_size=y.shape[0],
                input_height=y.shape[1],
                input_width=y.shape[2],
                kernel_size=(self.conv2_w.shape[2], self.conv2_w.shape[3]),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                has_bias=self.conv2_b is not None,
                groups=1,
                device=device,
                input_dtype=self.cfg.dtype,
                output_dtype=self.cfg.dtype,
                conv_config=self.conv_config,
            )
            if self.conv2_b is not None:
                if isinstance(self.conv2_b, ttnn.Tensor):
                    c2b_host = ttnn.from_torch(
                        ttnn.to_torch(self.conv2_b).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    c2b_host = ttnn.from_torch(
                        torch.from_numpy(self.conv2_b).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                self.conv2_b_prepared = ttnn.prepare_conv_bias(
                    bias_tensor=c2b_host,
                    input_memory_config=self.memory_config,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW",
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
                    device=device,
                    input_dtype=self.cfg.dtype,
                    output_dtype=self.cfg.dtype,
                    conv_config=self.conv_config,
                )
        y = ttnn.conv2d(
            input_tensor=y,
            weight_tensor=self.conv2_w_prepared,
            bias_tensor=self.conv2_b_prepared,
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
            conv_config=self.conv_config,
        )
        y = ttnn.relu(y)

        # Conv3
        if self.conv3_w_prepared is None or self.conv3_w_prepared.device() != device:
            if isinstance(self.conv3_w, ttnn.Tensor):
                c3_host = ttnn.from_torch(
                    ttnn.to_torch(self.conv3_w).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            else:
                c3_host = ttnn.from_torch(
                    torch.from_numpy(self.conv3_w).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            self.conv3_w_prepared = ttnn.prepare_conv_weights(
                weight_tensor=c3_host,
                input_memory_config=self.memory_config,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=self.conv3_w.shape[1],
                out_channels=self.conv3_w.shape[0],
                batch_size=y.shape[0],
                input_height=y.shape[1],
                input_width=y.shape[2],
                kernel_size=(self.conv3_w.shape[2], self.conv3_w.shape[3]),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                has_bias=self.conv3_b is not None,
                groups=1,
                device=device,
                input_dtype=self.cfg.dtype,
                output_dtype=self.cfg.dtype,
                conv_config=self.conv_config,
            )
            if self.conv3_b is not None:
                if isinstance(self.conv3_b, ttnn.Tensor):
                    c3b_host = ttnn.from_torch(
                        ttnn.to_torch(self.conv3_b).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    c3b_host = ttnn.from_torch(
                        torch.from_numpy(self.conv3_b).contiguous(),
                        dtype=self.cfg.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                self.conv3_b_prepared = ttnn.prepare_conv_bias(
                    bias_tensor=c3b_host,
                    input_memory_config=self.memory_config,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW",
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
                    device=device,
                    input_dtype=self.cfg.dtype,
                    output_dtype=self.cfg.dtype,
                    conv_config=self.conv_config,
                )
        y = ttnn.conv2d(
            input_tensor=y,
            weight_tensor=self.conv3_w_prepared,
            bias_tensor=self.conv3_b_prepared,
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
            conv_config=self.conv_config,
        )
        y = ttnn.relu(ttnn.to_layout(y, layout=ttnn.TILE_LAYOUT, dtype=self.cfg.dtype))  # non-negative depth
        y = ttnn.to_layout(y, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.cfg.dtype)
        return y
