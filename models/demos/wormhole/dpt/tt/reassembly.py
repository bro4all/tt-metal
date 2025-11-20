"""
DPT neck reassembly and fusion building blocks.
Mirror the HuggingFace DPTReassembleStage + conv stem.
"""
from dataclasses import dataclass
from typing import Any, Sequence
from typing import Sequence
import torch
import ttnn  # type: ignore


@dataclass
class ReassemblyConfig:
    hidden_size: int
    neck_hidden_sizes: Sequence[int]
    reassemble_factors: Sequence[float]
    fusion_hidden_size: int = 256
    dtype: Any = ttnn.bfloat16


class ReassembleLayer:
    """
    Projection + resize step from 1D ViT tokens to 2D feature map.
    """

    def __init__(self, out_channels, factor, proj_w, proj_b, resize_w=None, resize_b=None, dtype=ttnn.bfloat16):
        self.out_channels = out_channels
        self.factor = factor
        self.proj_w = proj_w
        self.proj_b = proj_b
        self.resize_w = resize_w
        self.resize_b = resize_b
        self.dtype = dtype

    def __call__(self, x_2d: ttnn.Tensor) -> ttnn.Tensor:
        device = x_2d.device()

        def to_tt(tensor):
            if tensor is None or isinstance(tensor, ttnn.Tensor):
                return tensor
            return ttnn.from_torch(
                torch.from_numpy(tensor).contiguous(), dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )

        # lazily materialize weights on device
        self.proj_w = to_tt(self.proj_w)
        self.proj_b = to_tt(self.proj_b)
        self.resize_w = to_tt(self.resize_w)
        self.resize_b = to_tt(self.resize_b)
        # projection 1x1
        B, H, W, _ = x_2d.shape
        kH, kW = self.proj_w.shape[2], self.proj_w.shape[3]
        y = ttnn.conv2d(
            input_tensor=x_2d,
            weight_tensor=self.proj_w,
            bias_tensor=self.proj_b,
            in_channels=self.proj_w.shape[1],
            out_channels=self.proj_w.shape[0],
            batch_size=B,
            input_height=H,
            input_width=W,
            kernel_size=(kH, kW),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            dtype=self.dtype,
        )

        # learned resize (conv / deconv) if available, otherwise fallback to scale factor
        if self.resize_w is not None:
            kH, kW = self.resize_w.shape[2], self.resize_w.shape[3]
            if self.factor > 1:
                # use conv_transpose for upsample
                res = ttnn.conv_transpose2d(
                    input_tensor=y,
                    weight_tensor=self.resize_w,
                    bias_tensor=self.resize_b,
                    in_channels=self.resize_w.shape[0],
                    out_channels=self.resize_w.shape[1],
                    batch_size=y.shape[0],
                    input_height=y.shape[1],
                    input_width=y.shape[2],
                    kernel_size=(kH, kW),
                    stride=(int(self.factor), int(self.factor)),
                    padding=(0, 0),
                    output_padding=(0, 0),
                    dilation=(1, 1),
                    groups=1,
                    mirror_kernel=True,
                    return_output_dim=True,
                    dtype=self.dtype,
                )
                y = res[0]
            else:
                stride = int(1 / self.factor)
                y = ttnn.conv2d(
                    input_tensor=y,
                    weight_tensor=self.resize_w,
                    bias_tensor=self.resize_b,
                    in_channels=self.resize_w.shape[1],
                    out_channels=self.resize_w.shape[0],
                    batch_size=y.shape[0],
                    input_height=y.shape[1],
                    input_width=y.shape[2],
                    kernel_size=(kH, kW),
                    stride=(stride, stride),
                    padding=(1, 1),
                    dilation=(1, 1),
                    groups=1,
                    dtype=self.dtype,
                )

        if self.factor != 1 and self.resize_w is None:
            # ttnn.upsample handles both >1 (upsample) and <1 (downsample when factor<1 treated as stride>1)
            scale = (float(self.factor), float(self.factor))
            y = ttnn.upsample(y, scale_factor=scale)
        return y


class ReassemblyStage:
    """
    Full DPT reassembly: readout projection, spatial reshape, projection/resize, then 3x3 conv to fusion size.
    """

    def __init__(self, cfg: ReassemblyConfig, weights):
        self.cfg = cfg
        self.readout_w = []
        self.readout_b = []
        self.layers = []
        self.convs = []

        # build per-stage modules
        for i, (chan, factor) in enumerate(zip(cfg.neck_hidden_sizes, cfg.reassemble_factors)):
            # readout project linear weights (when present)
            rw_key = f"neck.reassemble_stage.readout_projects.{i}.0.weight"
            rb_key = f"neck.reassemble_stage.readout_projects.{i}.0.bias"
            self.readout_w.append(weights.get(rw_key))
            self.readout_b.append(weights.get(rb_key))

            # projection + resize weights
            proj_w = weights[f"neck.reassemble_stage.layers.{i}.projection.weight"]
            proj_b = weights[f"neck.reassemble_stage.layers.{i}.projection.bias"]
            resize_w = weights.get(f"neck.reassemble_stage.layers.{i}.resize.weight")
            resize_b = weights.get(f"neck.reassemble_stage.layers.{i}.resize.bias")
            self.layers.append(ReassembleLayer(chan, factor, proj_w, proj_b, resize_w, resize_b, dtype=cfg.dtype))

            # fusion stem conv (3x3 -> fusion_hidden_size)
            conv_w = weights[f"neck.convs.{i}.weight"]
            conv_b = weights.get(f"neck.convs.{i}.bias")  # HF convs are bias=False, keep optional
            self.convs.append((conv_w, conv_b))

    def _apply_readout(self, tokens: ttnn.Tensor, stage: int) -> ttnn.Tensor:
        """
        tokens: (B, seq, hidden)
        Applies readout project (concatenate cls with patches) when weights exist.
        """
        device = tokens.device()
        if self.readout_w[stage] is not None and not isinstance(self.readout_w[stage], ttnn.Tensor):
            self.readout_w[stage] = ttnn.from_torch(
                torch.from_numpy(self.readout_w[stage]).contiguous(),
                dtype=self.cfg.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        if self.readout_b[stage] is not None and not isinstance(self.readout_b[stage], ttnn.Tensor):
            self.readout_b[stage] = ttnn.from_torch(
                torch.from_numpy(self.readout_b[stage]).contiguous(),
                dtype=self.cfg.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        # split cls and patches
        cls_token = ttnn.slice(tokens, (0, 0, 0), (tokens.shape[0], 1, tokens.shape[2]))
        patches = ttnn.slice(tokens, (0, 1, 0), (tokens.shape[0], tokens.shape[1], tokens.shape[2]))
        if self.readout_w[stage] is None:
            return patches

        cls_expanded = ttnn.repeat(cls_token, [1, patches.shape[1], 1])
        concat = ttnn.concat([patches, cls_expanded], dim=-1)
        projected = ttnn.linear(
            concat, self.readout_w[stage], bias=self.readout_b[stage], dtype=self.cfg.dtype
        )
        projected = ttnn.gelu(projected)
        return projected

    def __call__(self, tapped_tokens: Sequence[ttnn.Tensor], patch_hw: int) -> list[ttnn.Tensor]:
        """
        Args:
            tapped_tokens: list of ViT hidden states with CLS (B, seq+1, hidden)
            patch_hw: patch grid size (image_size // patch_size)
        Returns: list of feature maps ready for fusion, each shaped ~ (B, H, W, fusion_hidden_size)
        """
        outputs = []
        h = w = patch_hw
        for i, tokens in enumerate(tapped_tokens):
            patches = self._apply_readout(tokens, i)  # (B, seq, hidden)
            # reshape sequence to spatial grid
            feat = ttnn.reshape(patches, (patches.shape[0], h, w, self.cfg.hidden_size))
            # project + resize
            feat = self.layers[i](feat)
            # fuse conv to common hidden size
            conv_w, conv_b = self.convs[i]
            device = feat.device()
            if not isinstance(conv_w, ttnn.Tensor):
                conv_w = ttnn.from_torch(
                    torch.from_numpy(conv_w).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                )
                self.convs[i] = (conv_w, conv_b)
            if conv_b is not None and not isinstance(conv_b, ttnn.Tensor):
                conv_b = ttnn.from_torch(
                    torch.from_numpy(conv_b).contiguous(),
                    dtype=self.cfg.dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                )
                self.convs[i] = (conv_w, conv_b)
            kH, kW = conv_w.shape[2], conv_w.shape[3]
            fused = ttnn.conv2d(
                input_tensor=feat,
                weight_tensor=conv_w,
                bias_tensor=conv_b,
                in_channels=conv_w.shape[1],
                out_channels=conv_w.shape[0],
                batch_size=feat.shape[0],
                input_height=feat.shape[1],
                input_width=feat.shape[2],
                kernel_size=(kH, kW),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                dtype=self.cfg.dtype,
            )
            outputs.append(fused)
        return outputs
