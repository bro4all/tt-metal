# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from models.experimental.uniad.reference.fpn import FPN
from models.experimental.uniad.reference.resnet import ResNet


class Detection(nn.Module):
    def __init__(
        self,
    ):
        super(Detection, self).__init__()
        self.img_backbone = ResNet(
            depth=101,
            in_channels=3,
            stem_channels=None,
            base_channels=64,
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            out_indices=(1, 2, 3),
            style="caffe",
            deep_stem=False,
            avg_down=False,
            frozen_stages=4,
            conv_cfg=None,
            norm_cfg={"type": "BN2d", "requires_grad": False},
            norm_eval=True,
            dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
            stage_with_dcn=(False, False, True, True),
            plugins=None,
            with_cp=False,
            zero_init_residual=True,
            pretrained=None,
            init_cfg=None,
        )
        self.img_neck = FPN(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs="on_output",
            num_outs=4,
            relu_before_extra_convs=True,
        )

    def forward(self, x):
        x = self.img_backbone(x)
        x = self.img_neck(x)
        return x
