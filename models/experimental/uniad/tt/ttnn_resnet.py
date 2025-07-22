import ttnn

from models.experimental.uniad.tt.common import TtnnConv2D


class TtBottleneck:
    expansion = 4

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        is_downsample=False,
        blk_sharded=False,
        activation_dtype=ttnn.bfloat16,
        conv3_blk_sharded=False,
        planes=None,
        stride=1,
        dilation=1,
        style="pytorch",
        conv_cfg=None,
        dcn=None,
    ):
        assert style in ["pytorch", "caffe"]
        assert dcn is None or isinstance(dcn, dict)

        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.conv_cfg = conv_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.activation_dtype = activation_dtype
        self.is_downsample = is_downsample

        if self.style == "pytorch":
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.conv1 = TtnnConv2D(conv_args.conv1, conv_pth.conv1, device=device, activation="relu")
        # self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop("fallback_on_stride", False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = TtnnConv2D(conv_args.conv2, conv_pth.conv2, device=device, activation="relu", act_block_h=32)
        else:
            assert self.conv_cfg is None, "conv_cfg must be None for DCN"
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )

        self.conv3 = TtnnConv2D(conv_args.conv3, conv_pth.conv3, device=device, activation="", is_blk=conv3_blk_sharded)

        if is_downsample:
            self.downsample = TtnnConv2D(
                conv_args.downsample[0],
                conv_pth.downsample,
                device=device,
                activation="",
                is_blk=blk_sharded,
                activation_dtype=activation_dtype,
            )

    def __call__(self, x_identity):
        x, _, _ = self.conv1(x_identity)
        if self.activation_dtype == ttnn.bfloat8_b:
            x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
            x_identity = ttnn.add(x_identity, 0.0, dtype=ttnn.bfloat8_b)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x, _, _ = self.conv2(x)
        x, _, _ = self.conv3(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        if self.is_downsample:
            x_identity, _, _ = self.downsample(x_identity)
        x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG)

        x = ttnn.add(x, x_identity)
        x = ttnn.relu(x)

        ttnn.deallocate(x_identity)
        return x
