import numpy as np
from pathlib import Path
import pytest


p = Path("models/demos/wormhole/dpt/weights/fused_qkv_state_dict.npz")
if not p.exists():
    pytest.skip("Weights not generated; run convert_weights.py first", allow_module_level=True)


def test_qkv_shape():
    npz = np.load(p)
    qkv = npz["dpt.encoder.layer.0.attention.attention.qkv.weight"]
    assert qkv.shape == (3072, 1024)
    assert np.isfinite(qkv).all()


def test_neck_and_head_keys_exist():
    npz = np.load(p)
    expected = {
        "neck.reassemble_stage.layers.0.projection.weight": (256, 1024, 1, 1),
        "neck.reassemble_stage.layers.1.projection.weight": (512, 1024, 1, 1),
        "neck.reassemble_stage.layers.2.projection.weight": (1024, 1024, 1, 1),
        "neck.reassemble_stage.layers.3.projection.weight": (1024, 1024, 1, 1),
        "neck.convs.0.weight": (256, 256, 3, 3),
        "neck.fusion_stage.layers.0.residual_layer1.convolution1.weight": (256, 256, 3, 3),
        "head.head.0.weight": (128, 256, 3, 3),
        "head.head.4.weight": (1, 32, 1, 1),
    }
    for k, shape in expected.items():
        assert k in npz.files
        assert npz[k].shape == shape
