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
