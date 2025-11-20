import numpy as np
import pytest
from pathlib import Path

# Heavy end-to-end check; marked xfail until TTNN kernels are validated on hardware.


def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


@pytest.mark.xfail(reason="TTNN path not yet validated; requires Tenstorrent device")
def test_dpt_depth_matches_reference():
    weights_dir = Path("models/demos/wormhole/dpt/weights")
    ref_dir = Path("models/demos/wormhole/dpt/out_reference")
    assets_dir = Path("models/demos/wormhole/dpt/assets")

    if not weights_dir.exists() or not (weights_dir / "fused_qkv_state_dict.npz").exists():
        pytest.skip("Converted weights missing")
    ref_files = sorted(ref_dir.glob("*.npz"))
    if not ref_files:
        pytest.skip("Reference depth outputs missing; run run_pytorch_baseline.py first")
    img_files = sorted([p for p in assets_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not img_files:
        pytest.skip("Sample assets missing")

    # Defer heavy imports until we know we can run
    from PIL import Image
    import ttnn  # type: ignore
    from models.demos.wormhole.dpt.tt.dpt_ttnn import DPTTTNN, DPTConfig

    img_path = img_files[0]
    ref_path = ref_dir / f"{img_path.stem}.npz"
    if not ref_path.exists():
        pytest.skip(f"No reference depth for {img_path.name}")

    image = Image.open(img_path).resize((384, 384))
    arr = (np.array(image).astype(np.float32) / 127.5) - 1.0  # normalize to [-1,1]
    batch = arr[None, ...]  # NHWC

    model = DPTTTNN(DPTConfig(), weights_dir)
    tt_in = ttnn.from_torch(batch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    pred = model(tt_in)
    depth = ttnn.to_torch(pred).cpu().numpy().squeeze()

    ref_depth = np.load(ref_path)["depth"]
    assert depth.shape == ref_depth.shape
    score = _pcc(depth, ref_depth)
    assert score > 0.99
