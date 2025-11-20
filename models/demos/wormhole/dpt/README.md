# DPT-Large (MiDaS 3.0) on TTNN

Platforms: Wormhole (N150/N300). Goal: run DPT-Large depth estimation end-to-end with TTNN, match PyTorch PCC > 0.99, and reach ≥20 FPS @384×384 (stretch 40+ FPS / 512²).

## Layout
- `reference/` – PyTorch baseline runner (HF `Intel/dpt-large`) to dump NPZ/PNGs + metrics.
- `scripts/` – weight download/convert utilities for TT-friendly layouts.
- `tt/` – TTNN implementation modules and loaders.
- `tests/` – PCC + perf tests (TTNN vs PyTorch).
- `assets/` – sample KITTI/NYU frames for quick checks (small).

## Quickstart (baseline only, CPU/GPU)
```bash
pip install torch torchvision timm transformers huggingface_hub pillow opencv-python numpy
python models/demos/wormhole/dpt/reference/download_samples.py  # grab two small COCO images
python models/demos/wormhole/dpt/reference/run_pytorch_baseline.py \
  --images assets/samples \
  --output out/reference \
  --hf-token $HF_TOKEN
```

Outputs: `(npz, png)` per image, plus `metrics.json` with PCC/MAE.

## Next steps (TTNN)
1. Implement `tt/dpt_ttnn.py` (patch embed + 24-layer ViT-L/16 + reassembly/fusion/head).
2. Add PCC tests in `tests/` against `reference` dumps.
3. Add perf tests + profiler sheet to hit FPS targets.

> This scaffold is intentionally minimal; fill in TTNN modules and weight conversion logic in subsequent commits.
