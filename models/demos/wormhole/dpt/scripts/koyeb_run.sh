#!/bin/bash
# Convenience launcher for Koyeb N300 runs.
# Assumes repo is cloned at /root/tt-metal.
set -ex

# Install CPU dependencies needed for weight conversion and reference runs.
python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.26.4 torch==2.2.2+cpu torchvision==0.17.2+cpu transformers==4.38.2 timm==0.9.16 pillow==10.3.0 huggingface_hub==0.23.4

# Ensure demo assets are present (use branch-hosted copies to avoid coco downloads).
python3 - <<'PY'
import pathlib, urllib.request
base = pathlib.Path("models/demos/wormhole/dpt/assets")
base.mkdir(parents=True, exist_ok=True)
urls = {
    "dog.jpg": "https://raw.githubusercontent.com/bro4all/tt-metal/feature/dpt-large-ttnn/models/demos/wormhole/dpt/assets/dog.jpg",
    "coco_dog.jpg": "https://raw.githubusercontent.com/bro4all/tt-metal/feature/dpt-large-ttnn/models/demos/wormhole/dpt/assets/coco_dog.jpg",
    "coco_zebra.jpg": "https://raw.githubusercontent.com/bro4all/tt-metal/feature/dpt-large-ttnn/models/demos/wormhole/dpt/assets/coco_zebra.jpg",
}
for name, url in urls.items():
    out = base / name
    if not out.exists():
        urllib.request.urlretrieve(url, out)
        print(f"downloaded {name}")
PY

# Cache locations to avoid HOME pollution on the instance.
export PYTHONPATH=${PYTHONPATH:-/root/tt-metal}
export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/tmp/hf_cache}

# Convert weights, run deterministic PyTorch baseline, then TTNN correctness + perf.
python3 models/demos/wormhole/dpt/scripts/convert_weights.py --model Intel/dpt-large --outdir models/demos/wormhole/dpt/weights
python3 models/demos/wormhole/dpt/reference/run_pytorch_baseline.py --images models/demos/wormhole/dpt/assets --output models/demos/wormhole/dpt/out_reference
python3 models/demos/wormhole/dpt/scripts/run_ttnn_vs_ref.py --device-id ${DEVICE_ID:-0}
python3 models/demos/wormhole/dpt/scripts/bench_dpt_ttnn.py --device-id ${DEVICE_ID:-0} --size 384 --iters 50
