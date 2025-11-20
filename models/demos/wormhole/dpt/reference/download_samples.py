#!/usr/bin/env python3
"""
Download a few small RGB images for quick baseline/TTNN smoke tests.
Sources are public COCO sample images to avoid license issues.
"""
import argparse
from pathlib import Path
import urllib.request

SAMPLES = {
    "coco_dog": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "coco_zebra": "http://images.cocodataset.org/val2017/000000397133.jpg",
}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"Downloaded {dest.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("models/demos/wormhole/dpt/assets"))
    args = ap.parse_args()

    for name, url in SAMPLES.items():
        fetch(url, args.outdir / f"{name}.jpg")


if __name__ == "__main__":
    main()
