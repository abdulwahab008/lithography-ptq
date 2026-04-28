# Lithography PTQ (PixelILT)

Post-training quantization (**MinMax** and **PTQ4ViT / Hessian**) for lithography image-to-image models built on **ConViT** and **PVT** backbones, adapted from the [PTQ4ViT](https://arxiv.org/abs/2111.12293) codebase.

This repository is a **clean, GitHub-sized** layout: code + small sample images. **Model checkpoints (`.pth`) are not included** — add them under `checkpoints/` (see `checkpoints/README.md`).

Your original experiments under `PTQ4ViT/` and `Quantum Abdul*` on disk are **unchanged**; this folder is the copy meant for publishing.

## Full Artifacts (Google Drive)

Full PTQ artifacts (all checkpoints, full experiment outputs, and supporting files) are stored in Google Drive:

- https://drive.google.com/drive/folders/1mN__31JBIWaAt1s9D7qNTqRfUZDkE4kC?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

Use this repo for code and reproducible scripts, and use the Drive folder for large binary assets.

## Layout

| Path | Purpose |
|------|---------|
| `configs/`, `quant_layers/`, `utils/` | PTQ4ViT framework (quant modules, Hessian calibration) |
| `convit/` | ConViT lithography translator: MinMax PTQ, PTQ4ViT, inference, visual compare, EPE/L2 metrics |
| `pvt/` | PVT variant: same quantization patterns |
| `data/sample_targets/` | Demo target PNGs |
| `data/sample_pixelilt/` | Demo FP32 PixelILT outputs (reference for grids / metrics) |
| `checkpoints/` | **You** place `best_convit_translator.pth` and/or `best_pvt_translator.pth` here |
| `results/` | Generated grids, charts, and batch outputs (gitignored PNGs) |

## Setup

```bash
cd lithography-ptq
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Copy your trained .pth files into checkpoints/ (see checkpoints/README.md)
```

## ConViT (from repo root `lithography-ptq/`)

```bash
# FP32 PixelILT on sample targets
python convit/inference.py

# MinMax PTQ then save quantized weights next to convit/ + results under results/convit_ptq/
python convit/apply_ptq.py --bit 8

# PTQ4ViT (Hessian) — slower
python convit/apply_ptq4vit.py --bit 8

# Visual grid: PixelILT reference | FP32 | 8 / 6 / 4 bits
python convit/compare_bits_visual.py --mode minmax
python convit/compare_bits_visual.py --mode ptq4vit

python convit/stitch_compare.py

# EPE + L2 vs FP32 PixelILT (heavy: runs all bit widths × both schemes)
python convit/compute_metrics.py

# Verify a MinMax PTQ checkpoint
python convit/verify_ptq.py
```

## PVT

```bash
python pvt/apply_ptq_pvt.py --bit 8
python pvt/apply_ptq4vit_pvt.py --bit 8
python pvt/verify_ptq_pvt.py
```

## Tests

```bash
python3 tests/test_repo_layout.py
# or
python3 -m unittest tests.test_repo_layout
```

## License

Research / educational use. Include attribution to the original PTQ4ViT paper if you publish work based on this repo.
