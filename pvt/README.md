# PVT lithography PTQ

**Start here:** [`../README.md`](../README.md) — defaults use `../data/sample_targets/` and `../checkpoints/best_pvt_translator.pth`.

The sections below are the original research-folder notes.

# PVT + PTQ (inside PTQ4ViT)

## Checkpoint

`best_pvt_translator.pth` is copied from `Quantum Abdul 5/PixelILT_Project/models/` (when present in your tree).

## Bit-width

Use **`--bit 4`**, **`6`**, or **`8`** for W*A* on supported `nn.Linear` layers. **`ConvTranspose2d`** in the decoder stays **FP32**.

## Min-max PTQ

Uses calibration images from `../CViT pretrained/target` by default.

**W8A8 (default):**

```bash
cd "PVT pretrained"
python apply_ptq_pvt.py --model best_pvt_translator.pth --bit 8
```

**W4A4:**

```bash
python apply_ptq_pvt.py --model best_pvt_translator.pth --bit 4 --output-dir results_pvt_ptq_w4
```

Produces e.g. `best_pvt_translator_ptq_w4a4.pth` and PNGs under `--output-dir`.

## Verify

Use the same `--bit` as when you built the PTQ checkpoint:

```bash
python verify_ptq_pvt.py \
  --fp32-model best_pvt_translator.pth \
  --ptq-model best_pvt_translator_ptq_w4a4.pth \
  --bit 4
```

## Hessian PTQ4ViT (PVT encoder)

```bash
python apply_ptq4vit_pvt.py --model best_pvt_translator.pth --bit 4 --calib-num 32
```

Compare with min-max `apply_ptq_pvt.py` for speed vs. quality.
