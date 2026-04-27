# ConViT lithography PTQ

**Start here:** [`../README.md`](../README.md) — repo layout, `checkpoints/`, `data/sample_*`, and commands for this clean publishable tree.

The sections below are the original notes from the research folder (paths like `target/` → use `../data/sample_targets/` in this repo).

# ConViT + PTQ (inside PTQ4ViT)

## Bit-width

All apply scripts accept **`--bit 4`**, **`6`**, or **`8`** (weights and activations use the same width: W4A4, W6A6, W8A8 on supported layers). Lower bit-widths are more aggressive; compare outputs to FP32 after changing `--bit`.

## Min-max PTQ (fast)

Calibration images live in `target/` next to this README by default.

```bash
cd "CViT pretrained"
python apply_ptq.py --model best_convit_translator.pth --bit 4 --output-dir results_convit_ptq_w4
```

Produces a checkpoint such as `best_convit_translator_ptq_w4a4.pth` and PNGs under `--output-dir`.

## PTQ4ViT (Hessian-guided, slower)

```bash
python apply_ptq4vit.py --model best_convit_translator.pth --bit 4 --target-dir target --output-dir results_convit_ptq4vit_w4 --calib-num 32
```

## Verify

Match `--bit` to the checkpoint you built (re-wraps Linears with the same width):

```bash
python verify_ptq.py --fp32-model best_convit_translator.pth --ptq-model best_convit_translator_ptq_w4a4.pth --bit 4
```

Decoder `ConvTranspose2d` layers stay FP32 in these scripts.
