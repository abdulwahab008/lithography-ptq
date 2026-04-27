"""
Build a visual comparison grid: Target vs FP32 vs W8A8 vs W6A6 vs W4A4.

Two quantization modes:

  --mode minmax   (default, fast)
      Uses the simple per-tensor MinMax scheme from apply_ptq.py.

  --mode ptq4vit  (Hessian-guided, slower but matches PTQ4ViT paper)
      Uses the full PTQ4ViT pipeline from apply_ptq4vit.py: patch attention,
      wrap modules with PTQ4ViT quant modules, Hessian calibration.

For each bit width the FP32 checkpoint is reloaded, quantized, calibrated with
--calib-num images, then inferred on the sample cells only (no full sweep).
Outputs are stitched into one PNG grid.

Run from this folder:

    python compare_bits_visual.py \\
        --mode ptq4vit \\
        --cells cell0,cell100,cell1000,cell10000,cell10017 \\
        --calib-num 16 \\
        --out compare_fp_w8_w6_w4_ptq4vit.png
"""

import argparse
import glob
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = _ROOT
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from apply_ptq import (
    ConvitImageTranslator,
    calibrate,
    predict,
    wrap_linear_layers,
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_fp32_model(ckpt_path: str, device: torch.device) -> ConvitImageTranslator:
    model = ConvitImageTranslator(pretrained_model_name="convit_small", use_pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ────────────────────────────────────────────────────────────────────
# Mode: MinMax (apply_ptq.py)
# ────────────────────────────────────────────────────────────────────
def run_bit_minmax(ckpt_path, bits, calib_images, sample_paths, device, transform, threshold, detail):
    print(f"\n=== MinMax W{bits}A{bits} ===")
    model = load_fp32_model(ckpt_path, device)
    wrapped = wrap_linear_layers(model, w_bit=bits, a_bit=bits, verbose=False)
    t0 = time.time()
    calibrate(model, wrapped, calib_images, device, transform)
    print(f"W{bits}A{bits}: calib {time.time() - t0:.1f}s")
    outs = [np.array(predict(model, p, device, transform, threshold, detail)) for p in sample_paths]
    return outs


# ────────────────────────────────────────────────────────────────────
# Mode: PTQ4ViT (apply_ptq4vit.py, Hessian)
# ────────────────────────────────────────────────────────────────────
def run_bit_ptq4vit(ckpt_path, bits, calib_images, sample_paths, device, transform, threshold, detail, config_name):
    print(f"\n=== PTQ4ViT W{bits}A{bits} ({config_name}) ===")
    # Lazy import: apply_ptq4vit pulls in utils.quant_calib which is CUDA-tolerant but heavy.
    from importlib import import_module, reload

    from apply_ptq4vit import (
        ConvitImageTranslator as _ConvitPTQ4ViT,
        TargetImageDataset,
        patch_convit_attention,
        predict as _predict_ptq4vit,
        wrap_convit_modules,
    )
    from utils.quant_calib import HessianQuantCalibrator

    cfg = import_module(f"configs.{config_name}")
    reload(cfg)
    cfg.bit = bits
    cfg.w_bit = {name: bits for name in cfg.conv_fc_name_list}
    cfg.a_bit = {name: bits for name in cfg.conv_fc_name_list}
    cfg.A_bit = {name: bits for name in cfg.matmul_name_list}
    cfg.B_bit = {name: bits for name in cfg.matmul_name_list}
    # Trim Hessian search cost for a feasible demo run.
    for key in ("ptqsl_conv2d_kwargs", "ptqsl_linear_kwargs", "ptqsl_matmul_kwargs"):
        d = getattr(cfg, key, None)
        if d is None:
            continue
        d["search_round"] = 1
        d["eq_n"] = min(int(d.get("eq_n", 100)), 30)
        d["init_layerwise"] = False

    model = _ConvitPTQ4ViT(pretrained_model_name="convit_small", use_pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    patch_convit_attention(model.convit_encoder)
    wrapped = wrap_convit_modules(model, cfg)

    calib_ds = TargetImageDataset(calib_images, transform)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=4, shuffle=False, num_workers=0)

    t0 = time.time()
    HessianQuantCalibrator(model, wrapped, calib_loader, sequential=False, batch_size=4).batching_quant_calib()
    print(f"W{bits}A{bits}: Hessian calib {(time.time() - t0) / 60:.1f} min")

    outs = [np.array(_predict_ptq4vit(model, p, device, transform, threshold, detail)) for p in sample_paths]
    return outs


def load_target(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    return np.array(img)


def load_pixelilt(cell_name, results_dir):
    """Load pre-computed PixelILT (FP32) output for a given cell."""
    path = os.path.join(results_dir, f"{cell_name}_pixelILT_convit.png")
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("L").resize((224, 224))
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(_ROOT, "checkpoints", "best_convit_translator.pth"))
    parser.add_argument("--target-dir", default=os.path.join(_ROOT, "data", "sample_targets"))
    parser.add_argument(
        "--cells",
        default="cell100,cell1000,cell10000,cell10017",
        help="Comma-separated target basenames (without .png) to visualize.",
    )
    parser.add_argument("--calib-num", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--detail", type=float, default=2.0)
    parser.add_argument("--mode", choices=["minmax", "ptq4vit"], default="minmax")
    parser.add_argument("--config", default="PTQ4ViT", choices=["PTQ4ViT", "BasePTQ"])
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--pixelilt-dir",
        default=os.path.join(_ROOT, "data", "sample_pixelilt"),
        help="Directory with pre-computed FP32 PixelILT results.",
    )
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}  Mode: {args.mode}")

    if args.out is None:
        args.out = os.path.join(
            _ROOT,
            "results",
            "compare_fp_w8_w6_w4.png" if args.mode == "minmax" else "compare_fp_w8_w6_w4_ptq4vit.png",
        )

    all_images = sorted(glob.glob(os.path.join(args.target_dir, "*.png")))
    if not all_images:
        print(f"[ERROR] No images in {args.target_dir}")
        sys.exit(1)

    wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
    by_base = {os.path.splitext(os.path.basename(p))[0]: p for p in all_images}
    sample_paths = [by_base[w] for w in wanted if w in by_base]
    missing = [w for w in wanted if w not in by_base]
    if missing:
        print(f"[WARN] Not found in target dir: {missing}")
    if not sample_paths:
        sample_paths = all_images[:5]
    print(f"Samples: {[os.path.basename(p) for p in sample_paths]}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    calib_images = all_images[: args.calib_num]

    fp_model = load_fp32_model(args.model, device)
    fp_outs = [np.array(predict(fp_model, p, device, transform, args.threshold, args.detail)) for p in sample_paths]
    del fp_model

    bit_outs = {}
    for bits in (8, 6, 4):
        if args.mode == "minmax":
            bit_outs[bits] = run_bit_minmax(
                args.model, bits, calib_images, sample_paths, device, transform, args.threshold, args.detail
            )
        else:
            bit_outs[bits] = run_bit_ptq4vit(
                args.model, bits, calib_images, sample_paths, device, transform,
                args.threshold, args.detail, args.config,
            )

    cols = ["PixelILT", "FP32", "8 bits", "6 bits", "4 bits"]
    rows = len(sample_paths)
    fig, axes = plt.subplots(rows, len(cols), figsize=(3 * len(cols), 3 * rows))
    if rows == 1:
        axes = np.array([axes])
    for r, p in enumerate(sample_paths):
        cell_name = os.path.splitext(os.path.basename(p))[0]
        pilt = load_pixelilt(cell_name, args.pixelilt_dir)
        if pilt is not None:
            axes[r, 0].imshow(pilt, cmap="gray", vmin=0, vmax=255)
        else:
            axes[r, 0].imshow(fp_outs[r], cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_ylabel(cell_name, fontsize=9)
        axes[r, 1].imshow(fp_outs[r], cmap="gray", vmin=0, vmax=255)
        axes[r, 2].imshow(bit_outs[8][r], cmap="gray", vmin=0, vmax=255)
        axes[r, 3].imshow(bit_outs[6][r], cmap="gray", vmin=0, vmax=255)
        axes[r, 4].imshow(bit_outs[4][r], cmap="gray", vmin=0, vmax=255)
        for c in range(len(cols)):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=10)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nSaved grid: {args.out}")


if __name__ == "__main__":
    main()
