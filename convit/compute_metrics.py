"""
Compute EPE (Edge Placement Error) and L2 Loss between the FP32 PixelILT
baseline and each quantized bit-width (8 / 6 / 4 bits) for both MinMax and
PTQ4ViT schemes.

Quantized models are built on the fly (same as compare_bits_visual.py).
The FP32 PixelILT reference is loaded from results_convit/.

Produces:
  1.  A per-cell / per-config table on stdout.
  2.  A bar-chart PNG  (metrics_comparison.png).

    python compute_metrics.py
    python compute_metrics.py --cells cell100,cell1000,cell10000
    python compute_metrics.py --cells cell100,cell1000 --out my_metrics.png
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
from scipy import ndimage
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


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_fp32_model(ckpt, device):
    model = ConvitImageTranslator(pretrained_model_name="convit_small", use_pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model


def to_binary(img_array):
    """Convert a grayscale numpy array (0-255) to binary 0/1."""
    arr = np.array(img_array, dtype=np.float64)
    if arr.max() > 1.0:
        arr /= 255.0
    return (arr > 0.5).astype(np.float64)


def load_binary_from_file(path):
    img = Image.open(path).convert("L").resize((224, 224))
    return to_binary(np.array(img))


def edge_mask(binary):
    dilated = ndimage.binary_dilation(binary)
    eroded = ndimage.binary_erosion(binary)
    return (dilated.astype(np.float64) - eroded.astype(np.float64)).clip(0, 1)


def epe(ref_bin, pred_bin):
    """Average pixel distance from predicted edges to nearest reference edge."""
    ref_edges = edge_mask(ref_bin).astype(bool)
    pred_edges = edge_mask(pred_bin).astype(bool)
    if not ref_edges.any() or not pred_edges.any():
        return 0.0
    dist = ndimage.distance_transform_edt(~ref_edges)
    return float(dist[pred_edges].mean())


def l2_loss(ref_bin, pred_bin):
    return float(np.mean((ref_bin - pred_bin) ** 2))


# ── MinMax quantized inference ──────────────────────────────────────
def run_minmax(ckpt, bits, calib_imgs, sample_paths, device, transform, th, det):
    model = load_fp32_model(ckpt, device)
    wrapped = wrap_linear_layers(model, w_bit=bits, a_bit=bits, verbose=False)
    calibrate(model, wrapped, calib_imgs, device, transform)
    outs = [np.array(predict(model, p, device, transform, th, det)) for p in sample_paths]
    return outs


# ── PTQ4ViT quantized inference ────────────────────────────────────
def run_ptq4vit(ckpt, bits, calib_imgs, sample_paths, device, transform, th, det, config_name):
    from importlib import import_module, reload
    from apply_ptq4vit import (
        ConvitImageTranslator as _CIT,
        TargetImageDataset,
        patch_convit_attention,
        predict as _pred,
        wrap_convit_modules,
    )
    from utils.quant_calib import HessianQuantCalibrator

    cfg = import_module(f"configs.{config_name}")
    reload(cfg)
    cfg.bit = bits
    cfg.w_bit = {n: bits for n in cfg.conv_fc_name_list}
    cfg.a_bit = {n: bits for n in cfg.conv_fc_name_list}
    cfg.A_bit = {n: bits for n in cfg.matmul_name_list}
    cfg.B_bit = {n: bits for n in cfg.matmul_name_list}
    for key in ("ptqsl_conv2d_kwargs", "ptqsl_linear_kwargs", "ptqsl_matmul_kwargs"):
        d = getattr(cfg, key, None)
        if d is None:
            continue
        d["search_round"] = 1
        d["eq_n"] = min(int(d.get("eq_n", 100)), 30)
        d["init_layerwise"] = False

    model = _CIT(pretrained_model_name="convit_small", use_pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    patch_convit_attention(model.convit_encoder)
    wrapped = wrap_convit_modules(model, cfg)

    ds = TargetImageDataset(calib_imgs, transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    HessianQuantCalibrator(model, wrapped, loader, sequential=False, batch_size=4).batching_quant_calib()

    outs = [np.array(_pred(model, p, device, transform, th, det)) for p in sample_paths]
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(_ROOT, "checkpoints", "best_convit_translator.pth"))
    parser.add_argument("--target-dir", default=os.path.join(_ROOT, "data", "sample_targets"))
    parser.add_argument("--pixelilt-dir", default=os.path.join(_ROOT, "data", "sample_pixelilt"))
    parser.add_argument("--cells", default="cell100,cell1000,cell10000,cell10017")
    parser.add_argument("--calib-num", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--detail", type=float, default=2.0)
    parser.add_argument("--config", default="PTQ4ViT")
    parser.add_argument("--out", default=os.path.join(_ROOT, "results", "metrics_comparison.png"))
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_images = sorted(glob.glob(os.path.join(args.target_dir, "*.png")))
    calib_images = all_images[: args.calib_num]

    wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
    by_base = {os.path.splitext(os.path.basename(p))[0]: p for p in all_images}
    sample_paths = [by_base[w] for w in wanted if w in by_base]
    cell_names = [os.path.splitext(os.path.basename(p))[0] for p in sample_paths]
    print(f"Cells: {cell_names}")

    ref_bins = {}
    for cn in cell_names:
        fp = os.path.join(args.pixelilt_dir, f"{cn}_pixelILT_convit.png")
        if os.path.exists(fp):
            ref_bins[cn] = load_binary_from_file(fp)
        else:
            print(f"[WARN] No FP32 PixelILT for {cn}, computing on the fly...")
            m = load_fp32_model(args.model, device)
            out = np.array(predict(m, by_base[cn], device, transform, args.threshold, args.detail))
            ref_bins[cn] = to_binary(out)
            del m

    configs = []
    bit_widths = [8, 6, 4]

    for bits in bit_widths:
        label = f"MinMax {bits} bits"
        print(f"\n--- {label} ---")
        t0 = time.time()
        outs = run_minmax(args.model, bits, calib_images, sample_paths, device, transform,
                          args.threshold, args.detail)
        print(f"  done in {time.time() - t0:.1f}s")
        configs.append((label, bits, "minmax", outs))

    for bits in bit_widths:
        label = f"PTQ4ViT {bits} bits"
        print(f"\n--- {label} ---")
        t0 = time.time()
        outs = run_ptq4vit(args.model, bits, calib_images, sample_paths, device, transform,
                           args.threshold, args.detail, args.config)
        print(f"  done in {time.time() - t0:.1f}s")
        configs.append((label, bits, "ptq4vit", outs))

    results = {}
    print(f"\n{'Cell':<14}", end="")
    for label, _, _, _ in configs:
        print(f"  {label:>18s} EPE  {' ':>3s}L2", end="")
    print()
    print("-" * (14 + len(configs) * 30))

    for idx, cn in enumerate(cell_names):
        ref = ref_bins[cn]
        print(f"{cn:<14}", end="")
        for label, bits, mode, outs in configs:
            pred = to_binary(outs[idx])
            e = epe(ref, pred)
            l = l2_loss(ref, pred)
            results.setdefault(label, {"epe": [], "l2": []})
            results[label]["epe"].append(e)
            results[label]["l2"].append(l)
            print(f"  {e:>10.3f}  {l:>10.5f}     ", end="")
        print()

    print("-" * (14 + len(configs) * 30))
    print(f"{'MEAN':<14}", end="")
    for label, _, _, _ in configs:
        me = np.mean(results[label]["epe"])
        ml = np.mean(results[label]["l2"])
        print(f"  {me:>10.3f}  {ml:>10.5f}     ", end="")
    print()

    labels = [c[0] for c in configs]
    mean_epe = [np.mean(results[l]["epe"]) for l in labels]
    mean_l2 = [np.mean(results[l]["l2"]) for l in labels]

    n = len(labels)
    mm_idx = [i for i, c in enumerate(configs) if c[2] == "minmax"]
    hq_idx = [i for i, c in enumerate(configs) if c[2] == "ptq4vit"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(bit_widths))
    w = 0.35
    c_mm, c_hq = "#4C72B0", "#DD8452"

    mm_epe = [mean_epe[i] for i in mm_idx]
    hq_epe = [mean_epe[i] for i in hq_idx]
    mm_l2 = [mean_l2[i] for i in mm_idx]
    hq_l2 = [mean_l2[i] for i in hq_idx]

    b1 = ax1.bar(x - w / 2, mm_epe, w, label="MinMax", color=c_mm)
    b2 = ax1.bar(x + w / 2, hq_epe, w, label="PTQ4ViT", color=c_hq)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b} bits" for b in bit_widths])
    ax1.set_ylabel("EPE (pixels)")
    ax1.set_title("Mean Edge Placement Error")
    ax1.legend()
    for bar_set in (b1, b2):
        for bar in bar_set:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    b3 = ax2.bar(x - w / 2, mm_l2, w, label="MinMax", color=c_mm)
    b4 = ax2.bar(x + w / 2, hq_l2, w, label="PTQ4ViT", color=c_hq)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b} bits" for b in bit_widths])
    ax2.set_ylabel("L2 Loss (MSE)")
    ax2.set_title("Mean L2 Loss")
    ax2.legend()
    for bar_set in (b3, b4):
        for bar in bar_set:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved chart: {args.out}")


if __name__ == "__main__":
    main()
