"""Verify PVT PTQ checkpoint vs FP32."""

import argparse
import glob
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from apply_ptq_pvt import (
    PvtImageTranslator,
    MinMaxQuantLinear,
    wrap_linear_layers,
    set_mode,
)


def default_image_path():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    sample_dir = os.path.join(root, "data", "sample_targets")
    if not os.path.isdir(sample_dir):
        return None
    pngs = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
    return pngs[0] if pngs else None


def load_tensor(image_path, device):
    t = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    return t(img).unsqueeze(0).to(device)


def inspect_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        print("[FAIL] Checkpoint is not a state_dict.")
        return False
    wi = [k for k in ckpt if k.endswith(".w_interval")]
    ai = [k for k in ckpt if k.endswith(".a_interval")]
    print(f"[OK] Checkpoint keys: {len(ckpt)}")
    print(f"     w_interval: {len(wi)}, a_interval: {len(ai)}")
    if len(wi) == 0 or len(ai) == 0:
        print("[FAIL] Missing quant scales. Re-run apply_ptq_pvt.py.")
        return False
    return True


def verify_modules(model):
    qmods = [(n, m) for n, m in model.named_modules() if isinstance(m, MinMaxQuantLinear)]
    print(f"[OK] MinMaxQuantLinear layers: {len(qmods)}")
    wrong = [n for n, m in qmods if m.mode != "quant_forward"]
    bad = [n for n, m in qmods if (m.w_interval <= 0).any().item() or (m.a_interval <= 0).any().item()]
    if wrong:
        print(f"[FAIL] Not quant_forward: {wrong[:5]}...")
        return False
    if bad:
        print(f"[FAIL] Bad intervals: {bad[:5]}...")
        return False
    print("[OK] All quant linears valid")
    return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    default_fp32 = os.path.join(root, "checkpoints", "best_pvt_translator.pth")
    default_ptq = os.path.join(here, "best_pvt_translator_ptq_w8a8.pth")
    if not os.path.isfile(default_ptq):
        default_ptq = os.path.join(root, "checkpoints", "best_pvt_translator_ptq_w8a8.pth")

    parser = argparse.ArgumentParser(description="Verify PVT PTQ checkpoint")
    parser.add_argument("--fp32-model", type=str, default=default_fp32)
    parser.add_argument(
        "--ptq-model",
        type=str,
        default=default_ptq,
    )
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--bit", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--backbone", type=str, default="pvt_v2_b2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = args.image or default_image_path()
    if not img_path or not os.path.isfile(img_path):
        print("[FAIL] Pass --image or add PNGs under data/sample_targets/")
        sys.exit(1)

    print(f"Device: {device}\nImage: {img_path}\n")

    print("=== 1. Checkpoint ===")
    if not inspect_checkpoint(args.ptq_model):
        sys.exit(1)

    print("\n=== 2. PTQ model ===")
    model_q = PvtImageTranslator(pretrained_model_name=args.backbone, use_pretrained=False)
    wrapped = wrap_linear_layers(model_q, w_bit=args.bit, a_bit=args.bit, verbose=False)
    try:
        model_q.load_state_dict(torch.load(args.ptq_model, map_location=device), strict=True)
    except RuntimeError as e:
        print(f"[FAIL] load_state_dict: {e}")
        sys.exit(1)
    model_q.to(device)
    model_q.eval()
    set_mode(wrapped, "quant_forward")
    if not verify_modules(model_q):
        sys.exit(1)

    print("\n=== 3. FP32 baseline ===")
    fp32_path = args.fp32_model if os.path.isabs(args.fp32_model) else os.path.join(here, args.fp32_model)
    model_fp = PvtImageTranslator(pretrained_model_name=args.backbone, use_pretrained=False)
    model_fp.load_state_dict(torch.load(fp32_path, map_location=device), strict=True)
    model_fp.to(device)
    model_fp.eval()

    x = load_tensor(img_path, device)
    with torch.no_grad():
        y_fp = model_fp(x)
        y_q = model_q(x)

    diff = (y_fp - y_q).abs()
    print(f"Mean |FP32 - PTQ|: {diff.mean().item():.6f}")
    cos = torch.nn.functional.cosine_similarity(y_fp.flatten(), y_q.flatten(), dim=0).item()
    print(f"Cosine similarity: {cos:.6f}")
    if diff.max().item() == 0.0:
        print("[FAIL] Identical outputs — quant may be inactive")
        sys.exit(1)
    if cos < 0.95:
        print("[WARN] Low similarity — align checkpoints or increase calib images")

    print("\n[PASS] PVT PTQ verification OK.")


if __name__ == "__main__":
    main()
