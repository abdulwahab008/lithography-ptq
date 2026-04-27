"""
Verify that a PTQ checkpoint is valid and compare outputs vs FP32.

Run from ``CViT pretrained``:

    python verify_ptq.py \\
        --fp32-model /path/to/best_convit_translator.pth \\
        --ptq-model best_convit_translator_ptq_w8a8.pth \\
        --image target/cell0.png

Checks:
  1. Checkpoint contains ``w_interval`` / ``a_interval`` buffers (per quant layer).
  2. After load, all ``MinMaxQuantLinear`` modules use ``quant_forward`` and positive scales.
  3. Same input → PTQ output is close to FP32 but not bitwise-identical (quantization is active).
"""

import argparse
import glob
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from apply_ptq import (
    ConvitImageTranslator,
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
        print("[FAIL] Checkpoint is not a state_dict (expected flat dict of tensors).")
        return False
    wi = [k for k in ckpt if k.endswith(".w_interval")]
    ai = [k for k in ckpt if k.endswith(".a_interval")]
    print(f"[OK] Checkpoint keys: {len(ckpt)} total")
    print(f"     w_interval tensors: {len(wi)}")
    print(f"     a_interval tensors: {len(ai)}")
    if len(wi) == 0 or len(ai) == 0:
        print(
            "[FAIL] Missing quant scales — this is not a PTQ save from apply_ptq.py "
            "(or an old checkpoint before buffers were added). Re-run apply_ptq.py."
        )
        return False
    bad = [
        k
        for k in wi
        if (ckpt[k] <= 0).any().item()
    ]
    if bad:
        print(f"[FAIL] Non-positive w_interval in: {bad[:3]}...")
        return False
    return True


def verify_modules(model):
    qmods = [(n, m) for n, m in model.named_modules() if isinstance(m, MinMaxQuantLinear)]
    print(f"[OK] MinMaxQuantLinear layers in model: {len(qmods)}")
    wrong_mode = [n for n, m in qmods if m.mode != "quant_forward"]
    bad_scale = [
        n
        for n, m in qmods
        if (m.w_interval <= 0).any().item() or (m.a_interval <= 0).any().item()
    ]
    if wrong_mode:
        print(f"[FAIL] These layers are not in quant_forward mode: {wrong_mode[:5]}...")
        return False
    if bad_scale:
        print(f"[FAIL] Invalid intervals on: {bad_scale[:5]}...")
        return False
    print("[OK] All quant linears: mode=quant_forward, intervals > 0")
    return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    default_ptq = os.path.join(here, "best_convit_translator_ptq_w8a8.pth")
    if not os.path.isfile(default_ptq):
        default_ptq = os.path.join(root, "checkpoints", "best_convit_translator_ptq_w8a8.pth")

    parser = argparse.ArgumentParser(description="Verify PTQ checkpoint and FP32 vs PTQ outputs")
    parser.add_argument(
        "--fp32-model",
        type=str,
        default=os.path.join(root, "checkpoints", "best_convit_translator.pth"),
        help="Original float32 .pth (same architecture as training)",
    )
    parser.add_argument(
        "--ptq-model",
        type=str,
        default=default_ptq,
        help="PTQ checkpoint from apply_ptq.py",
    )
    parser.add_argument("--image", type=str, default=None, help="RGB PNG/JPEG for forward test")
    parser.add_argument("--bit", type=int, default=8, choices=[4, 6, 8])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = args.image or default_image_path()
    if not img_path or not os.path.isfile(img_path):
        print("[FAIL] No test image: pass --image path/to.png or add PNGs under data/sample_targets/")
        sys.exit(1)

    print(f"Device: {device}\nImage: {img_path}\n")

    print("=== 1. Checkpoint inspection ===")
    if not inspect_checkpoint(args.ptq_model):
        sys.exit(1)

    print("\n=== 2. Load PTQ model ===")
    model_q = ConvitImageTranslator(pretrained_model_name="convit_small", use_pretrained=False)
    wrapped = wrap_linear_layers(model_q, w_bit=args.bit, a_bit=args.bit, verbose=False)
    state_q = torch.load(args.ptq_model, map_location=device)
    try:
        model_q.load_state_dict(state_q, strict=True)
    except RuntimeError as e:
        print(f"[FAIL] PTQ state_dict does not match wrapped model:\n{e}")
        sys.exit(1)
    model_q.to(device)
    model_q.eval()
    set_mode(wrapped, "quant_forward")

    if not verify_modules(model_q):
        sys.exit(1)

    print("\n=== 3. FP32 baseline (no quant wrappers) ===")
    model_fp = ConvitImageTranslator(pretrained_model_name="convit_small", use_pretrained=False)
    state_fp = torch.load(args.fp32_model, map_location=device)
    model_fp.load_state_dict(state_fp, strict=True)
    model_fp.to(device)
    model_fp.eval()

    x = load_tensor(img_path, device)
    with torch.no_grad():
        y_fp = model_fp(x)
        y_q = model_q(x)

    diff = (y_fp - y_q).abs()
    print(f"Output shape: {tuple(y_q.shape)}")
    print(f"Mean |FP32 - PTQ|: {diff.mean().item():.6f}")
    print(f"Max  |FP32 - PTQ|: {diff.max().item():.6f}")

    if diff.max().item() == 0.0:
        print(
            "[FAIL] Outputs are identical — PTQ forward may not be applied "
            "(check modes and that you loaded the PTQ state_dict)."
        )
        sys.exit(1)

    cos = torch.nn.functional.cosine_similarity(
        y_fp.flatten(), y_q.flatten(), dim=0
    ).item()
    print(f"Cosine similarity (flattened outputs): {cos:.6f}")
    if cos < 0.95:
        print("[WARN] Low similarity; try more calibration images or W8A8 only.")

    print("\n=== Summary ===")
    print("[PASS] PTQ checkpoint has scales, modules are in quant_forward, outputs differ from FP32 as expected.")
    print("       (Small numerical gap is normal; compare images in results_convit_ptq/ vs FP32 inference.)")


if __name__ == "__main__":
    main()
