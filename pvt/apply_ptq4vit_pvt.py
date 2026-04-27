"""
PTQ4ViT (Hessian-guided) for PvtImageTranslator
=================================================
Applies the full PTQ4ViT quantization (Hessian metric, twin uniform for
softmax/GELU) to the PVT v2 encoder inside PvtImageTranslator.

The decoder's Linear layers get standard PTQSL quantization.
ConvTranspose2d layers remain in float32.
Bit-width: ``--bit`` 4, 6, or 8 (default 8); outputs use suffix ``_ptq4vit_w{bit}a{bit}.pth``.

Usage:
    python apply_ptq4vit_pvt.py \\
        --model best_pvt_translator.pth \\
        --target-dir "../CViT pretrained/target" \\
        --output-dir results_pvt_ptq4vit \\
        --bit 4 \\
        --calib-num 32

Compare output quality with results from ``apply_ptq_pvt.py`` (min-max PTQ).
"""

import os
import sys
import argparse
import glob
import types
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
from skimage import exposure
import timm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.models import MatMul
from utils.quant_calib import HessianQuantCalibrator
from importlib import import_module, reload


# ──────────────────────────────────────────────
# 1. Model definition (same decoder as min-max version)
# ──────────────────────────────────────────────
def _pvt_feature_dim(pretrained_model_name: str, encoder) -> int:
    if "v2_b0" in pretrained_model_name or "v2_b1" in pretrained_model_name:
        return 256
    if "v2_b2" in pretrained_model_name:
        return 512
    if any(s in pretrained_model_name for s in ("v2_b3", "v2_b4", "v2_b5")):
        return 512
    try:
        return encoder.head.in_features
    except Exception:
        return 512


class PvtImageTranslator(nn.Module):
    def __init__(self, pretrained_model_name="pvt_v2_b2", use_pretrained=False):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.pvt_encoder = timm.create_model(
            pretrained_model_name,
            pretrained=use_pretrained,
            num_classes=0,
        )
        d = _pvt_feature_dim(pretrained_model_name, self.pvt_encoder)
        self.decoder = nn.Sequential(
            nn.Linear(d, 4096),
            nn.ReLU(),
            nn.Linear(4096, 14 * 14 * 256),
            nn.ReLU(),
            nn.Unflatten(1, (256, 14, 14)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.pvt_encoder(x)
        output = self.decoder(features)
        if output.shape[-1] != 224 or output.shape[-2] != 224:
            output = F.interpolate(output, size=(224, 224), mode="bilinear", align_corners=False)
        return output


# ──────────────────────────────────────────────
# 2. Patch PVT attention to use MatMul modules
# ──────────────────────────────────────────────
def _pvt_attention_forward(self, x, feat_size: List[int]):
    """Patched PVT Attention.forward that uses self.matmul1/matmul2
    instead of inline @ and disables F.scaled_dot_product_attention."""
    B, N, C = x.shape
    H, W = feat_size
    q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

    if self.pool is not None:
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    else:
        if self.sr is not None:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)

    q = q * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1))
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v)

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def patch_pvt_attention(encoder):
    """Inject MatMul modules and monkey-patch forward methods on PVT encoder."""
    count = 0
    for name, module in encoder.named_modules():
        cls_name = type(module).__name__
        if cls_name == "Attention":
            module.matmul1 = MatMul()
            module.matmul2 = MatMul()
            module.forward = types.MethodType(_pvt_attention_forward, module)
            count += 2
    print(f"[PTQ4ViT-PVT] Injected {count} MatMul modules into {count // 2} Attention blocks.")


# ──────────────────────────────────────────────
# 3. PVT-aware module wrapping for PTQ4ViT
# ──────────────────────────────────────────────
def wrap_pvt_modules(model, cfg):
    """
    Replace Linear/Conv2d/MatMul modules with PTQ4ViT quant modules.
    Handles PVT-specific names: q, kv (attention) and fc1, fc2 (MLP).
    Decoder linears are wrapped as MLP layers. ConvTranspose2d stays FP32.
    """
    pvt_name_map = {
        "q": "qlinear_proj",
        "kv": "qlinear_proj",
        "proj": "qlinear_proj",
        "fc1": "qlinear_MLP_1",
        "fc2": "qlinear_MLP_2",
        "matmul1": "qmatmul_qk",
        "matmul2": "qmatmul_scorev",
    }

    wrapped_modules = {}
    module_dict = {}

    it = [(n, m) for n, m in model.named_modules()]
    for name, m in it:
        module_dict[name] = m
        idx = name.rfind(".")
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name not in module_dict:
            continue
        father_module = module_dict[father_name]
        short = name[idx + 1 if idx != 0 else idx:]

        if isinstance(m, nn.Conv2d) and not isinstance(m, nn.ConvTranspose2d):
            new_m = cfg.get_module(
                "qconv",
                m.in_channels, m.out_channels, m.kernel_size, m.stride,
                m.padding, m.dilation, m.groups, m.bias is not None, m.padding_mode,
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            wrapped_modules[name] = new_m
            setattr(father_module, short, new_m)

        elif isinstance(m, nn.Linear):
            qtype = pvt_name_map.get(short)
            if qtype is None:
                continue
            new_m = cfg.get_module(qtype, m.in_features, m.out_features)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            wrapped_modules[name] = new_m
            setattr(father_module, short, new_m)

        elif isinstance(m, MatMul):
            qtype = pvt_name_map.get(short)
            if qtype is None:
                continue
            new_m = cfg.get_module(qtype)
            wrapped_modules[name] = new_m
            setattr(father_module, short, new_m)

    print(f"[PTQ4ViT-PVT] Wrapped {len(wrapped_modules)} modules.")
    return wrapped_modules


# ──────────────────────────────────────────────
# 4. Calibration DataLoader from target PNGs
# ──────────────────────────────────────────────
class TargetImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 0


# ──────────────────────────────────────────────
# 5. Inference helpers
# ──────────────────────────────────────────────
def enhance_details(image, detail_factor=2.0):
    image_float = image.astype(np.float32)
    image_enhanced = exposure.equalize_adapthist(image_float)
    blurred = cv2.GaussianBlur(image_enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(image_enhanced, 1.0 + detail_factor, blurred, -detail_factor, 0)
    return np.clip(sharpened, 0, 1)


def to_black_white(output_image, threshold=0.3, detail_factor=1.5):
    if len(output_image.shape) == 3 and output_image.shape[2] == 3:
        grayscale = np.mean(output_image, axis=2)
    else:
        grayscale = output_image
    grayscale_enhanced = enhance_details(grayscale, detail_factor)
    binary = np.zeros_like(grayscale_enhanced)
    binary[grayscale_enhanced > threshold] = 1.0
    return Image.fromarray((binary * 255).astype(np.uint8))


def predict(model, image_path, device, transform, threshold=0.3, detail_factor=1.5):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)
    denorm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    output = denorm(output.squeeze(0).cpu())
    output = torch.clamp(output, 0, 1).permute(1, 2, 0).numpy()
    return to_black_white(output, threshold, detail_factor)


# ──────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────
def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, ".."))
    _default_target = os.path.join(_root, "data", "sample_targets")
    _default_ckpt = os.path.join(_root, "checkpoints", "best_pvt_translator.pth")
    _default_out = os.path.join(_root, "results", "pvt_ptq4vit")

    parser = argparse.ArgumentParser(description="Apply PTQ4ViT (Hessian) to PvtImageTranslator")
    parser.add_argument("--model", type=str, default=_default_ckpt)
    parser.add_argument("--backbone", type=str, default="pvt_v2_b2")
    parser.add_argument("--target-dir", type=str, default=_default_target)
    parser.add_argument("--output-dir", type=str, default=_default_out)
    parser.add_argument("--bit", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--calib-num", type=int, default=32)
    parser.add_argument("--config", type=str, default="PTQ4ViT", choices=["PTQ4ViT", "BasePTQ"])
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--detail", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PTQ4ViT-PVT] Device: {device}")
    print(f"[PTQ4ViT-PVT] Bit-width: W{args.bit}A{args.bit}")
    print(f"[PTQ4ViT-PVT] Config: {args.config}")

    # ── Load config ──
    cfg = import_module(f"configs.{args.config}")
    reload(cfg)
    cfg.bit = args.bit
    cfg.w_bit = {name: args.bit for name in cfg.conv_fc_name_list}
    cfg.a_bit = {name: args.bit for name in cfg.conv_fc_name_list}
    cfg.A_bit = {name: args.bit for name in cfg.matmul_name_list}
    cfg.B_bit = {name: args.bit for name in cfg.matmul_name_list}

    # ── Load model ──
    model_path = args.model if os.path.isabs(args.model) else os.path.join(_here, args.model)
    print(f"\n[PTQ4ViT-PVT] Loading model from: {model_path}")
    model = PvtImageTranslator(pretrained_model_name=args.backbone, use_pretrained=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ── Patch attention ──
    print("[PTQ4ViT-PVT] Patching PVT attention with MatMul modules...")
    patch_pvt_attention(model.pvt_encoder)

    # ── Wrap modules ──
    wrapped = wrap_pvt_modules(model, cfg)

    # ── Calibration loader ──
    all_images = sorted(glob.glob(os.path.join(args.target_dir, "*.png")))
    if not all_images:
        print(f"[ERROR] No .png images in: {args.target_dir}")
        sys.exit(1)

    calib_images = all_images[: args.calib_num]
    print(f"[PTQ4ViT-PVT] Using {len(calib_images)} / {len(all_images)} images for calibration")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    calib_ds = TargetImageDataset(calib_images, transform)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=4, shuffle=False)

    # ── Hessian calibration ──
    t0 = time.time()
    print(f"\n[PTQ4ViT-PVT] Running {args.config} Hessian calibration...")
    calibrator = HessianQuantCalibrator(
        model, wrapped, calib_loader, sequential=False, batch_size=4,
    )
    calibrator.batching_quant_calib()
    elapsed = time.time() - t0
    print(f"[PTQ4ViT-PVT] Calibration done in {elapsed / 60:.1f} min")

    # ── Save quantized model ──
    quant_path = os.path.join(
        _here,
        os.path.basename(model_path).replace(".pth", f"_ptq4vit_w{args.bit}a{args.bit}.pth"),
    )
    torch.save(model.state_dict(), quant_path)
    print(f"[PTQ4ViT-PVT] Quantized model saved → {quant_path}")

    # ── Inference ──
    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_here, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[PTQ4ViT-PVT] Running inference on {len(all_images)} images → {out_dir}")

    for img_path in tqdm(all_images, desc="Inference"):
        try:
            result = predict(model, img_path, device, transform, args.threshold, args.detail)
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{base}_pixelILT_pvt_ptq4vit.png")
            result.save(out_path)
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")

    print(f"\n[PTQ4ViT-PVT] Done! Results saved to: {out_dir}")
    print(f"  Compare with min-max PTQ results in: results_pvt_ptq/")


if __name__ == "__main__":
    main()
