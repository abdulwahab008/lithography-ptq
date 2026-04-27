"""
PTQ4ViT (Hessian-guided) for ConvitImageTranslator
===================================================
Applies the full PTQ4ViT quantization (Hessian metric, twin uniform for
softmax/GELU) to the ConViT encoder inside ConvitImageTranslator.

The decoder's Linear layers get standard PTQSL quantization.
ConvTranspose2d layers remain in float32.
Bit-width: ``--bit`` 4, 6, or 8 (default 8); outputs use suffix ``_ptq4vit_w{bit}a{bit}.pth``.

Usage:
    python apply_ptq4vit.py \\
        --model best_convit_translator.pth \\
        --target-dir target \\
        --output-dir results_convit_ptq4vit \\
        --bit 4 \\
        --calib-num 32

Compare output quality with results from ``apply_ptq.py`` (min-max PTQ).
"""

import os
import sys
import argparse
import glob
import types
import time

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
# 1. Model definition (same decoder as inference.py)
# ──────────────────────────────────────────────
class ConvitImageTranslator(nn.Module):
    def __init__(self, pretrained_model_name="convit_small", use_pretrained=False):
        super().__init__()
        self.convit_encoder = timm.create_model(
            pretrained_model_name,
            pretrained=use_pretrained,
            num_classes=0,
        )
        self.decoder = nn.Sequential(
            nn.Linear(432, 4096),
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
        features = self.convit_encoder(x)
        output = self.decoder(features)
        if output.shape[-1] != 224 or output.shape[-2] != 224:
            output = F.interpolate(output, size=(224, 224), mode="bilinear", align_corners=False)
        return output


# ──────────────────────────────────────────────
# 2. Patch ConViT attention to use MatMul modules
# ──────────────────────────────────────────────
def _gpsa_get_attention(self, x):
    """Patched GPSA.get_attention that uses self.matmul1 for q@k."""
    B, N, C = x.shape
    qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k = qk[0], qk[1]
    pos_score = self.rel_indices.expand(B, -1, -1, -1)
    pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    patch_score = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    patch_score = patch_score.softmax(dim=-1)
    pos_score = pos_score.softmax(dim=-1)

    gating = self.gating_param.view(1, -1, 1, 1)
    attn = (1.0 - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    attn /= attn.sum(dim=-1).unsqueeze(-1)
    attn = self.attn_drop(attn)
    return attn


def _gpsa_forward(self, x):
    """Patched GPSA.forward that uses self.matmul2 for attn@v."""
    B, N, C = x.shape
    if self.rel_indices is None or self.rel_indices.shape[1] != N:
        self.rel_indices = self.get_rel_indices(N)
    attn = self.get_attention(x)
    v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _mhsa_forward(self, x):
    """Patched MHSA.forward that uses self.matmul1 and self.matmul2."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def patch_convit_attention(encoder):
    """Inject MatMul modules and monkey-patch forward methods on the encoder."""
    for name, module in encoder.named_modules():
        cls_name = type(module).__name__
        if cls_name == "GPSA":
            module.matmul1 = MatMul()
            module.matmul2 = MatMul()
            module.get_attention = types.MethodType(_gpsa_get_attention, module)
            module.forward = types.MethodType(_gpsa_forward, module)
        elif cls_name == "MHSA":
            module.matmul1 = MatMul()
            module.matmul2 = MatMul()
            module.forward = types.MethodType(_mhsa_forward, module)


# ──────────────────────────────────────────────
# 3. ConViT-aware module wrapping for PTQ4ViT
# ──────────────────────────────────────────────
def wrap_convit_modules(model, cfg):
    """
    Replace Linear/Conv2d/MatMul modules with PTQ4ViT quant modules.
    Handles ConViT-specific names: qk, v, pos_proj (GPSA) and qkv (MHSA).
    Decoder linears are wrapped as MLP layers. ConvTranspose2d stays FP32.
    """
    convit_name_map = {
        "qk": "qlinear_qkv",
        "v": "qlinear_proj",
        "qkv": "qlinear_qkv",
        "proj": "qlinear_proj",
        "pos_proj": "qlinear_proj",
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
        short = name[idx + 1 if idx != 0 else idx :]

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
            qtype = convit_name_map.get(short)
            if qtype is None:
                continue
            new_m = cfg.get_module(qtype, m.in_features, m.out_features)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            wrapped_modules[name] = new_m
            setattr(father_module, short, new_m)

        elif isinstance(m, MatMul):
            qtype = convit_name_map.get(short)
            if qtype is None:
                continue
            new_m = cfg.get_module(qtype)
            wrapped_modules[name] = new_m
            setattr(father_module, short, new_m)

    print(f"[PTQ4ViT] Wrapped {len(wrapped_modules)} modules.")
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
    _default_ckpt = os.path.join(_root, "checkpoints", "best_convit_translator.pth")
    _default_out = os.path.join(_root, "results", "convit_ptq4vit")

    parser = argparse.ArgumentParser(description="Apply PTQ4ViT (Hessian) to ConvitImageTranslator")
    parser.add_argument("--model", type=str, default=_default_ckpt)
    parser.add_argument("--target-dir", type=str, default=_default_target)
    parser.add_argument("--output-dir", type=str, default=_default_out)
    parser.add_argument("--bit", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--calib-num", type=int, default=32)
    parser.add_argument("--config", type=str, default="PTQ4ViT", choices=["PTQ4ViT", "BasePTQ"])
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--detail", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PTQ4ViT] Device: {device}")
    print(f"[PTQ4ViT] Bit-width: W{args.bit}A{args.bit}")
    print(f"[PTQ4ViT] Config: {args.config}")

    # ── Load config ──
    cfg = import_module(f"configs.{args.config}")
    reload(cfg)
    cfg.bit = args.bit
    cfg.w_bit = {name: args.bit for name in cfg.conv_fc_name_list}
    cfg.a_bit = {name: args.bit for name in cfg.conv_fc_name_list}
    cfg.A_bit = {name: args.bit for name in cfg.matmul_name_list}
    cfg.B_bit = {name: args.bit for name in cfg.matmul_name_list}

    # ── Load model ──
    print(f"\n[PTQ4ViT] Loading model from: {args.model}")
    model = ConvitImageTranslator(pretrained_model_name="convit_small", use_pretrained=False)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ── Patch attention ──
    print("[PTQ4ViT] Patching ConViT attention with MatMul modules...")
    patch_convit_attention(model.convit_encoder)

    # ── Wrap modules ──
    wrapped = wrap_convit_modules(model, cfg)

    # ── Calibration loader ──
    all_images = sorted(glob.glob(os.path.join(args.target_dir, "*.png")))
    if not all_images:
        print(f"[ERROR] No .png images in: {args.target_dir}")
        sys.exit(1)

    calib_images = all_images[: args.calib_num]
    print(f"[PTQ4ViT] Using {len(calib_images)} / {len(all_images)} images for calibration")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    calib_ds = TargetImageDataset(calib_images, transform)
    calib_loader = torch.utils.data.DataLoader(calib_ds, batch_size=4, shuffle=False)

    # ── Hessian calibration ──
    t0 = time.time()
    print(f"\n[PTQ4ViT] Running {args.config} Hessian calibration...")
    calibrator = HessianQuantCalibrator(
        model, wrapped, calib_loader, sequential=False, batch_size=4,
    )
    calibrator.batching_quant_calib()
    elapsed = time.time() - t0
    print(f"[PTQ4ViT] Calibration done in {elapsed / 60:.1f} min")

    # ── Save quantized model ──
    quant_path = os.path.join(
        _here,
        os.path.basename(args.model).replace(".pth", f"_ptq4vit_w{args.bit}a{args.bit}.pth"),
    )
    torch.save(model.state_dict(), quant_path)
    print(f"[PTQ4ViT] Quantized model saved → {quant_path}")

    # ── Inference ──
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[PTQ4ViT] Running inference on {len(all_images)} images → {args.output_dir}")

    for img_path in tqdm(all_images, desc="Inference"):
        try:
            result = predict(model, img_path, device, transform, args.threshold, args.detail)
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(args.output_dir, f"{base}_pixelILT_convit_ptq4vit.png")
            result.save(out_path)
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")

    print(f"\n[PTQ4ViT] Done! Results saved to: {args.output_dir}")
    print("  Compare with min-max PTQ results in results/convit_ptq/ (after running apply_ptq.py).")


if __name__ == "__main__":
    main()
