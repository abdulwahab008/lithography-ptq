"""
PTQ (Post-Training Quantization) for PvtImageTranslator
======================================================
Min-max W*A* on all ``nn.Linear`` layers in ``pvt_encoder`` and ``decoder`` (``--bit`` 4, 6, or 8).
ConvTranspose2d stays FP32 (same pattern as ``CViT pretrained/apply_ptq.py``).

Usage (from this folder):

    python apply_ptq_pvt.py \\
        --model best_pvt_translator.pth \\
        --target-dir "../CViT pretrained/target" \\
        --output-dir results_pvt_ptq \\
        --bit 4

Default ``--target-dir`` uses the shared ConViT ``target`` folder for calibration.

Verify::

    python verify_ptq_pvt.py \\
        --fp32-model best_pvt_translator.pth \\
        --ptq-model best_pvt_translator_ptq_w4a4.pth \\
        --bit 4
"""

import os
import sys
import argparse
import glob
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


class MinMaxQuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, w_bit=8, a_bit=8):
        super().__init__(in_features, out_features, bias)
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.mode = "raw"
        self.register_buffer("w_interval", torch.tensor(-1.0))
        self.register_buffer("a_interval", torch.tensor(-1.0))
        self.w_qmax = 2 ** (w_bit - 1)
        self.a_qmax = 2 ** (a_bit - 1)

    def forward(self, x):
        if self.mode == "raw":
            return F.linear(x, self.weight, self.bias)
        if self.mode == "calibration":
            w_max = self.weight.data.abs().max()
            self.w_interval.copy_(w_max / (self.w_qmax - 1))
            a_max = x.detach().abs().max()
            self.a_interval.copy_(a_max / (self.a_qmax - 1))
            return F.linear(x, self.weight, self.bias)
        if self.mode == "quant_forward":
            assert (self.w_interval >= 0).all() and (self.a_interval >= 0).all(), (
                "Run calibration first!"
            )
            w_q = (self.weight / self.w_interval).round().clamp(-self.w_qmax, self.w_qmax - 1)
            w_sim = w_q * self.w_interval
            x_q = (x / self.a_interval).round().clamp(-self.a_qmax, self.a_qmax - 1)
            x_sim = x_q * self.a_interval
            return F.linear(x_sim, w_sim, self.bias)
        raise ValueError(f"Unknown mode: {self.mode}")


def _pvt_feature_dim(pretrained_model_name: str, encoder) -> int:
    if "v2_b0" in pretrained_model_name or "v2_b1" in pretrained_model_name:
        return 256
    if "v2_b2" in pretrained_model_name:
        return 512
    if "v2_b3" in pretrained_model_name or "v2_b4" in pretrained_model_name or "v2_b5" in pretrained_model_name:
        return 512
    if "twins_pcpvt_small" in pretrained_model_name:
        return 512
    if "twins_pcpvt_base" in pretrained_model_name:
        return 768
    if "twins_pcpvt_large" in pretrained_model_name:
        return 1024
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


def wrap_linear_layers(model, w_bit=8, a_bit=8, verbose=True):
    wrapped = {}

    def _log(msg):
        if verbose:
            print(msg)

    def _replace(parent, prefix=""):
        for name, module in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(module, nn.Linear):
                new_m = MinMaxQuantLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    w_bit=w_bit,
                    a_bit=a_bit,
                )
                new_m.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_m.bias.data = module.bias.data.clone()
                setattr(parent, name, new_m)
                wrapped[full_name] = new_m
                _log(f"  Wrapped: {full_name}  [{module.in_features} → {module.out_features}]")
            else:
                _replace(module, full_name)

    _log("\n[PTQ] Wrapping Linear layers in pvt_encoder...")
    _replace(model.pvt_encoder, "pvt_encoder")
    _log("\n[PTQ] Wrapping Linear layers in decoder...")
    _replace(model.decoder, "decoder")
    _log(f"\n[PTQ] Total wrapped layers: {len(wrapped)}")
    return wrapped


def set_mode(wrapped_modules, mode):
    for m in wrapped_modules.values():
        m.mode = mode


def calibrate(model, wrapped_modules, calib_images, device, transform):
    set_mode(wrapped_modules, "calibration")
    model.eval()
    print(f"\n[PTQ] Calibrating with {len(calib_images)} images...")
    with torch.no_grad():
        for img_path in tqdm(calib_images, desc="Calibration"):
            try:
                img = Image.open(img_path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                model(x)
            except Exception as e:
                print(f"  Skipping {img_path}: {e}")
    set_mode(wrapped_modules, "quant_forward")
    print("[PTQ] Calibration done. Switched to quant_forward mode.")


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


def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, ".."))
    _default_target = os.path.join(_root, "data", "sample_targets")
    _default_ckpt = os.path.join(_root, "checkpoints", "best_pvt_translator.pth")
    _default_out = os.path.join(_root, "results", "pvt_ptq")

    parser = argparse.ArgumentParser(description="Apply min-max PTQ to PvtImageTranslator")
    parser.add_argument("--model", type=str, default=_default_ckpt)
    parser.add_argument("--backbone", type=str, default="pvt_v2_b2")
    parser.add_argument("--target-dir", type=str, default=_default_target)
    parser.add_argument("--output-dir", type=str, default=_default_out)
    parser.add_argument("--bit", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--calib-num", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--detail", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PTQ] Device: {device}")
    print(f"[PTQ] Bit-width: W{args.bit}A{args.bit}")

    model_path = args.model if os.path.isabs(args.model) else os.path.join(_here, args.model)
    print(f"\n[PTQ] Loading model from: {model_path}")
    model = PvtImageTranslator(pretrained_model_name=args.backbone, use_pretrained=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    wrapped_modules = wrap_linear_layers(model, w_bit=args.bit, a_bit=args.bit)

    all_images = sorted(glob.glob(os.path.join(args.target_dir, "*.png")))
    if len(all_images) == 0:
        print(f"[ERROR] No .png images in: {args.target_dir}")
        sys.exit(1)

    calib_images = all_images[: args.calib_num]
    print(f"[PTQ] Using {len(calib_images)} / {len(all_images)} images for calibration")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    calibrate(model, wrapped_modules, calib_images, device, transform)

    quant_model_path = os.path.join(
        _here,
        os.path.basename(model_path).replace(".pth", f"_ptq_w{args.bit}a{args.bit}.pth"),
    )
    torch.save(model.state_dict(), quant_model_path)
    print(f"\n[PTQ] Quantized model saved → {quant_model_path}")

    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_here, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[PTQ] Running inference on {len(all_images)} images → {out_dir}")

    for img_path in tqdm(all_images, desc="Inference"):
        try:
            result = predict(model, img_path, device, transform, args.threshold, args.detail)
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{base}_pixelILT_pvt_ptq.png")
            result.save(out_path)
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")

    print(f"\n[PTQ] Done! Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
