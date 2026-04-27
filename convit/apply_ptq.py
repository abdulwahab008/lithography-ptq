"""
PTQ (Post-Training Quantization) for ConvitImageTranslator
==========================================================
Quantizes the convit_encoder Linear layers to W*A* (same bit-width for weights and activations).
Use ``--bit 4``, ``6``, or ``8`` (default 8). Example: ``--bit 4`` → W4A4 checkpoint suffix ``_ptq_w4a4.pth``.
The decoder's ConvTranspose2d layers remain in float32 (not supported by PTQ4ViT).

Usage:
    python apply_ptq.py \
        --model path/to/best_convit_translator.pth \
        --target-dir target \
        --output-dir results_convit_ptq \
        --bit 4

Default ``--target-dir`` is the ``target`` folder next to this script (``CViT pretrained/target``).

To confirm the PTQ checkpoint and compare outputs to FP32, run ``python verify_ptq.py --help``.

Requirements:
    pip install timm torch torchvision tqdm pillow
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

# ──────────────────────────────────────────────
# 1.  Minimal MinMax quantized Linear layer
# ──────────────────────────────────────────────
class MinMaxQuantLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear.
    Calibration step collects min/max of weights and activations,
    then quant_forward uses those to simulate integer arithmetic.
    """
    def __init__(self, in_features, out_features, bias=True, w_bit=8, a_bit=8):
        super().__init__(in_features, out_features, bias)
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.mode = "raw"          # raw | calibration | quant_forward
        self.register_buffer("w_interval", torch.tensor(-1.0))
        self.register_buffer("a_interval", torch.tensor(-1.0))
        self.w_qmax = 2 ** (w_bit - 1)
        self.a_qmax = 2 ** (a_bit - 1)

    def forward(self, x):
        if self.mode == "raw":
            return F.linear(x, self.weight, self.bias)

        elif self.mode == "calibration":
            # Record scale for weights
            w_max = self.weight.data.abs().max()
            self.w_interval.copy_(w_max / (self.w_qmax - 1))
            # Record scale for activations
            a_max = x.detach().abs().max()
            self.a_interval.copy_(a_max / (self.a_qmax - 1))
            return F.linear(x, self.weight, self.bias)

        elif self.mode == "quant_forward":
            assert (self.w_interval >= 0).all() and (self.a_interval >= 0).all(), (
                "Run calibration first!"
            )
            # Quantize weights
            w_q = (self.weight / self.w_interval).round().clamp(-self.w_qmax, self.w_qmax - 1)
            w_sim = w_q * self.w_interval
            # Quantize activations
            x_q = (x / self.a_interval).round().clamp(-self.a_qmax, self.a_qmax - 1)
            x_sim = x_q * self.a_interval
            return F.linear(x_sim, w_sim, self.bias)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ──────────────────────────────────────────────
# 2.  Model definition (same as inference.py)
# ──────────────────────────────────────────────
class ConvitImageTranslator(nn.Module):
    def __init__(self, pretrained_model_name="convit_small", use_pretrained=False):
        super().__init__()
        self.convit_encoder = timm.create_model(
            pretrained_model_name,
            pretrained=use_pretrained,
            num_classes=0
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
            nn.Tanh()
        )

    def forward(self, x):
        features = self.convit_encoder(x)
        output = self.decoder(features)
        if output.shape[-1] != 224 or output.shape[-2] != 224:
            output = F.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)
        return output


# ──────────────────────────────────────────────
# 3.  Replace encoder Linear layers with quantized ones
# ──────────────────────────────────────────────
def wrap_linear_layers(model, w_bit=8, a_bit=8, verbose=True):
    """
    Walk the convit_encoder and replace every nn.Linear
    with MinMaxQuantLinear. Decoder linears are also wrapped.
    ConvTranspose2d layers are left in float32.
    """
    wrapped = {}

    def _log(msg):
        if verbose:
            print(msg)

    def _replace(parent, prefix=""):
        for name, module in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(module, nn.Linear):
                new_m = MinMaxQuantLinear(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    w_bit=w_bit, a_bit=a_bit
                )
                new_m.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new_m.bias.data = module.bias.data.clone()
                setattr(parent, name, new_m)
                wrapped[full_name] = new_m
                _log(f"  Wrapped: {full_name}  [{module.in_features} → {module.out_features}]")
            else:
                _replace(module, full_name)

    _log("\n[PTQ] Wrapping Linear layers in encoder...")
    _replace(model.convit_encoder, "convit_encoder")
    _log(f"\n[PTQ] Wrapping Linear layers in decoder...")
    _replace(model.decoder, "decoder")
    _log(f"\n[PTQ] Total wrapped layers: {len(wrapped)}")
    return wrapped


# ──────────────────────────────────────────────
# 4.  Calibration
# ──────────────────────────────────────────────
def set_mode(wrapped_modules, mode):
    for m in wrapped_modules.values():
        m.mode = mode


def calibrate(model, wrapped_modules, calib_images, device, transform):
    """
    Single forward pass over calibration images to collect min/max statistics.
    """
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


# ──────────────────────────────────────────────
# 5.  Inference helpers (from inference.py)
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
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    output = denorm(output.squeeze(0).cpu())
    output = torch.clamp(output, 0, 1).permute(1, 2, 0).numpy()
    return to_black_white(output, threshold, detail_factor)


# ──────────────────────────────────────────────
# 6.  Main
# ──────────────────────────────────────────────
def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, ".."))
    _default_target = os.path.join(_root, "data", "sample_targets")
    _default_ckpt = os.path.join(_root, "checkpoints", "best_convit_translator.pth")
    _default_out = os.path.join(_root, "results", "convit_ptq")

    parser = argparse.ArgumentParser(description="Apply PTQ to ConvitImageTranslator")
    parser.add_argument("--model",      type=str, default=_default_ckpt,
                        help="Path to the trained .pth checkpoint")
    parser.add_argument("--target-dir", type=str, default=_default_target,
                        help="Directory of target images (used for calibration + inference)")
    parser.add_argument("--output-dir", type=str, default=_default_out,
                        help="Where to save PTQ inference results")
    parser.add_argument("--bit",        type=int, default=8, choices=[4, 6, 8],
                        help="Quantization bit-width for weights and activations")
    parser.add_argument("--calib-num",  type=int, default=32,
                        help="Number of images to use for calibration (default: 32)")
    parser.add_argument("--threshold",  type=float, default=0.3)
    parser.add_argument("--detail",     type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PTQ] Device: {device}")
    print(f"[PTQ] Bit-width: W{args.bit}A{args.bit}")

    # ── Load model ──
    print(f"\n[PTQ] Loading model from: {args.model}")
    model = ConvitImageTranslator(pretrained_model_name="convit_small", use_pretrained=False)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ── Wrap Linear layers ──
    wrapped_modules = wrap_linear_layers(model, w_bit=args.bit, a_bit=args.bit)

    # ── Collect calibration images ──
    all_images = sorted(glob.glob(os.path.join(args.target_dir, "*.png")))
    if len(all_images) == 0:
        print(f"[ERROR] No .png images found in: {args.target_dir}")
        print("  Please update --target-dir to point at your target images folder.")
        sys.exit(1)

    calib_images = all_images[:args.calib_num]
    print(f"[PTQ] Using {len(calib_images)} / {len(all_images)} images for calibration")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ── Calibrate ──
    calibrate(model, wrapped_modules, calib_images, device, transform)

    # ── Save quantized model (next to this script for easy discovery) ──
    quant_model_path = os.path.join(
        _here,
        os.path.basename(args.model).replace(".pth", f"_ptq_w{args.bit}a{args.bit}.pth"),
    )
    torch.save(model.state_dict(), quant_model_path)
    print(f"\n[PTQ] Quantized model saved → {quant_model_path}")

    # ── Run inference on all target images ──
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[PTQ] Running inference on {len(all_images)} images → {args.output_dir}")

    for img_path in tqdm(all_images, desc="Inference"):
        try:
            result = predict(model, img_path, device, transform, args.threshold, args.detail)
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(args.output_dir, f"{base}_pixelILT_convit_ptq.png")
            result.save(out_path)
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")

    print(f"\n[PTQ] Done! Results saved to: {args.output_dir}")
    print("      Compare with FP32 PixelILT in data/sample_pixelilt/ or your own results_convit/.")


if __name__ == "__main__":
    main()
