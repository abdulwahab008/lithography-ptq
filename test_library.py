"""
Test that the PTQ4ViT library components work end-to-end.

No ImageNet required — uses a single pretrained timm ViT with random
calibration tensors to verify:
  1. Model loading + MatMul patching  (utils/models.py)
  2. Module wrapping with PTQ4ViT and BasePTQ configs
  3. HessianQuantCalibrator calibration (utils/quant_calib.py)
  4. quant_forward mode produces outputs that differ from FP32
  5. Quantized int8 weight conversion (utils/integer.py)

Run from PTQ4ViT/:
    python test_library.py
"""

import sys
import os
import copy
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.models import get_net
from utils.device import DEVICE
import utils.net_wrap as net_wrap
from utils.quant_calib import HessianQuantCalibrator, QuantCalibrator
from utils.integer import quantize_int_weight


PASS = 0
FAIL = 0


def report(name, ok, detail=""):
    global PASS, FAIL
    tag = "[OK]" if ok else "[FAIL]"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  {tag} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


class _FakeDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return torch.randn(3, 224, 224), torch.randint(0, 1000, ()).item()


def fake_calib_loader(model, n_images=4, batch_size=2):
    ds = _FakeDataset(n_images)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def test_model_loading(model_name):
    print(f"\n=== 1. Load model: {model_name} ===")
    try:
        net = get_net(model_name)
        net.eval()
        report("get_net()", True, f"type={type(net).__name__}")
    except Exception as e:
        report("get_net()", False, str(e))
        return None

    has_matmul = any(
        "MatMul" in type(m).__name__
        for m in net.modules()
    )
    report("MatMul modules injected", has_matmul)

    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224).to(DEVICE)
        net.to(DEVICE)
        y = net(x)
    report("FP32 forward pass", y.shape[1] == 1000, f"output shape={tuple(y.shape)}")
    return net


def test_wrapping(net, config_name):
    print(f"\n=== 2. Wrap with config: {config_name} ===")
    from importlib import import_module, reload

    try:
        cfg = import_module(f"configs.{config_name}")
        reload(cfg)
    except Exception as e:
        report(f"load config {config_name}", False, str(e))
        return None

    try:
        wrapped = net_wrap.wrap_modules_in_net(net, cfg)
        n = len(wrapped)
        report(f"wrap_modules_in_net", n > 0, f"{n} modules wrapped")
    except Exception as e:
        report("wrap_modules_in_net", False, str(e))
        return None

    return wrapped


def test_calibration(net, wrapped, calib_loader, use_hessian=True):
    label = "HessianQuantCalibrator" if use_hessian else "QuantCalibrator"
    print(f"\n=== 3. Calibrate: {label} ===")

    try:
        t0 = time.time()
        if use_hessian:
            cal = HessianQuantCalibrator(
                net, wrapped, calib_loader, sequential=False, batch_size=2
            )
            cal.batching_quant_calib()
        else:
            cal = QuantCalibrator(net, wrapped, calib_loader, sequential=False)
            cal.parallel_quant_calib()
        elapsed = time.time() - t0
        report(f"{label} calibration", True, f"{elapsed:.1f}s")
    except Exception as e:
        report(f"{label} calibration", False, str(e))
        return False

    modes = set()
    for m in wrapped.values():
        modes.add(getattr(m, "mode", "?"))
    all_qf = modes == {"quant_forward"}
    report("All modules in quant_forward", all_qf, f"modes={modes}")
    return True


def test_quant_vs_fp32(net_q, net_fp, device):
    print(f"\n=== 4. FP32 vs quantized output ===")
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        y_fp = net_fp(x)
        y_q = net_q(x)

    diff = (y_fp - y_q).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    cos = nn.functional.cosine_similarity(
        y_fp.flatten(), y_q.flatten(), dim=0
    ).item()

    report("Outputs differ (PTQ active)", max_diff > 0, f"max_diff={max_diff:.6f}")
    report("Cosine similarity reasonable", cos > 0.5, f"cos={cos:.4f}")
    print(f"     mean|diff|={mean_diff:.6f}  max|diff|={max_diff:.6f}  cos={cos:.6f}")


def test_int8_weights(wrapped):
    print(f"\n=== 5. Int8 weight conversion ===")
    sample_name = list(wrapped.keys())[0]
    module = wrapped[sample_name]
    try:
        w_int = quantize_int_weight(module)
        report(
            "quantize_int_weight",
            w_int.dtype == torch.int8,
            f"dtype={w_int.dtype}, shape={tuple(w_int.shape)}, layer={sample_name}",
        )
    except Exception as e:
        report("quantize_int_weight", False, str(e))


def main():
    global PASS, FAIL
    model_name = "vit_tiny_patch16_224"
    config_name = "PTQ4ViT"

    print(f"PTQ4ViT Library Test")
    print(f"Device: {DEVICE}")
    print(f"Model:  {model_name}")
    print(f"Config: {config_name}")

    net = test_model_loading(model_name)
    if net is None:
        print("\n[ABORT] Model loading failed.")
        sys.exit(1)

    net_fp = copy.deepcopy(net)
    net_fp.to(DEVICE)
    net_fp.eval()

    wrapped = test_wrapping(net, config_name)
    if wrapped is None:
        print("\n[ABORT] Wrapping failed.")
        sys.exit(1)

    net.to(DEVICE)
    net.eval()

    calib_loader = fake_calib_loader(net, n_images=4, batch_size=2)
    ok = test_calibration(net, wrapped, calib_loader, use_hessian=True)

    if ok:
        test_quant_vs_fp32(net, net_fp, DEVICE)
        test_int8_weights(wrapped)

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("[ALL PASS] PTQ4ViT library is working in your repo.")
    else:
        print("[SOME FAILURES] See details above.")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
