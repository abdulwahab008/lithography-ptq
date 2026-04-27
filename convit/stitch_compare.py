"""
Stitch the two comparison grids (MinMax + PTQ4ViT) into a single labelled PNG
so every output (Target / FP32 / W8A8 / W6A6 / W4A4 × both quant schemes) is
visible at once.
"""

import os

from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_RES = os.path.join(_ROOT, "results")

TOP = os.path.join(_RES, "compare_fp_w8_w6_w4.png")
BOT = os.path.join(_RES, "compare_fp_w8_w6_w4_ptq4vit.png")
OUT = os.path.join(_RES, "compare_fp_w8_w6_w4_all.png")

BANNER_H = 60


def banner(width, text):
    img = Image.new("RGB", (width, BANNER_H), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        ((width - tw) // 2, (BANNER_H - th) // 2 - bbox[1]),
        text,
        fill=(240, 240, 240),
        font=font,
    )
    return img


def main():
    os.makedirs(_RES, exist_ok=True)
    for path, label in ((TOP, "MinMax grid"), (BOT, "PTQ4ViT grid")):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {label}: {path}\n"
                "Run from repo root:\n"
                "  python convit/compare_bits_visual.py --mode minmax\n"
                "  python convit/compare_bits_visual.py --mode ptq4vit"
            )
    top = Image.open(TOP).convert("RGB")
    bot = Image.open(BOT).convert("RGB")
    if top.width != bot.width:
        scale = top.width / bot.width
        bot = bot.resize((top.width, int(bot.height * scale)))
    w = top.width
    b1 = banner(w, "MinMax PTQ  (apply_ptq.py)")
    b2 = banner(w, "PTQ4ViT Hessian  (apply_ptq4vit.py)")
    out = Image.new("RGB", (w, b1.height + top.height + b2.height + bot.height), (255, 255, 255))
    y = 0
    out.paste(b1, (0, y)); y += b1.height
    out.paste(top, (0, y)); y += top.height
    out.paste(b2, (0, y)); y += b2.height
    out.paste(bot, (0, y))
    out.save(OUT)
    print(f"Saved combined grid: {OUT}  ({out.width}x{out.height})")


if __name__ == "__main__":
    main()
