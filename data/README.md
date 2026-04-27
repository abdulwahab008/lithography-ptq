# Data

- **`sample_targets/`** — A few lithography **target** PNGs (224×224 RGB) for calibration and demos.
- **`sample_pixelilt/`** — Matching **FP32 PixelILT** outputs (`*_pixelILT_convit.png`) used as the reference column in `convit/compare_bits_visual.py`.

For full experiments, point `--target-dir` / `--pixelilt-dir` at your own directories (same naming convention for PixelILT files).
