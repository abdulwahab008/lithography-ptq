# Checkpoints (not committed)

Place trained weights here so the default CLI paths work:

| File | Used by |
|------|---------|
| `best_convit_translator.pth` | `convit/` scripts (ConViT lithography model) |
| `best_pvt_translator.pth` | `pvt/` scripts (PVT lithography model) |

`.pth` files are **gitignored** (often ~900 MB each). Upload them to [Hugging Face Hub](https://huggingface.co/), Google Drive, or an internal file share, then download into this folder.

Primary project Drive folder (all PTQ work and checkpoints):

- https://drive.google.com/drive/folders/1mN__31JBIWaAt1s9D7qNTqRfUZDkE4kC?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

After copying weights, typical first commands from the repo root:

```bash
cd convit && python inference.py
cd convit && python apply_ptq.py --bit 8
```
