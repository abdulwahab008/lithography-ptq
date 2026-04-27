# Checkpoints (not committed)

Place trained weights here so the default CLI paths work:

| File | Used by |
|------|---------|
| `best_convit_translator.pth` | `convit/` scripts (ConViT lithography model) |
| `best_pvt_translator.pth` | `pvt/` scripts (PVT lithography model) |

`.pth` files are **gitignored** (often ~900 MB each). Upload them to [Hugging Face Hub](https://huggingface.co/), Google Drive, or an internal file share, then download into this folder.

After copying weights, typical first commands from the repo root:

```bash
cd convit && python inference.py
cd convit && python apply_ptq.py --bit 8
```
