# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "pillow", "numpy"]
# ///
"""Score every photo under a folder with a released aesthetics model. One file, runs anywhere uv does:

  uv run scan.py siglip2-base-mlp.pt ~/Pictures/trip [--out scores.csv] [--device cpu|mps|cuda]

Works with both release files: <backbone>-mlp.pt (head on the stock SigLIP2 base, downloaded from Hugging Face on
first use) and <backbone>-ft.pt (the fine-tuned backbone inside). Output: path,score sorted best first. The score is
the mean of the model's four outputs, each a z-score against one training set: 0 = an average photo there, +1 = one
standard deviation better. Use it to rank; it is not a 1-10 mark.
Photos are seen as the bench saw them: whole frame pressed into 224 x 224 (upright, per EXIF).
ponytail: JPEG/PNG/WebP/TIFF only. RAW and HEIC need a decoder (rawpy, pillow-heif): add when the shoots need it.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def load(path: Path) -> np.ndarray | None:
    try:
        im = Image.open(path)
        im.draft("RGB", (448, 448))
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB").resize((224, 224), Image.BICUBIC))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("folder")
    ap.add_argument("--out", default="scores.csv")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from transformers import AutoModel
    ck = torch.load(a.model, weights_only=False, map_location="cpu")     # our own file: it carries a numpy array
    net = AutoModel.from_pretrained("google/siglip2-base-patch16-224").vision_model
    mlp = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.Dropout(0.2), nn.Linear(1024, 256), nn.GELU(), nn.Linear(256, len(ck["heads"])))
    if "net" in ck:                                                      # fine-tuned: backbone weights inside
        net.load_state_dict(ck["net"], strict=False)
    mlp.load_state_dict(ck.get("mlp") or ck["state"])
    net.to(a.device).eval(), mlp.to(a.device).eval()
    mean, std = torch.from_numpy(ck["mean"]).to(a.device), float(ck["std"])
    files = sorted(p for p in Path(a.folder).expanduser().rglob("*") if p.suffix.lower() in EXT and not p.name.startswith("."))
    if not files:
        sys.exit(f"no photos ({', '.join(sorted(EXT))}) under {a.folder}")
    rows, bad = [], 0
    with torch.inference_mode():
        for i in range(0, len(files), 64):
            got = [(p, x) for p in files[i:i + 64] if (x := load(p)) is not None]
            bad += len(files[i:i + 64]) - len(got)
            if got:
                x = torch.from_numpy(np.stack([x for _, x in got])).to(a.device).permute(0, 3, 1, 2).float().div(255)
                s = mlp((net(pixel_values=(x - 0.5) / 0.5).pooler_output - mean) / std).mean(1).cpu().numpy()
                rows += [(str(p), float(v)) for (p, _), v in zip(got, s)]
            print(f"\r{min(i + 64, len(files))}/{len(files)}", end="", file=sys.stderr)
    rows.sort(key=lambda r: -r[1])
    with open(a.out, "w", newline="") as f:
        csv.writer(f).writerows([("path", "score"), *((p, f"{v:.4f}") for p, v in rows)])
    print(f"\n{len(rows)} scored, {bad} unreadable -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
