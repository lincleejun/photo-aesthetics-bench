"""One frozen backbone over the pixel cache -> one vector per manifest photo.

  uv run python -m aesbench.features siglip2-base [--limit 2000]   # -> ROOT/bench/features/<name>.f16 (N x D float16); resumable

Every backbone sees the same 224 x 224 whole-frame pixels (aesbench.pixels) with its own normalisation.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from aesbench.manifest import ROOT
from aesbench.pixels import DONE as PIXELS_DONE, open_cache, paths

FEATS = ROOT / "bench" / "features"
CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def siglip2(dev, name="google/siglip2-base-patch16-224"):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(name).vision_model.to(dev).eval()
    return HALF_NORM, lambda x: m(pixel_values=x).pooler_output


def clip_l14(dev):
    import open_clip
    m = open_clip.create_model("ViT-L-14", pretrained="openai").to(dev).eval()
    return CLIP_NORM, m.encode_image


def dinov2(dev):
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()

    def f(x):
        h = m(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)               # class token + mean patch token
    return IMAGENET_NORM, f


def bioclip(dev, block=15):
    """BioCLIP 2.5's block-15 mean patch token: the features the bird engine's aesthetic head already reads."""
    import open_clip
    m = open_clip.create_model("hf-hub:imageomics/bioclip-2.5-vith14").visual.to(dev).eval()
    got = {}
    m.transformer.resblocks[block].register_forward_hook(lambda _m, _i, o: got.__setitem__("t", o))

    def f(x):
        m(x)
        t = got["t"]
        t = t if t.shape[0] == x.shape[0] else t.transpose(0, 1)       # some open_clip versions are sequence-first
        return t[:, 1:].mean(1)
    return CLIP_NORM, f


BACKBONES = {"siglip2-base": siglip2, "clip-l14": clip_l14, "dinov2-base": dinov2, "bioclip25-b15": bioclip,
             "siglip2-so400m": lambda dev: siglip2(dev, "google/siglip2-so400m-patch14-224")}


def open_feats(name: str, n: int | None = None) -> np.ndarray:
    n = n or len(paths())
    raw = np.memmap(FEATS / f"{name}.f16", np.float16, "r")
    return raw.reshape(n, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone", choices=BACKBONES)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    n = len(paths())
    ready = int(PIXELS_DONE.read_text())
    stop = min(a.limit or n, ready)
    px = open_cache(n)
    (mean, std), encode = BACKBONES[a.backbone](a.device)
    mean, std = (torch.tensor(v, device=a.device).view(1, 3, 1, 1) for v in (mean, std))
    FEATS.mkdir(exist_ok=True)
    out_p, done_p = FEATS / f"{a.backbone}.f16", FEATS / f"{a.backbone}.done"
    start = int(done_p.read_text()) if done_p.exists() else 0
    out, t0 = None, time.time()
    with torch.inference_mode():
        for i in range(start, stop, a.batch):
            x = torch.from_numpy(np.ascontiguousarray(px[i:min(i + a.batch, stop)])).to(a.device)
            y = encode((x.permute(0, 3, 1, 2).float().div(255) - mean) / std).float().cpu().numpy()
            if out is None:
                out = np.memmap(out_p, np.float16, "r+" if out_p.exists() else "w+", shape=(n, y.shape[1]))
            out[i:i + len(y)] = y
            if (i // a.batch) % 40 == 39 or i + a.batch >= stop:
                out.flush()
                done_p.write_text(str(i + len(y)))
                print(f"{a.backbone}: {i + len(y)}/{stop}  {(i + len(y) - start) / (time.time() - t0):.0f} img/s", flush=True)
    print(f"{a.backbone} FEATURES DONE to row {stop} of {n}")


if __name__ == "__main__":
    main()
