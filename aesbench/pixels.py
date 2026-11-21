"""Decode every manifest photo once into a 224 x 224 RGB cache, so backbones and fine-tuning never touch the JPEGs again.

  uv run python -m aesbench.pixels            # -> ROOT/bench/pixels224.u8 (N x 224 x 224 x 3, manifest row order); resumable

The whole frame is pressed into the square (no crop: composition at the edges stays in view; aspect ratio does not).
ponytail: one framing for every backbone. If a backbone needs its own (crop, 384 px), add a second cache, not a flag here.
An unreadable photo stays all zeros and is listed in pixels224.bad.
"""
from __future__ import annotations

import csv
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from aesbench.manifest import OUT as MANIFEST, ROOT

SIZE = 224
CACHE = ROOT / "bench" / "pixels224.u8"
DONE = ROOT / "bench" / "pixels224.done"        # number of leading rows written
CHUNK = 2048
Image.MAX_IMAGE_PIXELS = None


def paths() -> list[str]:
    with open(MANIFEST, newline="") as f:
        return [r["path"] for r in csv.DictReader(f)]


def open_cache(n: int, mode: str = "r") -> np.memmap:
    return np.memmap(CACHE, np.uint8, mode, shape=(n, SIZE, SIZE, 3))


def decode(path: str) -> np.ndarray | None:
    try:
        im = Image.open(ROOT / path)
        im.draft("RGB", (SIZE * 2, SIZE * 2))                       # JPEG: decode at reduced size, several times faster
        return np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC))
    except Exception:
        return None


def main() -> None:
    p = paths()
    start = int(DONE.read_text()) if DONE.exists() else 0
    out = open_cache(len(p), "r+" if CACHE.exists() else "w+")
    t0 = time.time()
    with Pool(8) as pool, open(ROOT / "bench" / "pixels224.bad", "a") as bad:
        for i in range(start, len(p), CHUNK):
            for j, x in enumerate(pool.map(decode, p[i:i + CHUNK], chunksize=32)):
                if x is None:
                    bad.write(p[i + j] + "\n")
                else:
                    out[i + j] = x
            out.flush()
            DONE.write_text(str(min(i + CHUNK, len(p))))
            print(f"{min(i + CHUNK, len(p))}/{len(p)}  {(min(i + CHUNK, len(p)) - start) / (time.time() - t0):.0f} img/s", flush=True)
    print("PIXELS DONE")


if __name__ == "__main__":
    main()
