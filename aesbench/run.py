"""Let one contestant sit the exam: score every split=test image of the manifest with a pyiqa metric.

  uv run python -m aesbench.run topiq_iaa [--task aesthetic] [--limit 50] [--device mps]

Writes ROOT/bench/preds/<name>.csv (dataset,id,pred), appending, so a stopped run resumes where it was. A photo that
cannot be read or scored gets an empty pred and is counted against the contestant's coverage by score.py.
Every contestant sees the same pixels: RGB, longest side at most MAX_SIDE, then its own preprocessing.
"""
from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aesbench.manifest import OUT as MANIFEST, ROOT

PREDS = ROOT / "bench" / "preds"
MAX_SIDE = 1024      # ponytail: one size for all; SPAQ's 4000 px originals are scored downsized. Raise it for a quality-only bench.
Image.MAX_IMAGE_PIXELS = None


def test_rows(limit: int | None = None) -> list[dict]:
    rows, per = [], {}
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test" and (limit is None or per.get(r["dataset"], 0) < limit):
                per[r["dataset"]] = per.get(r["dataset"], 0) + 1
                rows.append(r)
    return rows


def load(path: str) -> torch.Tensor | None:
    try:
        im = Image.open(ROOT / path).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div(255)[None]
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metric")
    ap.add_argument("--task", help="for metrics with several heads (qalign, qrealign): aesthetic")
    ap.add_argument("--name", help="output name; default metric[-task]")
    ap.add_argument("--limit", type=int, help="first N test images per dataset (smoke test / timing)")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    import pyiqa

    name = a.name or (f"{a.metric}-{a.task}" if a.task else a.metric)
    out = PREDS / f"{name}.csv"
    PREDS.mkdir(exist_ok=True)
    done = {(r["dataset"], r["id"]) for r in csv.DictReader(open(out))} if out.exists() else set()
    rows = [r for r in test_rows(a.limit) if (r["dataset"], r["id"]) not in done]
    metric = pyiqa.create_metric(a.metric, device=a.device)
    kw = {"task_": a.task} if a.task else {}
    t0, failed = time.time(), 0
    with open(out, "a", newline="") as f, ThreadPoolExecutor(8) as pool, torch.inference_mode():
        w = csv.writer(f)
        if not done:
            w.writerow(["dataset", "id", "pred"])
        for i, (r, x) in enumerate(zip(rows, pool.map(load, (r["path"] for r in rows)))):
            try:
                pred = float(metric(x.to(a.device), **kw).flatten()[0])
            except Exception as e:
                pred, failed = "", failed + 1
                if failed <= 3:
                    print("failed:", r["dataset"], r["id"], repr(e)[:200], flush=True)
            w.writerow([r["dataset"], r["id"], pred])
            if i % 500 == 499:
                f.flush()
                print(f"{name}: {i + 1}/{len(rows)}  {(i + 1) / (time.time() - t0):.1f} img/s", flush=True)
    print(f"{name}: {len(rows)} scored, {failed} failed, {len(rows) / max(time.time() - t0, 1e-9):.1f} img/s -> {out}")


if __name__ == "__main__":
    main()
