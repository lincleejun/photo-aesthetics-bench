"""How well do people agree with people? The ceiling a model is marked against.

  uv run python -m aesbench.ceiling          # -> ROOT/bench/ceiling.json (aggregates only), read by aesbench.score

For each test set with per-vote data (AVA and PARA: vote histograms; EVA and FLICKR-AES: one row per vote), the votes
of every photo are split at random into two halves, 20 times:
  half    = Spearman between the two half-panel means (a second panel half the size against the first);
  ceiling = sqrt(2h / (1 + h)) (Spearman-Brown): what a scorer that knew the true mean would get against the
            published full-panel mean. A model above this is fitting the panel's noise.
TAD66K and AADB publish only the mean, so they have no row.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from aesbench.manifest import OUT as MANIFEST, ROOT
from aesbench.score import srcc

REPS = 20


def votes() -> dict[str, dict[str, list[float]]]:
    v = defaultdict(lambda: defaultdict(list))
    for line in open(ROOT / "AVA/AVA_dataset/AVA.txt"):
        p = line.split()
        v["ava"][p[1]] = [s for s, n in enumerate(map(int, p[2:12]), 1) for _ in range(n)]
    eva = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "EVA/data/votes_filtered.csv"), delimiter="="):
        eva[r["image_id"]].append(float(r["score"]))
    v["eva"] = eva
    for r in csv.DictReader(open(ROOT / "PARA/annotation/PARA-GiaaTest.csv")):
        v["para"][r["imageName"]] = [float(c[15:]) for c, n in r.items()
                                     if c.startswith("aestheticScore_") and c[15].isdigit() for _ in range(int(n))]
    for r in csv.DictReader(open(ROOT / "FLICKR-AES/iccv_17_aesthetics_dataset/FLICKR-AES_image_labeled_by_each_worker.csv"),
                            skipinitialspace=True):
        v["flickr_aes"][r["imagePair"].strip()].append(float(r["score"]))
    return v


def main() -> None:
    v, rng, out = votes(), np.random.default_rng(0), {}
    test = defaultdict(list)
    for r in csv.DictReader(open(MANIFEST, newline="")):
        if r["split"] == "test" and r["dataset"] in v:
            test[r["dataset"]].append(r["id"])
    for d, ids in test.items():
        panels = [np.array(v[d][k]) for k in ids if len(v[d].get(k, ())) >= 2]
        assert len(panels) > 0.99 * len(ids), (d, len(panels), len(ids))
        hs = []
        for _ in range(REPS):
            a, b = zip(*((p[: len(p) // 2].mean(), p[len(p) // 2:].mean()) for p in map(rng.permutation, panels)))
            hs.append(srcc(np.array(a), np.array(b)))
        h = float(np.mean(hs))
        out[d] = dict(half=h, ceiling=float(np.sqrt(2 * h / (1 + h))), n=len(panels),
                      votes=float(np.median([len(p) for p in panels])))
        print(d, out[d])
    (MANIFEST.parent / "ceiling.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
