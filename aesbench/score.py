"""Mark the exam: Spearman rank correlation of each contestant's scores with the human mean, per dataset, on split=test.

  uv run python -m aesbench.score            # reads ROOT/bench/preds/*.csv -> leaderboard.md (aggregates only)

Never pooled across datasets: scales and rater populations differ. A photo the contestant did not score counts as
the lowest rank (it cannot gain by skipping). 95% intervals: bootstrap over photos, 1000 resamples, seed 0.
The headline is the mean over the photo-aesthetics test sets the contestant did NOT train on (TRAINED_ON below);
cells it trained on are shown with a dagger and left out of that mean.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from aesbench.manifest import OUT as MANIFEST
from aesbench.run import PREDS

PHOTO = ["para", "eva", "tad66k", "aadb", "flickr_aes", "ava"]            # photo aesthetics: the exam proper
PROBE = ["koniq10k", "spaq", "paq2piq", "baid", "lapis"]                  # technical quality, paintings: side papers
SLICES = [("ava", "animal"), ("tad66k", "bird"), ("para", "animal")]
BOARD = Path(__file__).resolve().parents[1] / "leaderboard.md"

# What each contestant was trained on, from its own weight file name / model card. EVA's photos are AVA photos, so an
# AVA-trained model has seen them (with AVA's labels). "?" = not established yet: shown, but kept out of the headline.
AVA_ = {"ava", "eva"}
TRAINED_ON = {
    "topiq_iaa": AVA_, "topiq_iaa_res50": AVA_, "nima": AVA_, "nima-vgg16-ava": AVA_, "musiq-ava": AVA_,
    "laion_aes": AVA_,                                    # sac+logos+ava1-l14-linearMSE
    "clipiqa+": {"koniq10k"}, "topiq_nr": {"koniq10k"}, "musiq": {"koniq10k"},
    "qalign-aesthetic": None, "qrealign-mini-aesthetic": None,            # ponytail: read their cards, then fill in
}


def srcc(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def interval(pred: np.ndarray, gold: np.ndarray, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(gold), (n, len(gold)))
    return tuple(np.percentile([srcc(pred[i], gold[i]) for i in idx], [2.5, 97.5]))


def main() -> None:
    gold, tags = defaultdict(dict), {}
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                gold[r["dataset"]][r["id"]] = float(r["score"])
                tags[r["dataset"], r["id"]] = r["tags"]
    board = []
    for p in sorted(PREDS.glob("*.csv")):
        pred = defaultdict(dict)
        for r in csv.DictReader(open(p)):
            if r["pred"] not in ("", "nan"):
                pred[r["dataset"]][r["id"]] = float(r["pred"])
        seen = TRAINED_ON.get(p.stem, None)
        row = {"name": p.stem, "cells": {}, "known": seen is not None}
        for d in PHOTO + PROBE + SLICES:
            ds, tag = d if isinstance(d, tuple) else (d, None)
            ids = [k for k in gold[ds] if tag is None or tag in tags[ds, k].split("|")]
            have = [k for k in ids if k in pred[ds]]
            if len(have) < 0.5 * len(ids) or len(ids) < 30:
                continue                                                   # not sat (or still running)
            floor = min(pred[ds].values()) - 1.0
            x = np.array([pred[ds].get(k, floor) for k in ids])
            y = np.array([gold[ds][k] for k in ids])
            lo, hi = interval(x, y)
            row["cells"][d] = dict(srcc=srcc(x, y), lo=lo, hi=hi, n=len(ids), scored=len(have),
                                   seen=bool(seen) and ds in seen)
        unseen = [c["srcc"] for d, c in row["cells"].items() if d in PHOTO and not c["seen"]]
        row["headline"] = float(np.mean(unseen)) if unseen and row["known"] else None
        row["n_unseen"] = len(unseen)
        board.append(row)
    board.sort(key=lambda r: (r["headline"] is None, -(r["headline"] or 0)))

    def cell(r, d):
        c = r["cells"].get(d)
        return "—" if c is None else f"{c['srcc']:.3f}{'†' if c['seen'] else ''}"

    n = {d: len(gold[d]) for d in PHOTO + PROBE}
    out = ["# Leaderboard", "",
           "Spearman rank correlation with the human mean score on held-out photos; higher is better. † = the model was",
           "trained on that dataset (or on its photos), so the cell is not a held-out result and is left out of the mean.",
           "", "## Photo aesthetics", "",
           "| Model | Mean, unseen sets | " + " | ".join(f"{d} ({n[d]:,})" for d in PHOTO) + " |",
           "|---|---|" + "---|" * len(PHOTO)]
    for r in board:
        head = "?" if r["headline"] is None else f"**{r['headline']:.3f}** ({r['n_unseen']})"
        out.append(f"| {r['name']} | {head} | " + " | ".join(cell(r, d) for d in PHOTO) + " |")
    out += ["", "## Side papers: technical quality (koniq10k, spaq, paq2piq), paintings (baid, lapis), animal slices", "",
            "| Model | " + " | ".join(PROBE + [f"{d}:{t}" for d, t in SLICES]) + " |", "|---|" + "---|" * len(PROBE + SLICES)]
    for r in board:
        out.append(f"| {r['name']} | " + " | ".join(cell(r, d) for d in PROBE + SLICES) + " |")
    out += ["", "## 95% intervals (bootstrap over photos)", "", "| Model | " + " | ".join(PHOTO) + " |", "|---|" + "---|" * len(PHOTO)]
    for r in board:
        out.append(f"| {r['name']} | " + " | ".join(
            "—" if d not in r["cells"] else f"{r['cells'][d]['lo']:.3f}–{r['cells'][d]['hi']:.3f}" for d in PHOTO) + " |")
    BOARD.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    a, b = np.array([1.0, 2, 3, 4, 5]), np.array([2.0, 1, 4, 3, 5])
    assert abs(srcc(a, a) - 1) < 1e-12 and abs(srcc(a, -a) + 1) < 1e-12 and abs(srcc(a, b) - 0.8) < 1e-12
    main()
