"""Our contestants: a ridge head on one frozen backbone's features, trained on the pool's train split only.

  uv run python -m aesbench.train siglip2-base            # -> preds/ours-<backbone>-ridge{,-noava,-lodo}.csv (+ .json)

Each dataset's scores are z-scored with its own train mean and std before pooling, so the head never has to reconcile
0-1 crowd scores with 1-10 photo-club scores (Spearman within a dataset does not care about that shift and scale).
Photos are weighted so each dataset counts equally. alpha is chosen on the val split (mean Spearman over the datasets
trained on); test is predicted once, after that choice.

Three contestants per backbone:
  ridge        trained on all four pool datasets
  ridge-noava  the same without AVA
  ridge-lodo   leave-one-dataset-out: each pool dataset's test photos are scored by a head that never saw that dataset
               (EVA and PARA by the all-pool head). This is the "does it transfer" row.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from aesbench.features import open_feats
from aesbench.manifest import OUT as MANIFEST
from aesbench.run import PREDS
from aesbench.score import srcc

POOL = ["ava", "tad66k", "flickr_aes", "aadb"]
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def fit(X: np.ndarray, rows: list[dict], train_on: list[str]) -> tuple:
    """-> (predict(X) -> scores, chosen alpha, val Spearman per dataset)"""
    ds = np.array([r["dataset"] for r in rows])
    split = np.array([r["split"] for r in rows])
    y = np.array([float(r["score"]) for r in rows])
    tr = np.isin(ds, train_on) & (split == "train")
    z, w = np.zeros(len(y)), np.zeros(len(y))
    for d in train_on:
        m = tr & (ds == d)
        z[m] = (y[m] - y[m].mean()) / y[m].std()
        w[m] = 1.0 / m.sum()                                           # each dataset weighs the same
    Xt = X[tr].astype(np.float64)
    mean, std = Xt.mean(0), max(float(Xt.std()), 1e-6)
    Xs, wt = (Xt - mean) / std, w[tr] * tr.sum() / len(train_on)
    A, b = (Xs * wt[:, None]).T @ Xs, (Xs * wt[:, None]).T @ z[tr]
    predict = lambda coef: (lambda Q: ((Q.astype(np.float64) - mean) / std) @ coef)
    best = None
    for alpha in ALPHAS:
        coef = np.linalg.solve(A + alpha * np.eye(A.shape[0]), b)
        val = {d: srcc(predict(coef)(X[(ds == d) & (split == "val")]), y[(ds == d) & (split == "val")]) for d in train_on}
        if best is None or np.mean(list(val.values())) > np.mean(list(best[2].values())):
            best = (predict(coef), alpha, val)
    return best


def write(name: str, rows: list[dict], test: np.ndarray, pred: np.ndarray, meta: dict) -> None:
    with open(PREDS / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "id", "pred"])
        w.writerows([rows[i]["dataset"], rows[i]["id"], f"{pred[j]:.6f}"] for j, i in enumerate(test))
    (PREDS / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(name, json.dumps(meta))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone")
    a = ap.parse_args()
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    X = open_feats(a.backbone, len(rows))
    test = np.flatnonzero([r["split"] == "test" for r in rows])
    Xtest, tds = X[test], np.array([rows[i]["dataset"] for i in test])
    PREDS.mkdir(exist_ok=True)

    full, alpha, val = fit(X, rows, POOL)
    write(f"ours-{a.backbone}-ridge", rows, test, full(Xtest), dict(trained_on=POOL, alpha=alpha, val_srcc=val))
    noava = [d for d in POOL if d != "ava"]
    f2, alpha, val = fit(X, rows, noava)
    write(f"ours-{a.backbone}-ridge-noava", rows, test, f2(Xtest), dict(trained_on=noava, alpha=alpha, val_srcc=val))
    pred, alphas = full(Xtest), {}
    for d in POOL:
        fd, alphas[d], _ = fit(X, rows, [e for e in POOL if e != d])
        pred[tds == d] = fd(Xtest[tds == d])
    write(f"ours-{a.backbone}-ridge-lodo", rows, test, pred,
          dict(trained_on=[], alpha=alphas, note="each pool test set scored by a head trained without that dataset; "
                                                 "EVA and PARA by the all-pool head (which saw AVA photos, so EVA is not unseen)",
               seen_override=["eva"]))


if __name__ == "__main__":
    main()
