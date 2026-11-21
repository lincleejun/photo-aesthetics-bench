"""A second contestant per backbone: a shared MLP trunk with one output per training dataset.

  uv run python -m aesbench.train_mlp siglip2-base        # -> preds/ours-<backbone>-mlp.csv (+ .json)

Why: one linear head for all four datasets scored well below a head per dataset (AVA val 0.61 against 0.70 on
SigLIP2): the datasets disagree about what a good photo is. Here the trunk is shared and each dataset keeps its own
output, so they share what agrees and differ where they must. A photo from a training dataset is scored by that
dataset's output; a photo from anywhere else by the mean of the outputs (each is a z-score, so they are comparable).
Epoch chosen on val (mean Spearman over the four); test predicted once.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from torch import nn

from aesbench.features import open_feats
from aesbench.manifest import OUT as MANIFEST
from aesbench.score import srcc
from aesbench.train import POOL, write

EPOCHS, BATCH, SEED = 30, 2048, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    X = open_feats(a.backbone, len(rows)).astype(np.float32)
    ds = np.array([r["dataset"] for r in rows])
    split = np.array([r["split"] for r in rows])
    y = np.array([float(r["score"]) for r in rows], np.float32)
    head = np.array([POOL.index(d) if d in POOL else -1 for d in ds])
    tr = (head >= 0) & (split == "train")
    for d in POOL:                                                       # z-score per dataset, train statistics
        m = tr & (ds == d)
        y[ds == d] = (y[ds == d] - y[m].mean()) / y[m].std()
    mean, std = X[tr].mean(0), X[tr].std()
    T = lambda v: torch.from_numpy((v - mean) / std)
    Xt, yt, ht = T(X[tr]), torch.from_numpy(y[tr]), torch.from_numpy(head[tr])
    weight = torch.tensor([1.0 / (ht == k).sum() for k in range(len(POOL))])[ht]     # each dataset drawn equally often
    net = nn.Sequential(nn.Linear(X.shape[1], 1024), nn.GELU(), nn.Dropout(0.2), nn.Linear(1024, 256), nn.GELU(),
                        nn.Linear(256, len(POOL)))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.05)
    steps = EPOCHS * (len(yt) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    val = {d: (T(X[(ds == d) & (split == "val")]), y[(ds == d) & (split == "val")]) for d in POOL}
    best = (-1.0, None, None, 0)
    for epoch in range(EPOCHS):
        net.train()
        for _ in range(len(yt) // BATCH):
            i = torch.multinomial(weight, BATCH, replacement=True)
            loss = nn.functional.mse_loss(net(Xt[i]).gather(1, ht[i, None])[:, 0], yt[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.inference_mode():
            s = {d: srcc(net(v)[:, k].numpy(), g) for k, (d, (v, g)) in enumerate(val.items())}
        if np.mean(list(s.values())) > best[0]:
            best = (float(np.mean(list(s.values()))), {k: v.clone() for k, v in net.state_dict().items()}, s, epoch)
        print(epoch, {d: round(v, 4) for d, v in s.items()}, flush=True)
    net.load_state_dict(best[1])
    net.eval()
    test = np.flatnonzero(split == "test")
    with torch.inference_mode():
        out = net(T(X[test])).numpy()
    h = head[test]
    pred = np.where(h >= 0, out[np.arange(len(test)), np.maximum(h, 0)], out.mean(1))
    write(f"ours-{a.backbone}-mlp", rows, test, pred, dict(trained_on=POOL, epoch=best[3], val_srcc=best[2]))
    torch.save({"state": best[1], "mean": mean, "std": float(std), "heads": POOL},
               MANIFEST.parent / "models" / f"{a.backbone}-mlp.pt") if (MANIFEST.parent / "models").exists() else None


if __name__ == "__main__":
    main()
