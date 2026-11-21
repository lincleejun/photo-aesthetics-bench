# photo-aesthetics-bench

A held-out exam for models that give a photo an aesthetic score: which scorer ranks photos most like people do, on
photos it has never seen?

- **Questions**: the test splits of eleven public datasets (six of photo aesthetics, three of technical quality, two of
  paintings), about 61,000 photos. Official test splits where a dataset ships one; otherwise a split derived from a
  private salt.
- **Answers**: each dataset's human mean score, on its own scale.
- **Marking**: Spearman rank correlation per dataset, never pooled. Results: [leaderboard.md](leaderboard.md).

**No data is in this repository**: no photos, no scores, no manifest, no per-photo predictions, and not the salt that
decides which photos are held out. The datasets are research-only or carry no licence; download them yourself.

```
source env.sh                                  # caches and the virtualenv stay on the data disk
uv run python -m aesbench.manifest             # freeze the manifest (needs the datasets under $AESBENCH_ROOT)
uv run python -m aesbench.run topiq_iaa        # one contestant sits the exam (any pyiqa metric)
uv run python -m aesbench.score                # mark everyone -> leaderboard.md
```

| File | What it does |
|---|---|
| `aesbench/manifest.py` | One row per photo with a file and a score: dataset, split (train / val / test / excluded), role |
| `aesbench/run.py` | Scores every test photo with one model; resumable |
| `aesbench/score.py` | Spearman per dataset with bootstrap intervals; marks which sets a model trained on |
| `queue.sh` | Runs the list of contestants one after another |
