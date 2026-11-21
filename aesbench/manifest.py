"""Build the frozen manifest of the aesthetics bench: one row per image that has both a file and a human score.

  uv run python -m aesbench.manifest          # -> ROOT/bench/manifest.csv + its sha256

Columns: dataset, id, path (relative to ROOT), score (the dataset's own scale, never rescaled), votes, split, role, tags.
  split  train | val | test | excluded     (excluded = would leak into a test set; nobody trains or scores on it)
  role   pool     photo aesthetics; trainers use role=pool & split=train, select on val, report on test
         holdout  photo aesthetics that no model of ours ever trains on (EVA, PARA); only split=test is scored
         probe    measures something else (technical quality, paintings); scored, never trained on
Official splits are used where the dataset ships one; otherwise a split derived from md5(salt/dataset/id), so it never
moves. The salt lives next to the manifest and is never committed: the code alone does not reveal which images are held out.
Nothing under ROOT (images, scores, manifest, salt, per-image predictions) goes into git.
Duplicates handled here are the ones identifiable by id (EVA inside AVA, AVA inside PaQ-2-PiQ).
ponytail: no pixel-level duplicate search across datasets; do it from embeddings once features exist.
"""
from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Volumes/Media/traning")
OUT = ROOT / "bench" / "manifest.csv"
SALT = OUT.parent / "salt.txt"
COLS = ["dataset", "id", "path", "score", "votes", "split", "role", "tags"]


def bucket(dataset: str, key: str, n: int) -> int:
    return int(hashlib.md5(f"{SALT.read_text().strip()}/{dataset}/{key}".encode()).hexdigest(), 16) % n


def carve(dataset: str, key: str, official: str | None = None) -> str:
    """official train -> train/val (5% val); no official split -> 10% test, then 5% of the rest val."""
    if official is None and bucket(dataset, key + "#test", 10) == 0:
        return "test"
    return "val" if bucket(dataset, key + "#val", 20) == 0 else "train"


def lines(p: Path) -> list[list[str]]:
    return [l.split() for l in open(p) if l.strip()]


def table(p: Path, **kw) -> list[dict]:
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, **kw))


def eva() -> list[dict]:
    votes = defaultdict(list)
    for r in table(ROOT / "EVA/data/votes_filtered.csv", delimiter="="):
        votes[r["image_id"]].append(float(r["score"]))
    return [dict(dataset="eva", id=k, path=f"EVA/images/EVA_together/{k}.jpg", score=sum(v) / len(v), votes=len(v),
                 split="test", role="holdout", tags="") for k, v in votes.items()]


def ava(eva_ids: set[str]) -> list[dict]:
    d = ROOT / "AVA/AVA_dataset"
    lists = d / "aesthetics_image_lists"
    test = {l[0] for l in lines(lists / "generic_test.jpgl")}
    animal = {l[0] for l in lines(lists / "animal_test.jpgl")}
    rows = []
    for l in lines(d / "AVA.txt"):
        k, counts = l[1], [int(x) for x in l[2:12]]
        n = sum(counts)
        split = "test" if k in test else "excluded" if k in eva_ids else carve("ava", k, "train")
        rows.append(dict(dataset="ava", id=k, path=f"AVA/AVA_dataset/image/{k}.jpg",
                         score=sum((i + 1) * c for i, c in enumerate(counts)) / n, votes=n, split=split, role="pool",
                         tags="animal" if k in animal else ""))
    return rows


def tad66k() -> list[dict]:
    d = ROOT / "TAD66K/labels"
    bird = {r["image"] for s in ("train", "test") for r in table(d / "unmerge" / s / "bird.csv")
            } if (d / "unmerge/test/bird.csv").exists() else set()
    rows = []
    for s in ("train", "test"):
        for r in table(d / "merge" / f"{s}.csv"):
            k = r["image"]
            rows.append(dict(dataset="tad66k", id=k, path=f"TAD66K/{k}", score=float(r["score"]), votes="",
                             split="test" if s == "test" else carve("tad66k", k, "train"), role="pool",
                             tags="bird" if k in bird else ""))
    return rows


def aadb() -> list[dict]:
    d = ROOT / "AADB/AADB/imgListFiles_label"
    seen, rows = set(), []
    for name, split in (("TestNew", "test"), ("Validation", "val"), ("Train", "train")):   # test wins on overlap
        for k, s in lines(d / f"imgList{name}Regression_score.txt"):
            if k not in seen:
                seen.add(k)
                rows.append(dict(dataset="aadb", id=k, path=f"AADB/AADB/datasetImages_originalSize/{k}", score=float(s),
                                 votes=5, split=split, role="pool", tags=""))
    return rows


def flickr_aes() -> list[dict]:
    d = ROOT / "FLICKR-AES/iccv_17_aesthetics_dataset"
    n = Counter(r[" imagePair"].strip() for r in table(d / "FLICKR-AES_image_labeled_by_each_worker.csv"))
    return [dict(dataset="flickr_aes", id=k, path=f"FLICKR-AES/iccv_17_aesthetics_dataset/40K/{k}", score=float(s),
                 votes=n.get(k, ""), split=carve("flickr_aes", k), role="pool", tags="")
            for k, s in lines(d / "FLICKR-AES_image_score.txt")]


def para() -> list[dict]:
    rows = []
    for s in ("Train", "Test"):
        for r in table(ROOT / f"PARA/annotation/PARA-Giaa{s}.csv"):
            n = sum(int(v) for c, v in r.items() if c.startswith("aestheticScore_") and c[15].isdigit())
            rows.append(dict(dataset="para", id=r["imageName"], path=f"PARA/imgs/{r['sessionId']}/{r['imageName']}",
                             score=float(r["aestheticScore_mean"]), votes=n,
                             split="test" if s == "Test" else carve("para", r["imageName"], "train"), role="holdout",
                             tags=r["semantic"]))
    return rows


def koniq() -> list[dict]:
    split = {"training": "train", "validation": "val", "test": "test"}
    return [dict(dataset="koniq10k", id=r["image_name"], path=f"KonIQ-10k/koniq10k/1024x768/{r['image_name']}",
                 score=float(r["MOS"]), votes=int(float(r["c_total"])), split=split[r["set"]], role="probe", tags="")
            for r in table(ROOT / "KonIQ-10k/koniq10k/koniq10k_distributions_sets.csv")]


def spaq() -> list[dict]:
    import openpyxl
    d = ROOT / "SPAQ/SPAQ/Annotations"
    sheet = lambda n: list(openpyxl.load_workbook(d / n, read_only=True).active.iter_rows(min_row=2, values_only=True))
    animal = {r[0] for r in sheet("Scene category labels.xlsx") if r[0] and (r[1] or 0) >= 0.5}
    return [dict(dataset="spaq", id=r[0], path=f"SPAQ/SPAQ/TestImage/{r[0]}", score=float(r[1]), votes="",
                 split=carve("spaq", r[0]), role="probe", tags="animal" if r[0] in animal else "")
            for r in sheet("MOS and Image attribute scores.xlsx") if r[0]]


def paq2piq(ava_ids: set[str]) -> list[dict]:
    rows = []
    for r in table(ROOT / "PaQ-2-PiQ/FLIVE_Database/labels_image.csv"):
        k = r["name"]
        dup = k.split("/")[-1].startswith("AVA__") and k.split("AVA__")[1].split(".")[0] in ava_ids
        rows.append(dict(dataset="paq2piq", id=k, path=f"PaQ-2-PiQ/FLIVE_Database/database/{k}", score=float(r["mos"]),
                         votes="", split="excluded" if dup else carve("paq2piq", k), role="probe", tags="ava" if dup else ""))
    return rows


def baid() -> list[dict]:
    return [dict(dataset="baid", id=r["image"], path=f"BAID/BAID/images/{r['image']}", score=float(r["score"]), votes="",
                 split=s, role="probe", tags="painting")
            for s in ("train", "val", "test") for r in table(ROOT / f"BAID/official-repo/dataset/{s}_set.csv")]


def lapis() -> list[dict]:
    return [dict(dataset="lapis", id=r["image_id"], path=f"LAPIS/images/{r['path']}", score=float(r["mean_response"]),
                 votes=int(float(r["number of raters"])), split=s[:5].replace("valid", "val"), role="probe", tags="painting")
            for s in ("train", "validation", "test") for r in table(ROOT / f"LAPIS/giaa/{s}.csv")]


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    if not SALT.exists():
        SALT.write_text(os.urandom(16).hex() + "\n")
    e = eva()
    eva_ids = {r["id"] for r in e}
    a = ava(eva_ids)
    rows = e + a + tad66k() + aadb() + flickr_aes() + para() + koniq() + spaq() + paq2piq({r["id"] for r in a}) + baid() + lapis()
    present = defaultdict(set)                                          # one listdir per directory, not 540k stats
    kept, missing = [], Counter()
    for r in rows:
        d, name = os.path.split(r["path"])
        if d not in present:
            present[d] = set(os.listdir(ROOT / d)) if (ROOT / d).is_dir() else set()
        (kept.append(r) if name in present[d] else missing.update([r["dataset"]]))

    # the checks that fail if a split leaks
    keys = Counter((r["dataset"], r["id"]) for r in kept)
    assert max(keys.values()) == 1, [k for k, n in keys.items() if n > 1][:5]
    assert not {r["id"] for r in kept if r["dataset"] == "ava" and r["split"] in ("train", "val")} & eva_ids
    assert all(r["split"] in ("train", "val", "test", "excluded") for r in kept)

    kept.sort(key=lambda r: (r["dataset"], r["id"]))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows(kept)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT.parent / "manifest.sha256").write_text(f"{sha}  manifest.csv\n")

    count = Counter((r["dataset"], r["role"], r["split"]) for r in kept)
    print(f"{'dataset':11} {'role':8} {'train':>7} {'val':>6} {'test':>6} {'excl':>6}  no-file")
    for d, role in sorted({(k[0], k[1]) for k in count}, key=lambda x: ("pool", "holdout", "probe").index(x[1])):
        print(f"{d:11} {role:8} " + " ".join(f"{count[(d, role, s)]:>{n}}" for s, n in
                                              (("train", 7), ("val", 6), ("test", 6), ("excluded", 6))) + f"  {missing[d]}")
    print(f"{len(kept)} rows -> {OUT}\nsha256 {sha}")


if __name__ == "__main__":
    main()
