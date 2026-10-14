#!/usr/bin/env python3
"""Phase 2 — group-aware re-split without leakage.

Builds connected components:
  * same Roboflow / filename base_key
  * pHash near-dup edges (Hamming <= --ham, val/test <-> train and val <-> test)
  * May/Jan burst sessions (same prefix before final index)

Assigns whole groups to train/valid/test (~80/10/10, seed 42, stratified by has-neg).
Val/test groups keep a single representative image; other members are dropped (not train).
Writes landmine_combined_v2 + split_manifest.csv + groups.csv + data.yaml + sha256.

Does not modify the original dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined")
DST = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")
AUDIT = ROOT / "paper_results" / "audit"
SPLIT2 = ROOT / "paper_results" / "split_v2"
SEED = 42


class UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.p:
            self.p[x] = x
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def load_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def may_jan_session(name: str) -> str | None:
    """May_May_Afternoon_0_lwir_3_lwir_12.jpg -> session key without trailing index."""
    m = re.match(
        r"^(May_[^/]*?|Jan_[^/]*?)_lwir_(\d+)_lwir_(\d+)\.jpg$",
        name,
        re.I,
    )
    if m:
        return f"{m.group(1)}_lwir_{m.group(2)}"
    m = re.match(r"^(May_.*|Jan_.*)_lwir_(\d+)\.jpg$", name, re.I)
    if m:
        return f"{m.group(1)}_lwir_{m.group(2)}"
    return None


def pick_representative(members: list[dict], prefer_neg: bool = False) -> dict:
    """Prefer non-Roboflow multi-export, then optional negative, then size/name."""
    def score(r: dict):
        is_rf = 1 if ".rf." in r["name"] else 0
        is_neg = 0
        if prefer_neg:
            is_neg = 0 if int(r["n_boxes"]) == 0 else 1
        return (is_rf, is_neg, -int(r["bytes"]), r["name"])

    return sorted(members, key=score)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--manifest", type=Path, default=AUDIT / "manifest_v1.csv")
    ap.add_argument("--near", type=Path, default=AUDIT / "near_dup_pairs.csv")
    ap.add_argument("--ham", type=int, default=4, help="Union query<->train if min_hamming <= this")
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--valid-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.dst.exists() and any(args.dst.iterdir()) and not args.force:
        print(f"Refusing to write into non-empty {args.dst} (use --force)")
        return 1

    rows = load_manifest(args.manifest)
    print(f"manifest rows: {len(rows)}")

    uf = UF()
    key_of: dict[str, str] = {}  # filename -> key used in UF
    for r in rows:
        # unique node id = path
        key_of[r["name"]] = r["path"]
        uf.find(r["path"])
        # base_key connects same source image exports
        uf.union(r["path"], f"base::{r['base_key']}")

    # May/Jan burst sessions
    n_burst = 0
    for r in rows:
        sess = may_jan_session(r["name"])
        if sess:
            uf.union(r["path"], f"sess::{sess}")
            n_burst += 1
    print(f"May/Jan burst memberships: {n_burst}")

    # pHash near-dup edges
    n_edges = 0
    if args.near.exists():
        with open(args.near, newline="", encoding="utf-8") as f:
            for e in csv.DictReader(f):
                d = int(e["min_hamming"])
                if d > args.ham:
                    continue
                # map names -> paths via manifest
                # we stored names; resolve
                # build later with name_to_path
                n_edges += 1
                e["_d"] = d
        # second pass with name map
        name_to_path = {r["name"]: r["path"] for r in rows}
        with open(args.near, newline="", encoding="utf-8") as f:
            for e in csv.DictReader(f):
                d = int(e["min_hamming"])
                if d > args.ham:
                    continue
                a = name_to_path.get(e["query_name"])
                b = name_to_path.get(e["train_name"])
                if a and b:
                    uf.union(a, b)
    print(f"pHash edges used (ham<= {args.ham}): {n_edges}")

    # Also union test<-> valid near dups from near file if query is test and train is valid?
    # near_dup_pairs only has query valid/test vs train. Good enough.

    # Collect groups
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[uf.find(r["path"])].append(r)
    print(f"connected components: {len(groups)}")

    # Stratify: prefer groups with at least one negative into val/test slots carefully
    rng = random.Random(SEED)
    group_list = list(groups.values())
    rng.shuffle(group_list)

    # Sort large groups first for better count packing, but keep shuffle for tie-break
    group_list.sort(key=lambda g: -len(g))

    has_neg = [any(int(x["n_boxes"]) == 0 for x in g) for g in group_list]
    rng = random.Random(SEED)

    total_groups = len(group_list)
    # Aim for ~200 val groups and ~200 test groups (1 image kept each).
    # Rest -> train (all members kept). This yields ~150-250 eval images after
    # dropping non-reps, and keeps most multi-export copies in train only.
    n_valid_groups = 200
    n_test_groups = 200

    neg_groups = [i for i in range(total_groups) if has_neg[i]]
    rng.shuffle(neg_groups)
    # Keep a solid share of negatives in TRAIN (model needs background),
    # and put some in valid/test for FP evaluation.
    # ~49 negative images live in a subset of groups.
    neg_valid = neg_groups[: min(15, len(neg_groups))]
    rest = neg_groups[len(neg_valid):]
    neg_test = rest[: min(15, len(rest))]
    neg_train = rest[len(neg_test):]  # remainder stays train
    # If too few neg groups for train, steal back from valid
    if len(neg_train) < 10 and len(neg_valid) > 5:
        move = neg_valid[-(10 - len(neg_train)):]
        neg_valid = [i for i in neg_valid if i not in move]
        neg_train = move + neg_train
    print(f"neg groups: valid={len(neg_valid)} test={len(neg_test)} train={len(neg_train)}")

    assign: dict[int, str] = {}
    for i in neg_valid:
        assign[i] = "valid"
    for i in neg_test:
        assign[i] = "test"
    for i in neg_train:
        assign[i] = "train"

    # Fill remaining val/test slots with non-neg groups (prefer smaller groups
    # so we do not throw away large multi-export clusters into the 1-rep drop).
    remaining = [i for i in range(total_groups) if i not in assign]
    rng.shuffle(remaining)
    # smaller groups first for val/test fill (less data dropped)
    remaining_small_first = sorted(remaining, key=lambda i: (len(group_list[i]), rng.random()))

    need_valid = n_valid_groups - sum(1 for s in assign.values() if s == "valid")
    need_test = n_test_groups - sum(1 for s in assign.values() if s == "test")
    for i in remaining_small_first:
        if need_valid <= 0 and need_test <= 0:
            break
        if need_valid > 0 and (need_test <= 0 or len(group_list[i]) <= 6):
            assign[i] = "valid"
            need_valid -= 1
        elif need_test > 0:
            assign[i] = "test"
            need_test -= 1
        else:
            assign[i] = "valid"
            need_valid -= 1
    for i in remaining:
        if i not in assign:
            assign[i] = "train"

    from collections import Counter
    print("assigned groups by split:", Counter(assign[i] for i in assign))
    # kept image estimate
    kept_est = {"train": 0, "valid": 0, "test": 0}
    for i, sp in assign.items():
        kept_est[sp] += len(group_list[i]) if sp == "train" else 1
    print("kept image estimate:", kept_est)

    # Materialize
    if args.dst.exists() and args.force:
        shutil.rmtree(args.dst)
    for split in ("train", "valid", "test"):
        (args.dst / split / "images").mkdir(parents=True, exist_ok=True)
        (args.dst / split / "labels").mkdir(parents=True, exist_ok=True)

    manifest_out = []
    groups_out = []
    dropped = []
    for i, g in enumerate(group_list):
        split = assign[i]
        # source split of first member (for debugging)
        src_split = g[0]["split"]
        if split == "train":
            members = g  # keep all train members (multi-export diversity)
        else:
            group_has_neg = any(int(x["n_boxes"]) == 0 for x in g)
            rep = pick_representative(g, prefer_neg=group_has_neg)
            members = [rep]
            for r in g:
                if r["path"] != rep["path"]:
                    dropped.append({"path": r["path"], "reason": f"non_rep_of_{split}_group", "split_assign": split})
        gid = f"g{i:05d}"
        groups_out.append({
            "group_id": gid,
            "split": split,
            "n_members": len(g),
            "n_kept": len(members),
            "has_neg": int(any(int(x["n_boxes"]) == 0 for x in g)),
            "sources": "|".join(sorted({x["source"] for x in g})),
            "example": members[0]["name"],
        })
        for r in members:
            src_img = Path(r["path"])
            src_lbl = src_img.parent.parent / "labels" / (src_img.stem + ".txt")
            dst_img = args.dst / split / "images" / src_img.name
            dst_lbl = args.dst / split / "labels" / (src_img.stem + ".txt")
            if dst_img.exists():
                # collision across groups — should not happen; prefix
                dst_img = args.dst / split / "images" / f"{gid}_{src_img.name}"
                dst_lbl = args.dst / split / "labels" / f"{gid}_{src_img.stem}.txt"
            shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")
            h = hashlib.sha256(dst_img.read_bytes()).hexdigest()
            manifest_out.append({
                "image": str(dst_img.relative_to(args.dst)).replace("\\", "/"),
                "group_id": gid,
                "split": split,
                "source": r["source"],
                "base_key": r["base_key"],
                "n_boxes": r["n_boxes"],
                "sha256": h,
            })

    # data.yaml
    data_yaml = {
        "path": str(args.dst.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["Landmine"],
    }
    with open(args.dst / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    SPLIT2.mkdir(parents=True, exist_ok=True)
    with open(SPLIT2 / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["image", "group_id", "split", "source", "base_key", "n_boxes", "sha256"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(manifest_out, key=lambda x: (x["split"], x["image"])))
    with open(SPLIT2 / "groups.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["group_id", "split", "n_members", "n_kept", "has_neg", "sources", "example"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(groups_out)
    with open(SPLIT2 / "dropped_non_reps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "reason", "split_assign"])
        for d in dropped:
            w.writerow([d["path"], d["reason"], d["split_assign"]])

    # sha256 of manifest
    man_path = SPLIT2 / "split_manifest.csv"
    digest = hashlib.sha256(man_path.read_bytes()).hexdigest()
    (SPLIT2 / "sha256.txt").write_text(
        f"{digest}  split_manifest.csv\n", encoding="utf-8"
    )
    (SPLIT2 / "data.yaml").write_text(
        yaml.dump(data_yaml, sort_keys=False), encoding="utf-8"
    )

    # counts
    for split in ("train", "valid", "test"):
        imgs = list((args.dst / split / "images").iterdir())
        lbls = list((args.dst / split / "labels").iterdir())
        empty = sum(1 for p in lbls if p.stat().st_size == 0)
        boxes = 0
        for p in lbls:
            t = p.read_text(encoding="utf-8", errors="replace").strip()
            if t:
                boxes += len([ln for ln in t.splitlines() if ln.strip()])
        print(f"{split}: images={len(imgs)} labels={len(lbls)} empty={empty} box_lines={boxes}")

    print(f"dropped non-reps: {len(dropped)}")
    print(f"manifest sha256: {digest}")
    print(f"dst: {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
