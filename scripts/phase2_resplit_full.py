#!/usr/bin/env python3
"""Phase 2b — rebuild groups with FULL pHash graph, then re-split.

Connects images when min Hamming over 8 dihedral transforms is <= --ham
(default 2), plus base_key and May/Jan sessions. Then assigns whole groups
and writes landmine_combined_v2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined")
DST = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")
AUDIT = ROOT / "paper_results" / "audit"
SPLIT2 = ROOT / "paper_results" / "split_v2"
SEED = 42

sys.path.insert(0, str(ROOT / "scripts"))
from phase1_audit import dihedral, phash64, popcount_u64  # noqa: E402


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


def may_jan_session(name: str) -> str | None:
    m = re.match(r"^(May_.*?|Jan_.*?)_lwir_(\d+)_lwir_(\d+)\.jpg$", name, re.I)
    if m:
        return f"{m.group(1)}_lwir_{m.group(2)}"
    return None


def base_key(name: str) -> str:
    stem = Path(name).stem
    m = re.match(r"^(.+)\.rf\.[0-9a-fA-F]+$", stem)
    return m.group(1) if m else stem


def source_of(name: str) -> str:
    if name.startswith("ds0"):
        return "ds0"
    if name.startswith("ds1"):
        return "ds1"
    if name.startswith("May"):
        return "May_lwir"
    if name.startswith("Jan"):
        return "Jan_lwir"
    return "other"


def pick_rep(members, prefer_neg=False):
    def score(r):
        is_rf = 1 if ".rf." in r["name"] else 0
        is_neg = (0 if int(r["n_boxes"]) == 0 else 1) if prefer_neg else 0
        return (is_rf, is_neg, -int(r["bytes"]), r["name"])
    return sorted(members, key=score)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dst", type=Path, default=DST)
    ap.add_argument("--ham", type=int, default=0, help="pHash edge threshold (0 = exact hash under dihedral)")
    ap.add_argument("--same-source-only", action="store_true", default=True,
                    help="Only add pHash edges within the same source prefix")
    ap.add_argument("--cross-source-ham", type=int, default=-1,
                    help="If >=0, also allow cross-source edges at this hamming")
    ap.add_argument("--n-valid", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # ---- load all images + labels + hashes ----
    rows = []
    hashes: list[np.ndarray] = []
    for split in ("train", "valid", "test"):
        img_dir = args.src / split / "images"
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lbl = args.src / split / "labels" / (p.stem + ".txt")
            n_boxes = 0
            if lbl.exists():
                t = lbl.read_text(encoding="utf-8", errors="replace").strip()
                if t:
                    n_boxes = len([ln for ln in t.splitlines() if ln.strip()])
            with Image.open(p) as im:
                im = im.convert("RGB")
                hs = [phash64(t) for t in dihedral(im)]
                w, h = im.size
            rows.append({
                "path": str(p), "name": p.name, "orig_split": split,
                "bytes": p.stat().st_size, "n_boxes": n_boxes,
                "source": source_of(p.name), "base_key": base_key(p.name),
                "width": w, "height": h,
            })
            hashes.append(np.array(hs, dtype=np.uint64))
            if len(rows) % 500 == 0:
                print(f"  hashed {len(rows)}...")

    print(f"hashed all {len(rows)} images in {time.time()-t0:.1f}s")
    H = np.stack(hashes, axis=0)  # (N, 8)
    N = H.shape[0]

    # ---- union-find: base_key, session, pHash ----
    uf = UF()
    for i, r in enumerate(rows):
        uf.find(r["path"])
        uf.union(r["path"], f"base::{r['base_key']}")
        sess = may_jan_session(r["name"])
        if sess:
            uf.union(r["path"], f"sess::{sess}")

    # pHash edges: for each image, find others with min hamming <= ham
    # Use identity hash vs all 8 of others for speed (symmetric enough if we
    # check both directions via full tensor min).
    # Efficient: compare H[i,0] to H[j,:] and H[i,:] to H[j,0] etc.
    # Full (N,N,8) too big; do chunked.
    print("building pHash edges...")
    n_edges = 0
    chunk = 256
    src_arr = np.array([r["source"] for r in rows])
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        block = H[start:end, None, :]
        xor = block ^ H[None, :, :]
        dists = popcount_u64(xor.reshape(-1)).reshape(end - start, N, 8)
        mind = dists.min(axis=2)
        for a in range(end - start):
            i = start + a
            js = np.nonzero(mind[a] <= args.ham)[0]
            for j in js:
                j = int(j)
                if j <= i:
                    continue
                if src_arr[i] != src_arr[j]:
                    continue
                uf.union(rows[i]["path"], rows[j]["path"])
                n_edges += 1
        if start % 2048 == 0:
            print(f"  pairs scanned {end}/{N}, edges={n_edges}")
    print(f"pHash edges (ham<= {args.ham}, same source): {n_edges}")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[uf.find(r["path"])].append(i)
    print(f"components: {len(groups)}")

    # ---- assign groups ----
    rng = random.Random(SEED)
    group_list = list(groups.values())
    has_neg = [any(rows[i]["n_boxes"] == 0 for i in g) for g in group_list]
    # shuffle then sort by size desc with random tiebreak
    order = list(range(len(group_list)))
    rng.shuffle(order)
    order.sort(key=lambda i: -len(group_list[i]))

    neg_idx = [i for i in range(len(group_list)) if has_neg[i]]
    rng.shuffle(neg_idx)
    neg_valid = neg_idx[: min(15, len(neg_idx))]
    rest = neg_idx[len(neg_valid):]
    neg_test = rest[: min(15, len(rest))]
    neg_train = rest[len(neg_test):]
    if len(neg_train) < 8 and len(neg_valid) > 5:
        move = neg_valid[-(8 - len(neg_train)):]
        neg_valid = [i for i in neg_valid if i not in move]
        neg_train = move + neg_train
    print(f"neg groups valid={len(neg_valid)} test={len(neg_test)} train={len(neg_train)}")

    assign: dict[int, str] = {}
    for i in neg_valid:
        assign[i] = "valid"
    for i in neg_test:
        assign[i] = "test"
    for i in neg_train:
        assign[i] = "train"

    remaining = [i for i in range(len(group_list)) if i not in assign]
    rng.shuffle(remaining)
    remaining_small = sorted(remaining, key=lambda i: (len(group_list[i]), rng.random()))
    need_v = args.n_valid - sum(1 for s in assign.values() if s == "valid")
    need_t = args.n_test - sum(1 for s in assign.values() if s == "test")
    for i in remaining_small:
        if need_v <= 0 and need_t <= 0:
            break
        if need_v > 0 and (need_t <= 0 or len(group_list[i]) <= 8):
            assign[i] = "valid"
            need_v -= 1
        elif need_t > 0:
            assign[i] = "test"
            need_t -= 1
    for i in remaining:
        if i not in assign:
            assign[i] = "train"

    from collections import Counter
    print("groups by split:", Counter(assign[i] for i in assign))

    # ---- materialize v2 ----
    if args.dst.exists():
        shutil.rmtree(args.dst)
    for split in ("train", "valid", "test"):
        (args.dst / split / "images").mkdir(parents=True)
        (args.dst / split / "labels").mkdir(parents=True)

    manifest_out, groups_out, dropped = [], [], []
    used_names: dict[str, set] = defaultdict(set)
    for gi, members in enumerate(group_list):
        split = assign[gi]
        gid = f"g{gi:05d}"
        mem_rows = [rows[i] for i in members]
        if split == "train":
            keep = mem_rows
        else:
            prefer_neg = any(int(r["n_boxes"]) == 0 for r in mem_rows)
            rep = pick_rep(mem_rows, prefer_neg=prefer_neg)
            keep = [rep]
            for r in mem_rows:
                if r["path"] != rep["path"]:
                    dropped.append((r["path"], split))
        groups_out.append({
            "group_id": gid, "split": split, "n_members": len(mem_rows),
            "n_kept": len(keep),
            "has_neg": int(any(int(r["n_boxes"]) == 0 for r in mem_rows)),
            "sources": "|".join(sorted({r["source"] for r in mem_rows})),
            "example": keep[0]["name"],
        })
        for r in keep:
            src_img = Path(r["path"])
            src_lbl = src_img.parent.parent / "labels" / (src_img.stem + ".txt")
            name = src_img.name
            if name in used_names[split]:
                name = f"{gid}_{name}"
            used_names[split].add(name)
            dst_img = args.dst / split / "images" / name
            # Do NOT use Path.with_suffix on Roboflow names (foo.rf.HASH.jpg).
            stem = name.rsplit(".", 1)[0]
            dst_lbl = args.dst / split / "labels" / (stem + ".txt")
            shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")
            manifest_out.append({
                "image": f"{split}/images/{name}",
                "group_id": gid, "split": split, "source": r["source"],
                "base_key": r["base_key"], "n_boxes": r["n_boxes"],
                "sha256": hashlib.sha256(dst_img.read_bytes()).hexdigest(),
            })

    data_yaml = {
        "path": str(args.dst.resolve()).replace("\\", "/"),
        "train": "train/images", "val": "valid/images", "test": "test/images",
        "nc": 1, "names": ["Landmine"],
    }
    (args.dst / "data.yaml").write_text(yaml.dump(data_yaml, sort_keys=False), encoding="utf-8")

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
        w.writerow(["path", "split_assign"])
        w.writerows(dropped)

    digest = hashlib.sha256((SPLIT2 / "split_manifest.csv").read_bytes()).hexdigest()
    (SPLIT2 / "sha256.txt").write_text(f"{digest}  split_manifest.csv\n", encoding="utf-8")
    (SPLIT2 / "data.yaml").write_text(yaml.dump(data_yaml, sort_keys=False), encoding="utf-8")

    for split in ("train", "valid", "test"):
        imgs = list((args.dst / split / "images").iterdir())
        lbls = list((args.dst / split / "labels").iterdir())
        empty = sum(1 for p in lbls if p.stat().st_size == 0)
        boxes = 0
        for p in lbls:
            t = p.read_text(encoding="utf-8", errors="replace").strip()
            if t:
                boxes += len(t.splitlines())
        print(f"{split}: images={len(imgs)} empty={empty} boxes={boxes}")
    print(f"dropped={len(dropped)} sha256={digest}")
    print(f"done {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
