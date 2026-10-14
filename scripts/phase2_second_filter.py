#!/usr/bin/env python3
"""Phase 2c — second-method filter on pHash candidates + remove true leaks from eval.

For each val/test image whose min pHash Hamming to train is <= --ham:
  compute 32x32 RGB MSE (and PSNR) vs the best train neighbour.
If MSE <= --mse (same physical image / re-export), drop the eval image
(move group to train or delete from val/test).

Also reports how many pHash candidates were false positives (high MSE).
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
V2 = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")
AUDIT = ROOT / "paper_results" / "audit_v2"
OUT = ROOT / "paper_results" / "split_v2"


def small_rgb(path: Path, n: int = 32) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((n, n), Image.BILINEAR)
        return np.asarray(im, dtype=np.float64)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, default=V2)
    ap.add_argument("--near", type=Path, default=AUDIT / "near_dup_pairs.csv")
    ap.add_argument("--ham", type=int, default=2)
    ap.add_argument("--mse", type=float, default=80.0,
                    help="True leak if downsampled RGB MSE <= this (tune on samples)")
    ap.add_argument("--apply", action="store_true", help="Remove flagged eval images from v2")
    args = ap.parse_args()

    name_to_path = {}
    for split in ("train", "valid", "test"):
        for p in (args.v2 / split / "images").iterdir():
            name_to_path[p.name] = p

    rows = []
    with open(args.near, newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if int(e["min_hamming"]) > args.ham:
                continue
            qp = name_to_path.get(e["query_name"])
            tp = name_to_path.get(e["train_name"])
            if not qp or not tp:
                continue
            qimg, timg = small_rgb(qp), small_rgb(tp)
            m = mse(qimg, timg)
            rows.append({
                "query_split": e["query_split"],
                "query_name": e["query_name"],
                "train_name": e["train_name"],
                "hamming": int(e["min_hamming"]),
                "mse32": round(m, 2),
                "true_leak": int(m <= args.mse),
            })

    n_cand = len(rows)
    n_leak = sum(r["true_leak"] for r in rows)
    print(f"pHash candidates (ham<= {args.ham}): {n_cand}")
    print(f"true leaks (mse32<= {args.mse}): {n_leak} ({100*n_leak/max(n_cand,1):.1f}%)")
    print(f"false positives (hash only): {n_cand - n_leak}")

    # show borderline
    rows_sorted = sorted(rows, key=lambda r: r["mse32"])
    print("lowest MSE (true leaks):")
    for r in rows_sorted[:8]:
        print(f"  mse={r['mse32']:7.1f} ham={r['hamming']} {r['query_name'][:50]} <-> {r['train_name'][:50]}")
    print("highest MSE among candidates (hash FPs):")
    for r in rows_sorted[-5:]:
        print(f"  mse={r['mse32']:7.1f} ham={r['hamming']} {r['query_name'][:50]} <-> {r['train_name'][:50]}")

    out_csv = OUT / "second_method_filter.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["query_split", "query_name", "train_name", "hamming", "mse32", "true_leak"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["mse32"]))
    print(f"wrote {out_csv}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to remove true-leak eval images.")
        return 0

    # Remove unique query images flagged as true leaks from their eval split
    to_remove = sorted({r["query_name"] for r in rows if r["true_leak"] and r["query_split"] in ("valid", "test")})
    print(f"removing {len(to_remove)} eval images from v2...")
    removed = []
    for name in to_remove:
        for split in ("valid", "test"):
            img = args.v2 / split / "images" / name
            if img.exists():
                lbl = args.v2 / split / "labels" / (name.rsplit(".", 1)[0] + ".txt")
                img.unlink()
                if lbl.exists():
                    lbl.unlink()
                removed.append((split, name))
                break
    with open(OUT / "removed_leaks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "name"])
        w.writerows(removed)
    for split in ("train", "valid", "test"):
        n = len(list((args.v2 / split / "images").iterdir()))
        print(f"  {split}: {n} images remain")
    print(f"removed {len(removed)} -> {OUT / 'removed_leaks.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
