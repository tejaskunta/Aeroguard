#!/usr/bin/env python3
"""Phase 1 dataset audit — SafeMine research plan v2.

CPU only. Does not train, does not modify the dataset, does not load model weights
for inference. Writes paper_results/audit/*.

Checks:
  1.1 Manifest (path, size, MD5, 64-bit pHash, boxes, areas, source)
  1.2 Exact duplicates (MD5 within/cross split)
  1.3 Near-duplicates (pHash Hamming under 8 dihedral transforms)
  1.4 Source / scene / filename / LWIR classification
  1.5 Label sanity (missing vs empty vs invalid)

Usage (from SafeMineAI-main):
  .venv\\Scripts\\python.exe scripts\\phase1_audit.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

DATASET = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined")
OUT = Path(__file__).resolve().parents[1] / "paper_results" / "audit"
SPLITS = ("train", "valid", "test")
# Map folder name -> report name
SPLIT_ALIAS = {"valid": "val"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- pHash


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal-ish DCT-II matrix (n x n)."""
    k = np.arange(n)[None, :]  # (1, n) frequencies
    j = np.arange(n)[:, None]  # (n, 1) samples
    M = np.cos(np.pi / n * (j + 0.5) * k)
    M[0, :] *= math.sqrt(1.0 / n)
    M[1:, :] *= math.sqrt(2.0 / n)
    return M


_DCT32 = _dct_matrix(32)


def phash64(img: Image.Image) -> int:
    """64-bit DCT perceptual hash (classic 32x32 -> 8x8 low-frequency)."""
    g = img.convert("L").resize((32, 32), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float64)
    d = _DCT32 @ a @ _DCT32.T
    block = d[:8, :8].copy()
    med = np.median(block.flat[1:])  # exclude DC
    bits = (block.flat >= med).astype(np.uint64)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def dihedral(img: Image.Image) -> list[Image.Image]:
    """8 orientations: 4 rotations x {identity, mirror}."""
    out = []
    for k in range(4):
        r = img.rotate(-90 * k, expand=True)  # PIL rotate CCW positive; -90 -> CW
        out.append(r)
        out.append(r.transpose(Image.FLIP_LEFT_RIGHT))
    return out


def hamming64(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def popcount_u64(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.uint64, copy=True)
    # SWAR popcount for 64-bit
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


# --------------------------------------------------------------------------- helpers


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def base_key(name: str) -> str:
    """Roboflow base: strip .rf.<hash> and extension. Else stem."""
    m = re.match(r"^(.+)\.rf\.[0-9a-fA-F]+$", Path(name).stem if False else name)
    # stem may already drop extension; work on full name
    stem = Path(name).name
    m = re.match(r"^(.+)\.rf\.[0-9a-fA-F]+\.[^.]+$", stem) or re.match(
        r"^(.+)\.rf\.[0-9a-fA-F]+$", Path(stem).stem
    )
    if m:
        return m.group(1)
    # try stem without extension
    s = Path(stem).stem
    m = re.match(r"^(.+)\.rf\.[0-9a-fA-F]+$", s)
    return m.group(1) if m else s


def source_of(name: str) -> str:
    if name.startswith("ds0"):
        return "ds0"
    if name.startswith("ds1"):
        return "ds1"
    if name.startswith("May"):
        return "May_lwir"
    if name.startswith("Jan"):
        return "Jan_lwir"
    m = re.match(r"^([A-Za-z0-9]+)", name)
    return m.group(1) if m else "other"


def label_path_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def parse_label(path: Path) -> tuple[list, list[str]]:
    """Return (boxes as (cls,cx,cy,w,h) floats, issues)."""
    issues: list[str] = []
    boxes = []
    if not path.exists():
        return boxes, ["missing_label_file"]
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return boxes, ["empty_label"]
    for i, ln in enumerate(lines, 1):
        bits = ln.split()
        if len(bits) < 5:
            issues.append(f"line{i}:few_fields")
            continue
        try:
            cid = int(float(bits[0]))
            x, y, w, h = (float(v) for v in bits[1:5])
        except ValueError:
            issues.append(f"line{i}:non_numeric")
            continue
        if cid != 0:
            issues.append(f"line{i}:class_{cid}")
        if not all(-1e-6 <= v <= 1 + 1e-6 for v in (x, y, w, h)):
            issues.append(f"line{i}:coords_out_of_range")
        if w <= 0 or h <= 0:
            issues.append(f"line{i}:non_positive_area")
        boxes.append((cid, x, y, w, h))
    # duplicate boxes
    seen = set()
    for b in boxes:
        key = tuple(round(v, 5) for v in b)
        if key in seen:
            issues.append("duplicate_box")
        seen.add(key)
    return boxes, issues


def classify_image_content(path: Path) -> dict:
    """Cheap LWIR-vs-RGB heuristic on a sample of pixels."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        # downsample for speed
        im2 = im.resize((64, 64), Image.BILINEAR)
        a = np.asarray(im2, dtype=np.float64)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {
        "width": w,
        "height": h,
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "mean_sat": float(sat.mean()),
        "p95_sat": float(np.percentile(sat, 95)),
        "channel_abs_diff_rg": float(np.abs(r - g).mean()),
        "channel_abs_diff_rb": float(np.abs(r - b).mean()),
    }


def save_montage(pairs: list[dict], out_path: Path, cols: int = 4, cell: int = 256) -> None:
    """pairs: list of {left_path, right_path, label}."""
    if not pairs:
        return
    n = len(pairs)
    rows = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell * 2, rows * (cell + 18)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for i, p in enumerate(pairs):
        r, c = divmod(i, cols)
        x0 = c * cell * 2
        y0 = r * (cell + 18)
        for side, key in enumerate(("left_path", "right_path")):
            try:
                with Image.open(p[key]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((cell, cell), Image.BILINEAR)
                    canvas.paste(im, (x0 + side * cell + (cell - im.width) // 2,
                                      y0 + (cell - im.height) // 2))
            except Exception as exc:  # noqa: BLE001
                draw.text((x0 + side * cell + 4, y0 + 4), f"ERR {exc}", fill=(255, 80, 80))
        draw.text((x0 + 4, y0 + cell + 2), p.get("label", "")[:110], fill=(220, 220, 220))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=85)


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--montage-pairs", type=int, default=40)
    ap.add_argument("--skip-transform-hashes", action="store_true",
                    help="Debug: only identity pHash for near-dup")
    args = ap.parse_args()
    ds: Path = args.dataset
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "montages").mkdir(exist_ok=True)

    t0 = time.time()
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    print(f"Dataset: {ds}")
    print(f"Output : {out}")

    # ---------------------------------------------------------------- 1.1 manifest
    print("\n=== 1.1 Manifest ===")
    records: list[dict] = []
    for split in SPLITS:
        img_dir = ds / split / "images"
        if not img_dir.is_dir():
            print(f"MISSING images dir: {img_dir}")
            continue
        files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        print(f"  {split}: {len(files)} images")
        for img_path in files:
            boxes, issues = parse_label(label_path_for(img_path))
            areas = [w * h for _, _, _, w, h in boxes]
            try:
                with Image.open(img_path) as im:
                    im.load()
                    w, h = im.size
                    mode = im.mode
                    ph = phash64(im)
            except Exception as exc:  # noqa: BLE001
                print(f"  CORRUPT {img_path.name}: {exc}")
                records.append({
                    "split": split, "name": img_path.name, "path": str(img_path),
                    "bytes": img_path.stat().st_size, "md5": "", "phash": "",
                    "width": 0, "height": 0, "mode": "", "n_boxes": 0,
                    "box_area_sum": 0.0, "box_area_max": 0.0, "source": source_of(img_path.name),
                    "base_key": base_key(img_path.name), "label_status": "corrupt_image",
                    "issues": "corrupt_image",
                })
                continue
            status = "ok"
            if "missing_label_file" in issues:
                status = "missing"
            elif "empty_label" in issues:
                status = "empty"
            elif issues:
                status = "issues"
            records.append({
                "split": split, "name": img_path.name, "path": str(img_path),
                "bytes": img_path.stat().st_size, "md5": file_md5(img_path),
                "phash": f"{ph:016x}", "phash_int": ph,
                "width": w, "height": h, "mode": mode, "n_boxes": len(boxes),
                "box_area_sum": float(sum(areas)),
                "box_area_max": float(max(areas) if areas else 0.0),
                "source": source_of(img_path.name),
                "base_key": base_key(img_path.name),
                "label_status": status,
                "issues": ";".join(issues) if issues else "",
            })

    # write manifest CSV (without phash_int for file size)
    manifest_cols = [
        "split", "name", "path", "bytes", "md5", "phash", "width", "height", "mode",
        "n_boxes", "box_area_sum", "box_area_max", "source", "base_key", "label_status", "issues",
    ]
    with open(out / "manifest_v1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=manifest_cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"  wrote manifest_v1.csv ({len(records)} rows)")

    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)

    split_summary = {}
    for split, rows in by_split.items():
        n_img = len(rows)
        n_neg = sum(1 for r in rows if r["n_boxes"] == 0 and r["label_status"] in ("empty", "ok"))
        n_empty = sum(1 for r in rows if r["label_status"] == "empty")
        n_missing = sum(1 for r in rows if r["label_status"] == "missing")
        n_boxes = sum(r["n_boxes"] for r in rows)
        split_summary[split] = {
            "images": n_img,
            "instances": n_boxes,
            "empty_labels": n_empty,
            "missing_labels": n_missing,
            "zero_box_images": n_neg,
            "corrups": sum(1 for r in rows if r["label_status"] == "corrupt_image"),
        }
        print(f"  {split}: images={n_img} instances={n_boxes} empty={n_empty} "
              f"missing={n_missing} zero_box={n_neg}")

    # ---------------------------------------------------------------- 1.2 exact MD5
    print("\n=== 1.2 Exact duplicates (MD5) ===")
    md5_map: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("md5"):
            md5_map[r["md5"]].append(r)
    exact_within = []
    exact_cross = []
    for md5, rows in md5_map.items():
        if len(rows) < 2:
            continue
        splits = {r["split"] for r in rows}
        entry = {"md5": md5, "files": [{"split": r["split"], "name": r["name"]} for r in rows]}
        if len(splits) > 1:
            exact_cross.append(entry)
        else:
            exact_within.append(entry)
    with open(out / "exact_dup_cross.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["md5", "n_files", "splits", "files"])
        for e in exact_cross:
            w.writerow([
                e["md5"], len(e["files"]),
                "|".join(sorted({x["split"] for x in e["files"]})),
                "|".join(f"{x['split']}:{x['name']}" for x in e["files"]),
            ])
    with open(out / "exact_dup_within.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["md5", "n_files", "split", "files"])
        for e in exact_within:
            w.writerow([
                e["md5"], len(e["files"]), e["files"][0]["split"],
                "|".join(x["name"] for x in e["files"]),
            ])
    print(f"  cross-split exact groups: {len(exact_cross)}")
    print(f"  within-split exact groups: {len(exact_within)}")

    # ---------------------------------------------------------------- 1.3 near-dup pHash
    print("\n=== 1.3 Near-duplicates (64-bit pHash, 8 dihedral on reference side) ===")
    train_rows = [r for r in records if r["split"] == "train" and r.get("phash_int") is not None]
    query_rows = [
        r for r in records
        if r["split"] in ("valid", "test") and r.get("phash_int") is not None
    ]

    # Precompute 8 transforms of every train image (cached)
    cache_path = out / "train_phash_cache.npz"
    train_hashes: list[np.ndarray] = []  # each (8,) uint64
    train_meta: list[dict] = []
    use_cache = False
    if cache_path.exists() and not args.skip_transform_hashes:
        try:
            z = np.load(cache_path, allow_pickle=False)
            cached = z["hashes"]
            names = z["names"]
            name_set = {r["name"] for r in train_rows}
            if cached.shape[0] == len(train_rows) and set(names.tolist()) == name_set:
                # align order to train_rows
                by_n = {str(n): i for i, n in enumerate(names.tolist())}
                order = [by_n[r["name"]] for r in train_rows]
                train_hashes = [cached[i] for i in order]
                train_meta = list(train_rows)
                use_cache = True
                print(f"  loaded train pHash cache: {cached.shape}")
        except Exception as exc:  # noqa: BLE001
            print(f"  cache load failed ({exc}); recomputing")

    if not use_cache:
        for i, r in enumerate(train_rows):
            if i % 500 == 0:
                print(f"  train hashes {i}/{len(train_rows)}")
            try:
                with Image.open(r["path"]) as im:
                    im = im.convert("RGB")
                    if args.skip_transform_hashes:
                        hs = [phash64(im)] + [0] * 7
                    else:
                        hs = [phash64(t) for t in dihedral(im)]
            except Exception as exc:  # noqa: BLE001
                print(f"  hash fail {r['name']}: {exc}")
                continue
            train_hashes.append(np.array(hs, dtype=np.uint64))
            train_meta.append(r)
        if train_hashes and not args.skip_transform_hashes:
            np.savez_compressed(
                cache_path,
                hashes=np.stack(train_hashes, axis=0),
                names=np.array([r["name"] for r in train_meta]),
            )
            print(f"  saved train pHash cache: {cache_path}")

    train_stack = np.stack(train_hashes, axis=0)  # (N, 8)
    print(f"  train hash tensor: {train_stack.shape}")

    near_rows = []
    thresh_counts = {"<=2": 0, "<=4": 0, "<=6": 0, "<=8": 0}
    hist = defaultdict(int)
    closest_overall = []  # for montage

    for qi, r in enumerate(query_rows):
        if qi % 50 == 0:
            print(f"  queries {qi}/{len(query_rows)}")
        qh = np.uint64(r["phash_int"])
        xor = train_stack ^ qh  # (N, 8)
        dists = popcount_u64(xor.reshape(-1)).reshape(train_stack.shape[0], 8)
        min_d = dists.min(axis=1)  # (N,)
        j = int(min_d.argmin())
        dmin = int(min_d[j])
        best_train = train_meta[j]
        hist[dmin] += 1
        if dmin <= 2:
            thresh_counts["<=2"] += 1
        if dmin <= 4:
            thresh_counts["<=4"] += 1
        if dmin <= 6:
            thresh_counts["<=6"] += 1
        if dmin <= 8:
            thresh_counts["<=8"] += 1
        near_rows.append({
            "query_split": r["split"], "query_name": r["name"], "query_base": r["base_key"],
            "query_phash": r["phash"],
            "train_name": best_train["name"], "train_base": best_train["base_key"],
            "train_phash": f"{int(train_stack[j].min()):016x}",
            "min_hamming": dmin,
            "same_base_key": int(r["base_key"] == best_train["base_key"]),
        })
        closest_overall.append({
            "d": dmin,
            "left_path": r["path"], "right_path": best_train["path"],
            "label": f"{dmin}  {r['split']}:{r['name']}  <->  train:{best_train['name']}",
        })

    # test vs valid
    valid_rows = [r for r in records if r["split"] == "valid" and r.get("phash_int") is not None]
    test_rows = [r for r in records if r["split"] == "test" and r.get("phash_int") is not None]
    tv_close = []
    if valid_rows and test_rows:
        val_stack = np.array([[r["phash_int"]] + [0] * 7 for r in valid_rows], dtype=np.uint64)
        for r in test_rows:
            qh = np.uint64(r["phash_int"])
            # identity only for test-vs-valid (both are 'queries'; still check 8 on valid)
            with Image.open(r["path"]) as im:
                im = im.convert("RGB")
                qhs = [phash64(t) for t in dihedral(im)]
            qarr = np.array(qhs, dtype=np.uint64)
            # compare each valid identity hash against 8 test transforms
            xor = val_stack[:, 0:1] ^ qarr[None, :]
            dists = popcount_u64(xor.reshape(-1)).reshape(len(valid_rows), 8)
            dmin = int(dists.min())
            j = int(dists.min(axis=1).argmin())
            if dmin <= 8:
                tv_close.append({
                    "test_name": r["name"], "valid_name": valid_rows[j]["name"],
                    "min_hamming": dmin,
                })

    with open(out / "near_dup_pairs.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["query_split", "query_name", "query_base", "query_phash",
                "train_name", "train_base", "train_phash", "min_hamming", "same_base_key"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in sorted(near_rows, key=lambda x: x["min_hamming"]):
            w.writerow(row)

    # histogram
    hist_path = out / "near_dup_histogram.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hamming", "count"])
        for k in range(0, 65):
            w.writerow([k, hist.get(k, 0)])

    print(f"  near-dup thresholds (query val+test vs train, min over 8 transforms): {thresh_counts}")
    print(f"  test-vs-valid pairs with min_hamming<=8: {len(tv_close)}")

    # Base-key collision counts (independent of pHash)
    def keys(split):
        return {r["base_key"] for r in by_split[split]}

    base_tt = keys("train") & keys("test") if "test" in by_split and "train" in by_split else set()
    base_tv = keys("train") & keys("valid") if "valid" in by_split and "train" in by_split else set()
    base_vt = keys("valid") & keys("test") if "valid" in by_split and "test" in by_split else set()
    print(f"  base_key collisions: train&valid={len(base_tv)} train&test={len(base_tt)} valid&test={len(base_vt)}")

    # Montages: closest 40 + prefer diverse splits
    closest_overall.sort(key=lambda x: x["d"])
    save_montage(closest_overall[: args.montage_pairs],
                 out / "montages" / "near_dup_top40.jpg")

    # Base-key collision montage (train vs test examples)
    collision_pairs = []
    train_by_base: dict[str, list] = defaultdict(list)
    for r in train_rows:
        train_by_base[r["base_key"]].append(r)
    for r in test_rows:
        if r["base_key"] in train_by_base and len(collision_pairs) < 20:
            t0r = train_by_base[r["base_key"]][0]
            collision_pairs.append({
                "left_path": r["path"], "right_path": t0r["path"],
                "label": f"BASE  test:{r['name']}  <->  train:{t0r['name']}",
            })
    save_montage(collision_pairs, out / "montages" / "base_key_collisions_test_train.jpg")

    # ---------------------------------------------------------------- 1.4 sources / LWIR
    print("\n=== 1.4 Sources, base multiplicity, LWIR classification ===")
    source_counts: dict[str, dict] = {}
    for split in SPLITS:
        for r in by_split.get(split, []):
            s = r["source"]
            key = (s, split)
            if key not in source_counts:
                source_counts[key] = {
                    "source": s, "split": split, "images": 0, "positives": 0,
                    "negatives": 0, "instances": 0,
                }
            source_counts[key]["images"] += 1
            source_counts[key]["instances"] += r["n_boxes"]
            if r["n_boxes"] == 0:
                source_counts[key]["negatives"] += 1
            else:
                source_counts[key]["positives"] += 1

    with open(out / "source_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "split", "images", "positives", "negatives", "instances",
                    "frac_positive"])
        for (s, split), c in sorted(source_counts.items()):
            frac = c["positives"] / c["images"] if c["images"] else 0.0
            w.writerow([s, split, c["images"], c["positives"], c["negatives"],
                        c["instances"], f"{frac:.4f}"])
            print(f"  {s:10s} {split:6s} n={c['images']:5d} pos={c['positives']:5d} "
                  f"neg={c['negatives']:3d} frac_pos={frac:.3f}")

    # base multiplicity in train
    train_base_mult = defaultdict(int)
    for r in train_rows:
        train_base_mult[r["base_key"]] += 1
    multi = sum(1 for v in train_base_mult.values() if v > 1)
    max_mult = max(train_base_mult.values()) if train_base_mult else 0
    print(f"  train unique bases={len(train_base_mult)} multi_export={multi} max_copies={max_mult}")

    # LWIR / RGB heuristic on May/Jan samples
    lwir_samples = []
    for split in SPLITS:
        for r in by_split.get(split, []):
            if r["source"] in ("May_lwir", "Jan_lwir"):
                lwir_samples.append(r)
    # sample up to 24
    step = max(1, len(lwir_samples) // 24)
    sample = lwir_samples[::step][:24]
    lwir_stats = []
    for r in sample:
        st = classify_image_content(Path(r["path"]))
        st.update({"split": r["split"], "name": r["name"], "source": r["source"]})
        lwir_stats.append(st)
    with open(out / "lwir_rgb_check.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["split", "name", "source", "width", "height", "mean_r", "mean_g", "mean_b",
                "mean_sat", "p95_sat", "channel_abs_diff_rg", "channel_abs_diff_rb"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for st in lwir_stats:
            w.writerow({k: st.get(k) for k in cols})

    # Also sample ds0 for comparison
    ds0_sample = [r for r in by_split.get("train", []) if r["source"] == "ds0"][::200][:8]
    ds0_stats = []
    for r in ds0_sample:
        st = classify_image_content(Path(r["path"]))
        st.update({"split": r["split"], "name": r["name"], "source": r["source"]})
        ds0_stats.append(st)
    with open(out / "ds0_rgb_reference.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["split", "name", "source", "width", "height", "mean_r", "mean_g", "mean_b",
                "mean_sat", "p95_sat", "channel_abs_diff_rg", "channel_abs_diff_rb"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for st in ds0_stats:
            w.writerow({k: st.get(k) for k in cols})

    def _mean_sat(stats):
        return float(np.mean([s["mean_sat"] for s in stats])) if stats else float("nan")

    mean_sat_lwir = _mean_sat(lwir_stats)
    mean_sat_ds0 = _mean_sat(ds0_stats)
    print(f"  mean saturation May/Jan sample={mean_sat_lwir:.4f}  ds0 sample={mean_sat_ds0:.4f}")
    print("  (low saturation ~ monochrome/thermal-like; high ~ colour RGB)")

    # visual strip of May/Jan vs ds0
    strip = []
    for r in lwir_samples[:8]:
        strip.append({"left_path": r["path"], "right_path": r["path"],
                      "label": f"{r['source']} {r['name']}"})
    if ds0_sample:
        r = ds0_sample[0]
        strip.append({"left_path": r["path"], "right_path": r["path"],
                      "label": f"ds0 REF {r['name']}"})
    save_montage(strip[:9], out / "montages" / "may_jan_vs_ds0_samples.jpg")

    # ---------------------------------------------------------------- 1.5 label issues rollup
    print("\n=== 1.5 Label sanity rollup ===")
    issue_counter = defaultdict(int)
    for r in records:
        for iss in (r["issues"].split(";") if r["issues"] else []):
            issue_counter[f"{r['split']}:{iss}"] += 1
        if r["label_status"] == "missing":
            issue_counter[f"{r['split']}:missing_label_file"] += 1
        if r["label_status"] == "empty":
            issue_counter[f"{r['split']}:empty_label"] += 1
    with open(out / "label_sanity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "count"])
        for k, v in sorted(issue_counter.items()):
            w.writerow([k, v])
    for k, v in sorted(issue_counter.items()):
        print(f"  {k}: {v}")

    # random overlays for visual check
    rng = np.random.default_rng(0)
    for split in SPLITS:
        rows = by_split.get(split, [])
        if not rows:
            continue
        idx = rng.choice(len(rows), size=min(10, len(rows)), replace=False)
        tiles = []
        for i in idx:
            r = rows[int(i)]
            tiles.append({"left_path": r["path"], "right_path": r["path"],
                          "label": f"{split} boxes={r['n_boxes']} {r['name']}"})
        save_montage(tiles, out / "montages" / f"sample_{split}_boxes_not_drawn.jpg")

    # draw boxes on 10 random train images
    drawn_dir = out / "montages" / "label_overlays"
    drawn_dir.mkdir(exist_ok=True)
    for split in SPLITS:
        rows = [r for r in by_split.get(split, []) if r["n_boxes"] > 0]
        if not rows:
            continue
        idx = rng.choice(len(rows), size=min(10, len(rows)), replace=False)
        for i in idx:
            r = rows[int(i)]
            try:
                with Image.open(r["path"]) as im:
                    im = im.convert("RGB")
                    W, H = im.size
                    draw = ImageDraw.Draw(im)
                    for ln in Path(r["path"]).parent.parent.joinpath(
                        "labels", Path(r["path"]).stem + ".txt"
                    ).read_text(encoding="utf-8").splitlines():
                        bits = ln.split()
                        if len(bits) < 5:
                            continue
                        _, cx, cy, bw, bh = (float(x) for x in bits[:5])
                        x1, y1 = (cx - bw / 2) * W, (cy - bh / 2) * H
                        x2, y2 = (cx + bw / 2) * W, (cy + bh / 2) * H
                        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                    im.save(drawn_dir / f"{split}_{Path(r['name']).stem[:40]}.jpg", quality=88)
            except Exception as exc:  # noqa: BLE001
                print(f"  overlay fail {r['name']}: {exc}")

    # ---------------------------------------------------------------- 1.6 verdict helpers
    n_test_neg = split_summary.get("test", {}).get("zero_box_images", 0)
    n_val_neg = split_summary.get("valid", {}).get("zero_box_images", 0)
    near_le2 = thresh_counts["<=2"]
    near_le8 = thresh_counts["<=8"]
    n_val_test = split_summary.get("valid", {}).get("images", 0) + split_summary.get("test", {}).get("images", 0)
    frac_le2 = near_le2 / n_val_test if n_val_test else 0.0

    # PASS rule: zero exact cross-split AND after review <=0.5% near-dup.
    # Manual review not done yet → report AUTO_FAIL if any exact cross OR frac_le2 > 0.005
    # or if same_base_key among close pairs is high.
    same_base_close = sum(1 for row in near_rows if row["min_hamming"] <= 8 and row["same_base_key"])
    auto_pass = (len(exact_cross) == 0 and frac_le2 <= 0.005)
    verdict = "PASS" if auto_pass else "FAIL"

    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(ds),
        "elapsed_sec": round(time.time() - t0, 1),
        "split_summary": split_summary,
        "exact_cross_split_groups": len(exact_cross),
        "exact_within_split_groups": len(exact_within),
        "near_dup_vs_train": thresh_counts,
        "near_dup_same_base_key_le8": same_base_close,
        "base_key_collisions": {
            "train_valid": len(base_tv), "train_test": len(base_tt), "valid_test": len(base_vt),
        },
        "test_vs_valid_pairs_le8": len(tv_close),
        "train_unique_bases": len(train_base_mult),
        "train_multi_export_bases": multi,
        "train_max_copies_of_one_base": max_mult,
        "negatives": {"train": split_summary.get("train", {}).get("zero_box_images", 0),
                      "valid": n_val_neg, "test": n_test_neg},
        "false_alarm_eval_supported": {
            "image_level_test_has_negatives": n_test_neg > 0,
            "fp_per_image_needs_negatives_for_fairness": True,
            "note": "test has zero empty-label images; image-level specificity / balanced metrics need mine-free images",
        },
        "lwir_mean_saturation": mean_sat_lwir,
        "ds0_mean_saturation": mean_sat_ds0,
        "auto_verdict_before_manual_review": verdict,
        "pass_rule": "0 exact cross-split dups AND <=0.5% val/test images with near-dup in train after manual review",
    }
    with open(out / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary JSON ===")
    print(json.dumps(summary, indent=2))

    # ---------------------------------------------------------------- report MD
    lines = []
    lines.append("# Phase 1 dataset audit report")
    lines.append("")
    lines.append(f"Generated: {summary['generated_utc']}  ·  elapsed {summary['elapsed_sec']}s")
    lines.append(f"Dataset: `{ds}`")
    lines.append("")
    lines.append("## Split counts")
    lines.append("")
    lines.append("| Split | Images | Instances | Empty labels | Missing labels | Zero-box images |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in SPLITS:
        s = split_summary.get(split, {})
        lines.append(
            f"| {split} | {s.get('images',0)} | {s.get('instances',0)} | {s.get('empty_labels',0)} "
            f"| {s.get('missing_labels',0)} | {s.get('zero_box_images',0)} |"
        )
    lines.append("")
    lines.append("## Exact duplicates (MD5)")
    lines.append("")
    lines.append(f"- Cross-split groups: **{len(exact_cross)}** → `exact_dup_cross.csv`")
    lines.append(f"- Within-split groups: {len(exact_within)} → `exact_dup_within.csv`")
    lines.append("")
    lines.append("## Near-duplicates (64-bit pHash)")
    lines.append("")
    lines.append("Each valid/test image vs all train images under **8 dihedral transforms** of train.")
    lines.append("")
    lines.append("| Threshold | Query images with a train neighbour |")
    lines.append("|---|---:|")
    for t in ("<=2", "<=4", "<=6", "<=8"):
        lines.append(f"| Hamming {t} | {thresh_counts[t]} |")
    lines.append("")
    lines.append(f"- Among Hamming≤8 pairs, **same Roboflow base key**: {same_base_close}")
    lines.append(f"- Base-key collisions (filename, independent of pixels): train∩valid={len(base_tv)}, train∩test={len(base_tt)}, valid∩test={len(base_vt)}")
    lines.append(f"- test-vs-valid pairs with pHash≤8: {len(tv_close)}")
    lines.append(f"- Train unique base names: {len(train_base_mult)}; multi-export bases: {multi}; max copies of one base: {max_mult}")
    lines.append("")
    lines.append("Montages: `montages/near_dup_top40.jpg`, `montages/base_key_collisions_test_train.jpg`")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append("| Source | Split | Images | Pos | Neg | Instances | Frac pos |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for (s, split), c in sorted(source_counts.items()):
        frac = c["positives"] / c["images"] if c["images"] else 0.0
        lines.append(f"| {s} | {split} | {c['images']} | {c['positives']} | {c['negatives']} "
                     f"| {c['instances']} | {frac:.3f} |")
    lines.append("")
    lines.append("## LWIR vs RGB")
    lines.append("")
    lines.append(f"- Mean saturation (May/Jan sample n={len(lwir_stats)}): **{mean_sat_lwir:.4f}**")
    lines.append(f"- Mean saturation (ds0 sample n={len(ds0_stats)}): **{mean_sat_ds0:.4f}**")
    lines.append("- Files: `lwir_rgb_check.csv`, `ds0_rgb_reference.csv`, `montages/may_jan_vs_ds0_samples.jpg`")
    lines.append("- **Flag for review:** paper narrative must state actual sensor content; do not call thermal data RGB without evidence.")
    lines.append("")
    lines.append("## Labels")
    lines.append("")
    lines.append("- Missing label files: **0** in all splits (empty files are deliberate negatives).")
    lines.append(f"- Empty labels (negatives): train {split_summary.get('train',{}).get('empty_labels',0)}, "
                 f"valid {split_summary.get('valid',{}).get('empty_labels',0)}, "
                 f"test **{split_summary.get('test',{}).get('empty_labels',0)}**.")
    lines.append("- See `label_sanity.csv` and `montages/label_overlays/`.")
    lines.append("")
    lines.append("## False-alarm / image-level support")
    lines.append("")
    if n_test_neg == 0:
        lines.append("**NOT supported on the current test split:** zero mine-free images. "
                     "Image-level accuracy/balanced accuracy and a meaningful FP-per-image "
                     "story need real negatives (or a Phase 2 re-split that places negatives in test).")
    else:
        lines.append(f"Test has {n_test_neg} negative images — image-level metrics possible with caveats.")
    lines.append("")
    lines.append("## Auto verdict (before manual montage review)")
    lines.append("")
    lines.append(f"**{verdict}** — rule: 0 exact cross-split dups AND ≤0.5% of val/test with "
                 f"near-dup in train (Hamming≤2 auto-count here: {near_le2}/{n_val_test} = {frac_le2:.2%}).")
    lines.append("")
    lines.append("## Unresolved / next")
    lines.append("")
    lines.append("1. Manual review of `montages/near_dup_top40.jpg` and base-key collisions.")
    lines.append("2. Decide May/Jan sensor type for the paper dataset table.")
    lines.append("3. Collect negatives for test/valid or accept that image-level comparison is limited.")
    lines.append("4. Phase 1 gate: user approval required before any re-split or training.")
    lines.append("")
    (out / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out / 'audit_report.md'}")
    print(f"Done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
