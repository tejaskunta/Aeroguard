#!/usr/bin/env python3
"""Phase 5.1 — build the stage-2 crop dataset (plan §Phase 5.1).

Positives : every train GT box with jitter (shift <=10% of box size, scale
            U(0.9, 1.2)), 20% context padding, square crop, 224x224.
Negatives : (a) random background crops, same size distribution, IoU < 0.1 with
            any GT; (b) YOLO false positives on train images at conf 0.05 from an
            EARLY checkpoint (default epoch20, needs save_period=5 from Phase 3);
            (c) crops from real negative (empty-label) train images.
Val crops : positives + random background from valid/ for classifier monitoring
            only (final eval is pipeline-level).

Outputs under --out:
  train/pos/*.jpg  train/neg/*.jpg
  val/pos/*.jpg    val/neg/*.jpg
  manifest.csv     (path, label, split, kind, source_image, box)
  stats.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def read_boxes(lbl_path: Path, w: int, h: int):
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8", errors="replace").splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(x) for x in p[1:5])
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
    return boxes


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def jittered_square(box, rng, pad=0.2):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    scale = rng.uniform(0.9, 1.2)
    side = max(bw, bh) * scale * (1 + 2 * pad)
    cx = (x1 + x2) / 2 + rng.uniform(-0.1, 0.1) * bw
    cy = (y1 + y2) / 2 + rng.uniform(-0.1, 0.1) * bh
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def clip_square(sq, w, h):
    x1, y1, x2, y2 = sq
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def save_crop(img: Image.Image, sq, out_path: Path, size=224):
    x1, y1, x2, y2 = (int(round(v)) for v in sq)
    crop = img.crop((x1, y1, x2, y2)).resize((size, size), Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, quality=92)


def random_bg_square(boxes, w, h, rng, size_pool, tries=25):
    for _ in range(tries):
        side = rng.choice(size_pool) if size_pool else rng.uniform(64, 256)
        cx = rng.uniform(side / 2, w - side / 2) if w > side + 1 else w / 2
        cy = rng.uniform(side / 2, h - side / 2) if h > side + 1 else h / 2
        sq = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        sq = clip_square(sq, w, h)
        if sq is None:
            continue
        cand = list(sq)
        if all(iou(cand, b) < 0.1 for b in boxes):
            return cand
    return None


def mine_fp_squares(yolo_weights, img_dir: Path, label_dir: Path, rng, conf=0.05,
                    pad=0.2, max_per_image=5):
    """FP crops from an early checkpoint at low conf (plan §Phase 5.1)."""
    from ultralytics import YOLO
    yolo = YOLO(str(yolo_weights))
    fps = {}
    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for i, p in enumerate(paths):
        res = yolo.predict(str(p), conf=conf, imgsz=640, verbose=False)[0]
        h, w = res.orig_shape
        boxes = read_boxes(label_dir / (p.name.rsplit(".", 1)[0] + ".txt"), w, h)
        if not len(res.boxes):
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        kept = 0
        for (x1, y1, x2, y2) in xyxy:
            if kept >= max_per_image:
                break
            det = [float(x1), float(y1), float(x2), float(y2)]
            if any(iou(det, b) >= 0.3 for b in boxes):
                continue  # matches GT -> not an FP
            bw, bh = det[2] - det[0], det[3] - det[1]
            side = max(bw, bh) * (1 + 2 * pad)
            cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
            sq = clip_square((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), w, h)
            if sq:
                fps.setdefault(p, []).append(list(sq))
                kept += 1
        if (i + 1) % 500 == 0:
            print(f"  fp-mine {i + 1}/{len(paths)}", flush=True)
    return fps


def build_split(split_root: Path, out_root: Path, rng: random.Random,
                yolo_fp=None, imgsz=224, target_neg_per_pos=1.0):
    img_dir = split_root / "images"
    lbl_dir = split_root / "labels"
    rows = []
    size_pool = []
    n_pos = n_neg_rand = n_neg_fp = n_neg_empty = 0

    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    # pass 1: positives (+ record sizes)
    loaded = {}
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        boxes = read_boxes(lbl_dir / (p.name.rsplit(".", 1)[0] + ".txt"), w, h)
        loaded[p] = (img, boxes)
        for bi, b in enumerate(boxes):
            sq = clip_square(jittered_square(b, rng), w, h)
            if sq is None:
                continue
            side = sq[2] - sq[0]
            size_pool.append(side)
            name = f"{p.stem}__b{bi}.jpg"
            save_crop(img, sq, out_root / "pos" / name, imgsz)
            rows.append([str(out_root / "pos" / name), 1, split_root.name, "gt_jitter",
                        p.name, bi])
            n_pos += 1

    # pass 2: negatives
    neg_budget = int(n_pos * target_neg_per_pos)
    for p in paths:
        img, boxes = loaded[p]
        w, h = img.size
        if len(boxes) == 0:
            # real negative image: one full-frame-ish square + one random bg
            side = min(w, h)
            sq = clip_square(((w - side) / 2, (h - side) / 2,
                              (w + side) / 2, (h + side) / 2), w, h)
            if sq:
                name = f"{p.stem}__empty.jpg"
                save_crop(img, sq, out_root / "neg" / name, imgsz)
                rows.append([str(out_root / "neg" / name), 0, split_root.name, "empty_image",
                            p.name, ""])
                n_neg_empty += 1
        elif len(boxes) > 0:
            # up to 3 accepted background crops per image (enough negatives for 1:3)
            for k in range(3):
                sq = random_bg_square(boxes, w, h, rng, size_pool)
                if not sq:
                    break
                name = f"{p.stem}__bg{k}.jpg"
                save_crop(img, sq, out_root / "neg" / name, imgsz)
                rows.append([str(out_root / "neg" / name), 0, split_root.name, "random_bg",
                            p.name, ""])
                n_neg_rand += 1

    if yolo_fp is not None:
        fpm = mine_fp_squares(yolo_fp, img_dir, lbl_dir, rng)
        for p, sqs in fpm.items():
            img = loaded[p][0] if p in loaded else Image.open(p).convert("RGB")
            for j, sq in enumerate(sqs[:4]):
                name = f"{p.stem}__fp{j}.jpg"
                save_crop(img, sq, out_root / "neg" / name, imgsz)
                rows.append([str(out_root / "neg" / name), 0, split_root.name, "yolo_fp",
                            p.name, ""])
                n_neg_fp += 1

    stats = {"pos": n_pos, "neg_random_bg": n_neg_rand, "neg_yolo_fp": n_neg_fp,
             "neg_empty_image": n_neg_empty,
             "neg_total": n_neg_rand + n_neg_fp + n_neg_empty}
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yolo", default=None,
                    help="early YOLO checkpoint for FP mining (e.g. .../epoch20.pt); "
                         "omit to skip FP negatives")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(args.seed)
    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for split, fp in (("train", args.yolo), ("valid", None)):
        rows, stats = build_split(root / split, out / split, rng,
                                  yolo_fp=fp, imgsz=args.imgsz)
        print(split, stats, flush=True)
        all_rows += rows
        (out / split / "stats.json").write_text(json.dumps(stats, indent=2),
                                                encoding="utf-8")

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split", "kind", "source_image", "box"])
        w.writerows(all_rows)
    print(f"wrote {out / 'manifest.csv'} ({len(all_rows)} crops)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
