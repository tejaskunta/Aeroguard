#!/usr/bin/env python3
"""Phase 6 E5 — dataset audit: flag test and train images in the lowest 10% of
Grad-CAM energy-in-box (plan §Phase 6, E5).

For each split: YOLO at the deployment proposal conf, greedy-match GT, Grad-CAM
per image (top pre-NMS class score), energy share in the matched GT box.
Flag per-split lowest 10% of TP energies; write flag lists + worst-of montage
for hand review (LABEL_FIXES.csv is produced after the hand pass).

Outputs under --out:
  e5_flags.csv        (split, image, det, energy_share, enrichment, flagged)
  e5_summary.json
  e5_worst_train.png, e5_worst_test.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import two_stage_eval as ts  # noqa: E402
from phase6_gradcam import YoloCAM, match_map, energy_stats  # noqa: E402


def run_split(split_root: Path, ycam: YoloCAM, conf: float):
    """One row per image: energy share in the UNION of all GT boxes (plan E5
    flags images, not instances — per-det rows flood multi-mine images with
    zeros because only the salient instance receives CAM energy)."""
    rows = []
    img_dir = split_root / "images"
    lbl_dir = split_root / "labels"
    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for k, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = ts._read_gt(str(lbl_dir / (p.name.rsplit(".", 1)[0] + ".txt")), w, h)
        if not gt:
            continue
        res = ycam.yolo.predict(img, conf=conf, imgsz=640, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        dets = [{"box": list(map(float, b)), "conf": float(c)} for b, c in zip(boxes, confs)]
        d2g = match_map(gt, dets)
        n_matched = sum(1 for g in d2g if g >= 0)
        heat, _ = ycam.cam(img)
        es, a_s, en, pk = energy_stats(heat, gt)      # union of all GT boxes
        rows.append({"split": split_root.name, "image": p.name, "n_gt": len(gt),
                     "n_matched": n_matched, "energy_share": es,
                     "enrichment": en, "peak_inside": int(pk), "flagged": 0})
        if (k + 1) % 500 == 0:
            print(f"  {split_root.name} {k + 1}/{len(paths)}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--montage", type=int, default=12)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ycam = YoloCAM(args.yolo)

    rows = []
    for split in ("test", "train"):
        print(f"scanning {split}...", flush=True)
        rows += run_split(data / split, ycam, args.conf)

    for split in ("test", "train"):
        srows = [r for r in rows if r["split"] == split]
        es = np.array([r["energy_share"] for r in srows])
        thr = np.quantile(es, 0.10) if len(es) else 1.0
        for r in srows:
            r["flagged"] = int(r["energy_share"] <= thr)

    with open(out / "e5_flags.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for split in ("test", "train"):
        srows = [r for r in rows if r["split"] == split]
        es = np.array([r["energy_share"] for r in srows])
        fl = [r for r in srows if r["flagged"]]
        summary[split] = {"n_tp": len(srows), "flagged": len(fl),
                          "thr_energy": float(np.quantile(es, 0.10)) if len(es) else None,
                          "median_energy": float(np.median(es)) if len(es) else None}
        worst = sorted(fl, key=lambda r: r["energy_share"])[:args.montage]
        tiles = []
        for r in worst:
            img = cv2.imread(str(data / split / "images" / r["image"]))
            if img is None:
                continue
            img = cv2.resize(img, (300, int(300 * img.shape[0] / img.shape[1])))
            img = img[:300]
            if img.shape[0] < 300:
                img = cv2.copyMakeBorder(img, 0, 300 - img.shape[0], 0, 0,
                                         cv2.BORDER_CONSTANT, value=0)
            cv2.putText(img, f"{r['energy_share']:.3f}", (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            tiles.append(img)
        if tiles:
            cols = 6
            nrows = (len(tiles) + cols - 1) // cols
            sheet = np.zeros((nrows * 300, cols * 300, 3), np.uint8)
            for i, t in enumerate(tiles):
                sheet[(i // cols) * 300:(i // cols + 1) * 300,
                      (i % cols) * 300:(i % cols + 1) * 300] = t[:300, :300]
            cv2.imwrite(str(out / f"e5_worst_{split}.png"), sheet)
    (out / "e5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
