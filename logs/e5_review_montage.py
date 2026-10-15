#!/usr/bin/env python3
"""E5 hand-review aid: draw GT boxes (yellow) on the worst-flagged images from
e5_flags.csv -> e5_worst_{split}_annotated.png. Review aid only, no paper numbers."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import two_stage_eval as ts  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
DATA = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")
GC = ROOT / "paper_results" / "gradcam"
N = 12

rows = list(csv.DictReader(open(GC / "e5_flags.csv", encoding="utf-8")))
for split in ("test", "train"):
    fl = sorted((r for r in rows if r["split"] == split and r["flagged"] == "1"),
                key=lambda r: float(r["energy_share"]))[:N]
    tiles = []
    for r in fl:
        p = DATA / split / "images" / r["image"]
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = ts._read_gt(str(DATA / split / "labels" / (p.name.rsplit(".", 1)[0] + ".txt")), w, h)
        for x1, y1, x2, y2 in gt:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        img = cv2.resize(img, (360, int(360 * h / w)))
        cv2.putText(img, f"{float(r['energy_share']):.3f} ngt={r['n_gt']}", (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(img)
    cols = 4
    th = max(t.shape[0] for t in tiles)
    nrows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((nrows * th, cols * 360, 3), np.uint8)
    for i, t in enumerate(tiles):
        sheet[(i // cols) * th:(i // cols) * th + t.shape[0],
              (i % cols) * 360:(i % cols + 1) * 360] = t
    out = GC / f"e5_worst_{split}_annotated.png"
    cv2.imwrite(str(out), sheet)
    print("wrote", out)
