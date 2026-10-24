#!/usr/bin/env python3
"""Rule 11 (raw outputs): dump per-image stage-2 records (boxes, conf, cls_p,
gt) for val and test at a fixed proposal conf -> <out>/raw_preds_{val,test}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import two_stage_eval as ts  # noqa: E402

DATA = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default=str(
        ROOT / "runs/detect/runs/detect/landmine_yolov8s_75ep_s0_v2/weights/last.pt"))
    ap.add_argument("--cls", default=str(ROOT / "paper_results/stage2/s2_ratio1_s0/best.pt"))
    ap.add_argument("--out", default=str(ROOT / "paper_results/stage2/s2_ratio1_s0"))
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--pad", type=float, default=0.2)
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--splits", default="valid,test")
    args = ap.parse_args()
    outd = Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)
    names = {"valid": "raw_preds_val.json", "test": "raw_preds_test.json"}
    for split in args.splits.split(","):
        recs = ts.build_records(args.yolo, args.cls, str(DATA / split / "images"),
                                str(DATA / split / "labels"), conf=args.conf,
                                pad=args.pad, crop_size=args.crop_size)
        p = outd / names[split]
        p.write_text(json.dumps(recs), encoding="utf-8")
        print(split, "images", len(recs), "dets", sum(len(r["dets"]) for r in recs), "->", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
