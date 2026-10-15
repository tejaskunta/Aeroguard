#!/usr/bin/env python3
"""Rule 11 (raw outputs): dump per-image stage-2 records (boxes, conf, cls_p,
gt) for val and test at conf 0.02 (arm-A min conf) ->
paper_results/stage2/s2_ratio1_s0/raw_preds_{val,test}.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import two_stage_eval as ts  # noqa: E402

DATA = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")
YOLO = ROOT / "runs/detect/runs/detect/landmine_yolov8s_75ep_s0_v2/weights/last.pt"
CLS = ROOT / "paper_results/stage2/s2_ratio1_s0/best.pt"
OUTD = ROOT / "paper_results/stage2/s2_ratio1_s0"

for split, out_name in (("valid", "raw_preds_val.json"), ("test", "raw_preds_test.json")):
    recs = ts.build_records(str(YOLO), str(CLS), str(DATA / split / "images"),
                            str(DATA / split / "labels"), conf=0.02)
    p = OUTD / out_name
    p.write_text(json.dumps(recs), encoding="utf-8")
    n_det = sum(len(r["dets"]) for r in recs)
    print(split, "images", len(recs), "dets", n_det, "->", p)
print("done")
