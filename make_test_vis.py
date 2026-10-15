"""Visualization-only pass on the frozen test split (and val for batch grids).

Writes per-seed:
  paper_results/vis/s_seed{N}/val/   — ultralytics val plots (curves, CM, batch GT+pred)
  paper_results/vis/s_seed{N}/test/  — same on test at the seed's frozen OP conf

NO metric/threshold decisions are made here. Official numbers stay in
paper_results/yolo/s_seed{N}/{val,test}.json (already frozen). This only
produces images for the PDF reports. Logged in RUN_LOG as visualization-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"C:\Users\Delta\Downloads\SafeMineAI-main")
DATA = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")
OUT = ROOT / "paper_results" / "vis"

from ultralytics import YOLO


def main() -> None:
    for seed in (0, 1, 2):
        run_dir = ROOT / "runs" / "detect" / "runs" / "detect" / f"landmine_yolov8s_75ep_s{seed}_v2"
        weights = run_dir / "weights" / "best.pt"
        test = json.load(open(ROOT / "paper_results" / "yolo" / f"s_seed{seed}" / "test.json", encoding="utf-8"))
        op_conf = float(test["ops"]["op97_primary"]["conf"])
        model = YOLO(str(weights))

        for split, conf in (("val", 0.25), ("test", op_conf)):
            out = OUT / f"s_seed{seed}" / split
            out.mkdir(parents=True, exist_ok=True)
            print(f"seed {seed} {split} conf={conf} -> {out}")
            model.val(
                data=str(DATA),
                split=split,
                imgsz=640,
                conf=conf,
                save=True,
                plots=True,
                project=str(out),
                name="vis",
                exist_ok=True,
                verbose=False,
            )
    print("done")


if __name__ == "__main__":
    main()
