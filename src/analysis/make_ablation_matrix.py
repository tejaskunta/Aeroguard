#!/usr/bin/env python3
"""Stage-2 ablation matrix (Block B).

Builds crops, trains the crop classifier, and dumps raw val/test records for
each ablation arm. Inference-side rules (P threshold, arm selection) are
computed later from these records by analysis_stats.py, so this script only
does the GPU work that actually requires a retrain.

Variants:
  pad0        context margin 0.0 (vs 0.2)
  pad04       context margin 0.4
  crop160     classifier input 160x160 (vs 224)
  crop320     classifier input 320x320
  nohardneg   no YOLO false-positive negative mining

The YOLO detector is held fixed at the seed-0 operational checkpoint so the
only thing that changes is the stage-2 configuration.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "src" / "common"))
sys.stdout.reconfigure(encoding="utf-8")

V2 = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")
RUN_S0 = ROOT / "runs/detect/runs/detect/landmine_yolov8s_75ep_s0_v2"
EPOCH20 = RUN_S0 / "weights/epoch20.pt"
YOLO_LAST = RUN_S0 / "weights/last.pt"

VARIANTS = [
    ("pad0", 0.0, 224, True),
    ("pad04", 0.4, 224, True),
    ("crop160", 0.2, 160, True),
    ("crop320", 0.2, 320, True),
    ("nohardneg", 0.2, 224, False),
]

PY = sys.executable


def run(cmd: list[str], log) -> None:
    log.write("$ " + " ".join(cmd) + "\n")
    log.flush()
    subprocess.run(cmd, cwd=str(ROOT), check=True, stdout=log, stderr=subprocess.STDOUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated variant names")
    ap.add_argument("--force-build", action="store_true")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    want = {s for s in args.only.split(",") if s} or {v[0] for v in VARIANTS}

    for name, pad, size, hard in VARIANTS:
        if name not in want:
            continue
        out = ROOT / "paper_results/ablation" / name
        crops = out / "crops"
        logp = ROOT / "logs" / f"ablation_{name}.log"
        logp.parent.mkdir(parents=True, exist_ok=True)
        with open(logp, "a", encoding="utf-8") as log:
            log.write(f"\n===== {name} pad={pad} size={size} hardneg={hard} =====\n")
            (out).mkdir(parents=True, exist_ok=True)
            (out / "meta.json").write_text(json.dumps(
                {"variant": name, "pad": pad, "crop_size": size, "hardneg": hard,
                 "yolo": str(YOLO_LAST), "fp_miner": str(EPOCH20) if hard else None},
                indent=2), encoding="utf-8")

            if args.force_build or not (crops / "manifest.csv").exists():
                cmd = [PY, str(ROOT / "src" / "pipeline" / "phase5_build_crops.py"), "--data", str(V2), "--out", str(crops),
                       "--pad", str(pad), "--imgsz", str(size), "--seed", "0"]
                if hard:
                    cmd += ["--yolo", str(EPOCH20)]
                run(cmd, log)

            best = out / "cls" / "best.pt"
            if not args.skip_train and (args.force_train or not best.exists()):
                run([PY, str(ROOT / "src" / "training" / "phase5_train_stage2.py"), "--crops", str(crops),
                     "--out", str(out / "cls"), "--ratio", "1", "--seed", "0",
                     "--epochs", str(args.epochs)], log)

            for split, fn in (("valid", "raw_preds_val.json"), ("test", "raw_preds_test.json")):
                target = out / fn
                if args.force_build or not target.exists():
                    run([PY, str(ROOT / "src" / "analysis" / "dump_raw_stage2.py"), "--yolo", str(YOLO_LAST),
                         "--cls", str(best), "--out", str(out), "--conf", "0.02",
                         "--pad", str(pad), "--crop-size", str(size),
                         "--splits", split], log)
            log.write(f"===== {name} done =====\n")
        print(f"{name}: {logp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
