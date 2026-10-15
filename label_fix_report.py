#!/usr/bin/env python3
"""Report test metrics with and without the two objectively-wrong test boxes
(ds0_0801 box on tree trunk, ds0_0883 box on fence — E4 hand review,
independently confirmed by E5: both are the only zero-energy test images and
the model matches nothing in them).

Protocol (plan: test labels fixed only when objectively wrong, numbers reported
with and without the fix): build records ONCE at conf 0.02 with the original
labels (arm-A min conf), then blank the two label files and re-evaluate the
frozen arm settings read from paper_results/twostage/s0/test_report.json.
Outputs paper_results/label_fix_report.json.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import two_stage_eval as ts  # noqa: E402

DATA = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")
YOLO = ROOT / "runs/detect/runs/detect/landmine_yolov8s_75ep_s0_v2/weights/last.pt"
CLS = ROOT / "paper_results/stage2/s2_ratio1_s0/best.pt"
REPORT = ROOT / "paper_results/twostage/s0/test_report.json"
OUT = ROOT / "paper_results/label_fix_report.json"

FIX_IMAGES = ["ds0_0801_jpg.rf.7fec8a048a84003b9dcad3f5ee1eb201.jpg",
              "ds0_0883_jpg.rf.5a0b82d51317a21637b5f97493e77bcf.jpg"]


def main() -> int:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    conf_a = rep["arm_a_conf"]
    conf_b = rep["arm_b_conf"]
    c = rep["arms"]["C_two_stage"]
    conf_c, thr_c, prot_c = c["yolo_conf"], c["cls_thr"], c["protect_conf"]

    recs = ts.build_records(str(YOLO), str(CLS), str(DATA / "test" / "images"),
                            str(DATA / "test" / "labels"), conf=conf_a)
    # sanity: baseline must reproduce the frozen report
    base_c = ts.evaluate(recs, conf_c, thr_c, prot_c)
    assert abs(base_c["precision"] - c["precision"]) < 1e-6, (base_c["precision"], c["precision"])
    assert abs(base_c["recall"] - c["recall"]) < 1e-6, (base_c["recall"], c["recall"])

    fixed = copy.deepcopy(recs)
    n_removed = 0
    for r in fixed:
        if r["name"] in FIX_IMAGES:
            n_removed += len(r["gt"])
            r["gt"] = []
    assert n_removed == 2, n_removed

    arms = {"A_yolo_only_deploy": conf_a, "B_yolo_only_matched_recall": conf_b,
            "C_two_stage": None}
    out = {"fix_images": FIX_IMAGES, "boxes_removed": n_removed, "arms": {}}
    for arm, conf in arms.items():
        if arm.startswith("C"):
            before = ts.evaluate(recs, conf_c, thr_c, prot_c)
            after = ts.evaluate(fixed, conf_c, thr_c, prot_c)
        else:
            before = ts.evaluate(recs, conf, None, 1.01)
            after = ts.evaluate(fixed, conf, None, 1.01)
        out["arms"][arm] = {k: {"before": before[k], "after": after[k]}
                            for k in ("precision", "recall", "f2", "fp_per_image",
                                      "img_recall", "img_precision")}
        print(arm, json.dumps(out["arms"][arm], indent=1))
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
