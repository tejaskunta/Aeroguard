"""A4 diagnostic (POST-HOC, not a replacement for the primary result).

Question
--------
Seed 4's two-stage arm looks worse than its own YOLO-only baseline
(C precision 0.677 vs B 0.729, FP/img 0.619 vs 0.481). The seeded suspicion is
that the comparison is unfair: the two-stage arm consumes stage-1 proposals at
conf 0.05, while arm B is YOLO-only at conf 0.125. The classifier is therefore
asked to clean up a pool that is 2.5x larger and much dirtier than the pool the
baseline ever sees.

This script asks a narrow question: if stage 1 proposes at conf 0.125 (the same
conf arm B uses) and the classifier threshold is re-chosen on VAL under the same
rule as the primary experiment (largest tau within --max-recall-drop of stage-1
recall, then lowest val FP/img across the protect grid), does the two-stage arm
still lose to arm B on test?

Provenance
----------
Everything here is recomputed from already-saved raw predictions
(`paper_results/stage2/s2_ratio1_s4/raw_preds_{val,test}.json`, built at conf
0.02 with per-proposal cls_p). No model is run, no test image is re-read, and no
threshold is chosen on test. Detections with conf in [0.02, 0.05) that the
primary run never scored are simply excluded by `keep()` at conf 0.125, so the
recomputation is exact for the conf-0.125 setting.

WARNING: this is a post-hoc diagnostic. Its config was chosen after seeing the
primary result, so it cannot replace the primary number and must be reported as
such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))
import two_stage_eval as ts  # noqa: E402

S2 = ROOT / "paper_results" / "stage2" / "s2_ratio1_s4"
OUT = ROOT / "paper_results" / "analysis"

STAGE1_CONF = 0.125  # arm B's conf, instead of the primary 0.05
PROTECT_GRID = [0.4, 0.6, 0.8, None]
MAX_RECALL_DROP = 0.005  # same rule as the primary run


def jload(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    val = jload(S2 / "raw_preds_val.json")
    test = jload(S2 / "raw_preds_test.json")

    s1_val = ts.evaluate(val, STAGE1_CONF)
    s1_test = ts.evaluate(test, STAGE1_CONF)
    print(f"stage 1 @ conf {STAGE1_CONF}: "
          f"val R {s1_val['recall']:.4f} FP/img {s1_val['fp_per_image']:.3f} | "
          f"test R {s1_test['recall']:.4f} FP/img {s1_test['fp_per_image']:.3f}")

    # same val sweep as the primary run, only stage-1 conf differs
    candidates = []
    for P in PROTECT_GRID:
        protect = 1.01 if P is None else float(P)
        tau = ts.pick_cls_threshold(val, STAGE1_CONF, protect, MAX_RECALL_DROP)
        m = ts.evaluate(val, STAGE1_CONF, tau, protect)
        row = {"P": "none" if P is None else P, "tau": tau, "protect": protect,
               "val_recall": m["recall"], "val_precision": m["precision"],
               "val_fp_per_image": m["fp_per_image"],
               "recall_drop": s1_val["recall"] - m["recall"]}
        candidates.append(row)
        print(f"  P={row['P']:>4} tau={tau:.2f} val R {m['recall']:.4f} "
              f"P {m['precision']:.4f} FP/img {m['fp_per_image']:.3f}")

    ok = [r for r in candidates if r["recall_drop"] <= MAX_RECALL_DROP + 1e-9]
    chosen = min(ok, key=lambda r: (r["val_fp_per_image"],
                                    -float(r["P"] if r["P"] != "none" else 0)))
    tau, protect = float(chosen["tau"]), chosen["protect"]

    arm_c = ts.evaluate(test, STAGE1_CONF, tau, protect)
    arm_c_ci = ts.bootstrap(test, STAGE1_CONF, tau, protect)

    # arm B at the same stage-1 conf, matched to the *test* recall of arm C is NOT
    # allowed (that would pick on test). Reuse the primary run's arm B: it was chosen
    # on VAL for the primary config; here we additionally report plain YOLO-only at
    # 0.125, which is the identical operating point arm B used.
    arm_b_test = ts.evaluate(test, 0.125)

    primary = jload(ROOT / "paper_results" / "twostage" / "s4" / "test_report.json")

    out = {
        "kind": "post-hoc diagnostic (NOT the primary result)",
        "script": "src/analysis/a4_s4_conf_diagnostic.py",
        "seed": 4,
        "stage1_conf_primary": 0.05,
        "stage1_conf_here": STAGE1_CONF,
        "max_recall_drop": MAX_RECALL_DROP,
        "val_sweep": candidates,
        "chosen_on_val": chosen,
        "chosen_effective": {"yolo_conf": STAGE1_CONF, "cls_thr": tau,
                             "protect_conf": protect},
        "stage1_val": s1_val,
        "stage1_test": s1_test,
        "arm_c_test_here": {**arm_c, **arm_c_ci},
        "arm_b_test": arm_b_test,
        "paired_C_minus_B": ts.paired_bootstrap_diff(
            test,
            {"yolo_conf": STAGE1_CONF, "cls_thr": tau, "protect_conf": protect},
            {"yolo_conf": 0.125, "cls_thr": None, "protect_conf": 1.01}),
        "primary_arm_c_test": primary["arms"]["C_two_stage"],
        "primary_arm_b_test": primary["arms"]["B_yolo_only_matched_recall"],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "a4_s4_stage1_conf0125_diagnostic.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nchosen on val: P={chosen['P']} tau={tau:.2f}")
    print(f"arm C (conf {STAGE1_CONF}): P {arm_c['precision']:.4f} "
          f"R {arm_c['recall']:.4f} FP/img {arm_c['fp_per_image']:.3f}")
    print(f"arm B (conf 0.125):   P {arm_b_test['precision']:.4f} "
          f"R {arm_b_test['recall']:.4f} FP/img {arm_b_test['fp_per_image']:.3f}")
    print(f"paired C-B: {out['paired_C_minus_B']}")
    print(f"\nprimary arm C (conf 0.05): P {primary['arms']['C_two_stage']['precision']:.4f} "
          f"R {primary['arms']['C_two_stage']['recall']:.4f} "
          f"FP/img {primary['arms']['C_two_stage']['fp_per_image']:.3f}")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()