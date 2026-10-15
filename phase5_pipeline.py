#!/usr/bin/env python3
"""Phase 5.3 — two-stage pipeline evaluation (plan §Phase 5.3).

Thresholds chosen on VAL only; test evaluated ONCE under the frozen choice.
Arms on test:
  A: YOLO alone at its Phase-3 deployment OP conf (--arm-a-conf from phase3 val.json)
  B: YOLO alone at the conf matching the two-stage VAL recall (ts.yolo_only_at_recall)
  C: two-stage (proposals @ --conf + classifier threshold from the val sweep)
Val sweep: tau in {0..1 step 0.01} x P in {0.4, 0.6, 0.8, none}; per plan, largest
tau within --max-recall-drop of stage-1 recall; across P choose lowest val FP/img.

Outputs under --out:
  sweep_val.csv, val_report.json, test_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import two_stage_eval as ts  # noqa: E402


def size_tercile_recall(records, yolo_conf, cls_thr, protect_conf, iou_thr=0.5):
    """GT recall by image-area tercile; each det greedily claims best unmatched GT."""
    areas, matched = [], []
    for r in records:
        if not r["gt"]:
            continue
        kept = [d for d in r["dets"]
                if ts.keep(d, yolo_conf, cls_thr, protect_conf)]
        order = sorted(range(len(kept)), key=lambda i: -kept[i]["conf"])
        hit = [False] * len(r["gt"])
        for i in order:
            best, bi = -1, -1
            for j, g in enumerate(r["gt"]):
                if hit[j]:
                    continue
                v = ts.iou_xyxy(kept[i]["box"], g)
                if v > best:
                    best, bi = v, j
            if bi >= 0 and best >= iou_thr:
                hit[bi] = True
        for g, h in zip(r["gt"], hit):
            x1, y1, x2, y2 = g
            areas.append((x2 - x1) * (y2 - y1))
            matched.append(h)
    if not areas:
        return {}, None
    areas = np.asarray(areas)
    matched = np.asarray(matched, dtype=bool)
    q1, q2 = np.quantile(areas, [1 / 3, 2 / 3])
    out = {}
    for lo, hi, name in ((-np.inf, q1, "small"), (q1, q2, "mid"), (q2, np.inf, "large")):
        m = (areas > lo) & (areas <= hi)
        out[name] = {"n": int(m.sum()),
                     "recall": float(matched[m].mean()) if m.any() else None}
    return out, [float(q1), float(q2)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", required=True, help="YOLO weights (chosen ckpt, e.g. last.pt)")
    ap.add_argument("--cls", required=True, help="stage-2 best.pt (MineClassifier state_dict)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.05, help="YOLO proposal conf")
    ap.add_argument("--arm-a-conf", type=float, default=None,
                    help="Phase-3 deployment OP conf (primary_op_conf from phase3 val.json)")
    ap.add_argument("--protect-grid", default="0.4,0.6,0.8,none")
    ap.add_argument("--max-recall-drop", type=float, default=0.005)
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--val-only", action="store_true", help="dry run: stop after the val sweep")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def build(split, build_conf):
        d = data / split
        t0 = time.time()
        rec = ts.build_records(args.yolo, args.cls, d / "images", d / "labels",
                               conf=build_conf, crop_size=args.crop_size)
        return rec, (time.time() - t0) / max(len(rec), 1)

    # build at the lowest conf we will ever evaluate, so no band (e.g. arm A at
    # 0.02) is silently clipped out of the records (keep() does the real filter)
    arm_a_pre = args.arm_a_conf if args.arm_a_conf is not None else args.conf
    build_conf_val = min(args.conf, arm_a_pre)

    print(f"building val records (conf>={build_conf_val})...", flush=True)
    val, val_t = build("valid", build_conf_val)
    print(f"val: {len(val)} imgs, {val_t * 1000:.0f} ms/img", flush=True)

    s1 = ts.evaluate(val, args.conf)
    s1_recall = s1["recall"]
    print(f"stage1 YOLO@{args.conf}: img recall={s1_recall:.4f} "
          f"fp/img={s1['fp_per_image']:.3f}", flush=True)

    protect_grid = [None if p in ("none", "None") else float(p)
                    for p in args.protect_grid.split(",")]
    rows, candidates = [], []
    for P in protect_grid:
        protect = 1.01 if P is None else P
        tau = ts.pick_cls_threshold(val, args.conf, protect, args.max_recall_drop)
        m = ts.evaluate(val, args.conf, tau, protect)
        row = {"P": "none" if P is None else P, "tau": tau,
               "recall": m["recall"], "precision": m["precision"],
               "fp_per_image": m["fp_per_image"],
               "img_recall": m["img_recall"],
               "recall_drop": s1_recall - m["recall"]}
        rows.append(row)
        candidates.append(row)
        print(f"  P={row['P']} tau={tau} recall={m['recall']:.4f} "
              f"fp/img={m['fp_per_image']:.3f}", flush=True)

    with open(out / "sweep_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in candidates if r["recall_drop"] <= args.max_recall_drop + 1e-9]
    if not ok:
        ok = [max(candidates, key=lambda r: r["recall"])]
    chosen = min(ok, key=lambda r: (r["fp_per_image"], -float(r["P"] if r["P"] != "none" else 0)))
    protect = 1.01 if chosen["P"] == "none" else float(chosen["P"])
    tau = float(chosen["tau"])

    val_two = ts.evaluate(val, args.conf, tau, protect)
    b_val = ts.yolo_only_at_recall(val, val_two["recall"])
    arm_b = b_val["yolo_conf"] if b_val else args.conf
    arm_a = args.arm_a_conf if args.arm_a_conf is not None else args.conf
    if arm_b < build_conf_val:  # rare: rebuild val so arm B's band exists
        val, val_t = build("valid", arm_b)
        val_two = ts.evaluate(val, args.conf, tau, protect)
        b_val = ts.yolo_only_at_recall(val, val_two["recall"])
        arm_b = b_val["yolo_conf"] if b_val else args.conf
    print(f"chosen P={chosen['P']} tau={tau}; armA conf={arm_a} armB conf={arm_b}",
          flush=True)

    (out / "val_report.json").write_text(json.dumps({
        "stage1": s1, "chosen": chosen,
        "chosen_effective": {"yolo_conf": args.conf, "cls_thr": tau, "protect_conf": protect},
        "arm_a_conf": arm_a, "arm_b_conf": arm_b,
        "ms_per_image": val_t * 1000,
        "val_at_chosen": val_two,
        "val_at_arm_a": ts.evaluate(val, arm_a),
        "val_at_arm_b": ts.evaluate(val, arm_b),
    }, indent=2), encoding="utf-8")

    if args.val_only:
        print("--val-only: stopping before any test evaluation")
        return 0

    print(f"building test records (conf>={min(args.conf, arm_a, arm_b)}; single test eval)...",
          flush=True)
    test, test_t = build("test", min(args.conf, arm_a, arm_b))
    print(f"test: {len(test)} imgs, {test_t * 1000:.0f} ms/img", flush=True)

    cfg_a = {"yolo_conf": arm_a}
    cfg_b = {"yolo_conf": arm_b}
    cfg_c = {"yolo_conf": args.conf, "cls_thr": tau, "protect_conf": protect}
    arms = {
        "A_yolo_only_deploy": {**cfg_a, **ts.evaluate(test, arm_a),
                               **ts.bootstrap(test, arm_a)},
        "B_yolo_only_matched_recall": {**cfg_b, **ts.evaluate(test, arm_b),
                                       **ts.bootstrap(test, arm_b)},
        "C_two_stage": {**cfg_c, **ts.evaluate(test, args.conf, tau, protect),
                        **ts.bootstrap(test, args.conf, tau, protect)},
    }
    paired = ts.paired_bootstrap_diff(test, cfg_c, cfg_b)

    terciles = {}
    for name, conf, thr, prot in (("A", arm_a, None, 1.01), ("B", arm_b, None, 1.01),
                                  ("C", args.conf, tau, protect)):
        tr, bounds = size_tercile_recall(test, conf, thr, prot)
        terciles[name] = {"recall_by_tercile": tr, "area_bounds_px2": bounds}

    crops_n = np.array([len(r["dets"]) for r in test])
    report = {
        "chosen_on_val": chosen,
        "chosen_effective": {"yolo_conf": args.conf, "cls_thr": tau, "protect_conf": protect},
        "arm_a_conf": arm_a, "arm_b_conf": arm_b,
        "arms": arms,
        "paired_diff_C_minus_B": paired,
        "size_terciles": terciles,
        "cost": {"ms_per_image": test_t * 1000,
                 "crops_per_image_mean": float(crops_n.mean()),
                 "crops_per_image_p95": float(np.percentile(crops_n, 95))},
        "yolo_only_curve": ts.yolo_only_curve(test)[::5],
    }
    (out / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"arms": {k: {x: v[x] for x in
                                   ("precision", "recall", "fp_per_image", "img_recall")}
                               for k, v in arms.items()},
                      "paired_diff_C_minus_B": paired, "cost": report["cost"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
