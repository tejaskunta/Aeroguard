#!/usr/bin/env python3
"""Phase 3 evaluation — every number for plan §Phase 3.3–3.8, from saved files.

Protocol (plan §3): thresholds chosen on VAL only; test evaluated once with the
frozen choices; every test touch logged in RUN_LOG.md. Uses the two_stage_eval
matching/metrics/bootstrap core (the same code the paper's two-stage arm uses).

Outputs under --out:
  raw_preds_val_{best,last}.json, raw_preds_test.json   (conf 0.001, ground rule 11)
  val.json        mAP (Ultralytics conf 0.001), P/R/F2/FP-img @0.10 & @0.25,
                  recall-first OPs (0.95/0.97/0.98), best-vs-last choice
  test.json       mAP, metrics at OPs, image-level metrics, per-source,
                  bootstrap 95% CIs (1000 image resamples, seed 0)
  pr_curve_val.csv, pr_curve_test.csv                   (conf sweep at IoU 0.5)
  size_latency.json                                     (size table inputs)
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import two_stage_eval as ts  # noqa: E402


# ------------------------------------------------------------------ helpers

def source_of(name: str) -> str:
    if name.startswith("ds0"):
        return "ds0"
    if name.startswith("ds1"):
        return "ds1"
    if "Jan" in name:
        return "Jan_lwir"
    if "May" in name:
        return "May_lwir"
    return "other"


def split_paths(data_root: Path, split: str):
    img = data_root / split / "images"
    lbl = data_root / split / "labels"
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    names = sorted(p.name for p in img.iterdir() if p.name.lower().endswith(exts))
    return [img / n for n in names], [lbl / (n.rsplit(".", 1)[0] + ".txt") for n in names], names


def predict_records(yolo, paths, labels, names, conf=0.001, imgsz=640, iou=0.7, max_det=300):
    records = []
    for p, lp, n in zip(paths, labels, names):
        w_img = None
        res = yolo.predict(str(p), conf=conf, imgsz=imgsz, iou=iou, max_det=max_det,
                           verbose=False)[0]
        h, w = res.orig_shape
        gt = ts._read_gt(str(lp), w, h)
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        records.append({"name": n, "gt": gt,
                        "dets": [{"box": [float(v) for v in b], "conf": float(c)}
                                 for b, c in zip(boxes, confs)]})
        del w_img
    return records


def ultralytics_map(yolo, data_yaml, split):
    # Ultralytics split names are train/val/test; the folder is called valid/
    # but data.yaml maps it under the key `val`.
    split = {"valid": "val"}.get(split, split)
    m = yolo.val(data=str(data_yaml), split=split, conf=0.001, imgsz=640, iou=0.7,
                 max_det=300, plots=False, verbose=False)
    return {"map50": float(m.box.map50), "map50_95": float(m.box.map),
            "precision_mean": float(m.box.mp), "recall_mean": float(m.box.mr),
            "conf_used": 0.001, "imgsz": 640, "nms_iou": 0.7, "max_det": 300}


def metrics_at(records, conf):
    m = ts.evaluate(records, conf)
    out = {k: m[k] for k in ("tp", "fp", "fn", "precision", "recall", "f2",
                             "fp_per_image", "img_accuracy", "img_precision", "img_recall")}
    return out


def image_level(records, conf, iou_thr=0.5):
    c = ts.per_image_counts(records, conf, None, 1.01, iou_thr)
    m = ts.metrics_from_counts(c)
    itp, ifp, ifn, itn = m["img_tp"], m["img_fp"], m["img_fn"], m["img_tn"]
    total = itp + ifp + ifn + itn
    acc = (itp + itn) / total if total else 0.0
    pos, neg = itp + ifn, ifp + itn
    bal = 0.5 * (itp / pos if pos else 0.0) + 0.5 * (itn / neg if neg else 0.0)
    pe = ((pos / total) * ((itp + ifp) / total) + (neg / total) * ((itn + ifn) / total)) if total else 0.0
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    return {"confusion": {"tp": itp, "fp": ifp, "fn": ifn, "tn": itn},
            "accuracy": acc, "balanced_accuracy": bal, "kappa": kappa,
            "precision": m["img_precision"], "recall": m["img_recall"]}


def recall_first_ops(records, targets=(0.95, 0.97, 0.98), step=0.005):
    out = {}
    for t in targets:
        best = None
        c = 1.0
        while c >= -1e-9:
            if ts.evaluate(records, round(c, 3))["recall"] >= t:
                best = round(c, 3)
                break
            c -= step
        out[str(t)] = best
    return out


def pr_curve(records, n=101):
    rows = []
    for c in np.linspace(0.0, 1.0, n):
        m = ts.evaluate(records, float(c))
        rows.append((round(float(c), 4), m["precision"], m["recall"], m["fp_per_image"]))
    return rows


def per_source(records, conf):
    out = {}
    by_src = {}
    for r in records:
        by_src.setdefault(source_of(r["name"]), []).append(r)
    for src, rs in sorted(by_src.items()):
        m = ts.evaluate(rs, conf)
        out[src] = {"n_images": len(rs), "precision": m["precision"], "recall": m["recall"],
                    "fp_per_image": m["fp_per_image"], "tp": m["tp"], "fp": m["fp"],
                    "fn": m["fn"]}
    return out


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def save_records(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")
    print(f"wrote {path} ({len(records)} images)")


def size_latency(yolo, sample_path, imgsz=640, warmup=10, reps=50):
    import torch
    model = yolo.model
    params = sum(p.numel() for p in model.parameters())
    try:
        info = yolo.info(verbose=False, detailed=False)
        flops = float(info[3]) if info and len(info) > 3 else None
    except Exception:
        flops = None
    wsize = None
    w = getattr(yolo, "ckpt_path", None) or None
    if w and Path(w).exists():
        wsize = Path(w).stat().st_size / (1024 * 1024)
    times = []
    use_cuda = torch.cuda.is_available()
    for i in range(warmup + reps):
        yolo.predict(str(sample_path), conf=0.25, imgsz=imgsz, verbose=False)
        if i == warmup - 1 and use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yolo.predict(str(sample_path), conf=0.25, imgsz=imgsz, verbose=False)
        if use_cuda:
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"params": params, "gflops": flops, "weights_mb": wsize,
            "latency_ms_median": statistics.median(times),
            "latency_ms_p95": times[int(0.95 * (len(times) - 1))],
            "imgsz": imgsz, "fp": "FP32 PC GPU", "reps": reps, "warmup": warmup}


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing weights/")
    ap.add_argument("--data", required=True, help="v2 dataset root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--val-only", action="store_true",
                    help="stop after val.json (dry run; does not touch test)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    random.seed(0)
    np.random.seed(0)

    run = Path(args.run)
    data_root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_yaml = data_root / "data.yaml"

    from ultralytics import YOLO

    val_paths, val_labels, val_names = split_paths(data_root, "valid")
    test_paths, test_labels, test_names = split_paths(data_root, "test")

    # ---- raw val predictions for BOTH checkpoints (val-only decision material)
    raw = {}
    for ckpt in ("best", "last"):
        w = run / "weights" / f"{ckpt}.pt"
        yolo = YOLO(str(w))
        print(f"raw val predictions @conf 0.001 — {ckpt}", flush=True)
        raw[ckpt] = predict_records(yolo, val_paths, val_labels, val_names)
        save_records(out / f"raw_preds_val_{ckpt}.json", raw[ckpt])

    # ---- checkpoint choice + OPs on val only (plan §3.7, §3.4)
    op_search = {ck: recall_first_ops(raw[ck]) for ck in ("best", "last")}
    target = "0.97"
    choice = {}
    for ck in ("best", "last"):
        thr = op_search[ck][target] if op_search[ck][target] is not None else op_search[ck]["0.95"]
        if thr is None:
            thr = 0.25
        m = ts.evaluate(raw[ck], thr)
        choice[ck] = {"op_conf": thr, "val_precision": m["precision"],
                      "val_recall": m["recall"], "val_fp_per_image": m["fp_per_image"]}
    # pick: reaches the 0.97 target on val with best precision; tie -> higher conf
    reached = {ck: v for ck, v in choice.items() if v["val_recall"] >= 0.97 - 1e-9}
    pool = reached or choice
    chosen = max(pool, key=lambda ck: (pool[ck]["val_precision"], pool[ck]["op_conf"]))
    primary_conf = choice[chosen]["op_conf"]
    print(f"checkpoint choice (val only): {chosen} @ conf {primary_conf}", flush=True)

    val_json = {"ultralytics_map": {}, "at_conf": {}, "recall_first_ops": op_search,
                "checkpoint_choice": {"chosen": chosen, "rule": "val recall>=0.97 then max precision",
                                      **{k: choice[k] for k in ("best", "last")},
                                      "primary_op_conf": primary_conf}}

    # ---- val metrics for chosen ckpt
    yolo = YOLO(str(run / "weights" / f"{chosen}.pt"))
    val_json["ultralytics_map"]["val"] = ultralytics_map(yolo, data_yaml, "valid")
    for c in (0.10, 0.25):
        val_json["at_conf"][str(c)] = metrics_at(raw[chosen], c)
    val_json["image_level_at_op"] = image_level(raw[chosen], primary_conf)
    save_json(out / "val.json", val_json)

    if args.val_only:
        print("--val-only: stopping before any test evaluation")
        return 0

    # ---- test: ONCE, chosen ckpt + frozen OPs
    print("raw test predictions @conf 0.001 (test touched once)", flush=True)
    test_records = predict_records(yolo, test_paths, test_labels, test_names)
    save_records(out / "raw_preds_test.json", test_records)

    test_json = {"ultralytics_map": {"test": ultralytics_map(yolo, data_yaml, "test")},
                 "at_conf": {}, "chosen_on_val": {"checkpoint": chosen,
                                                  "primary_op_conf": primary_conf,
                                                  **op_search}}
    for c in (0.10, 0.25):
        test_json["at_conf"][str(c)] = metrics_at(test_records, c)
    test_json["ops"] = {}
    for name, conf in [("op95", op_search[chosen]["0.95"]),
                       ("op97_primary", primary_conf),
                       ("op98", op_search[chosen]["0.98"])]:
        if conf is None:
            test_json["ops"][name] = None
            continue
        m = ts.evaluate(test_records, conf)
        ci = ts.bootstrap(test_records, conf, n_boot=1000, seed=0)
        test_json["ops"][name] = {"conf": conf, **m, **ci,
                                  "image_level": image_level(test_records, conf)}
    test_json["per_source_at_op"] = per_source(test_records, primary_conf)
    save_json(out / "test.json", test_json)

    # ---- PR curves
    for split_name, recs in (("val", raw[chosen]), ("test", test_records)):
        with open(out / f"pr_curve_{split_name}.csv", "w", encoding="utf-8") as f:
            f.write("conf,precision,recall,fp_per_image\n")
            for row in pr_curve(recs):
                f.write(",".join(str(v) for v in row) + "\n")
        print(f"wrote pr_curve_{split_name}.csv")

    # ---- size table inputs
    save_json(out / "size_latency.json", size_latency(yolo, str(val_paths[0]), args.imgsz))
    print("phase3_eval done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
