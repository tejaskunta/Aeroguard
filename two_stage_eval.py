"""Two-stage landmine detection evaluation.

Stage 1: YOLO proposes boxes at a LOW confidence (high recall, many false positives).
Stage 2: a crop classifier (e.g. MobileNetV3-Large) rejects proposals that look like background.

The pure-numpy core (matching, metrics, threshold selection, bootstrap) is separate from the
model glue so it can be unit-tested without weights or data.

Record format (one dict per image):
    {"name": str,
     "gt":   [[x1, y1, x2, y2], ...],                 # pixel xyxy
     "dets": [{"box": [x1,y1,x2,y2], "conf": float, "cls_p": float}, ...]}

Protocol (important for a paper):
    * choose the classifier threshold on VAL only
    * report on TEST once
    * compare against YOLO-only at MATCHED RECALL (threshold chosen on VAL), not at an arbitrary threshold
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


# ----------------------------------------------------------------------------- core

def iou_xyxy(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def keep(det, yolo_conf, cls_thr, protect_conf) -> bool:
    """cls_thr=None means stage 1 only. Detections with conf >= protect_conf bypass stage 2
    (recall-safe: the classifier can only veto borderline proposals)."""
    if det["conf"] < yolo_conf:
        return False
    if cls_thr is None or det["conf"] >= protect_conf:
        return True
    return det["cls_p"] >= cls_thr


def match_image(gt, dets, iou_thr=0.5):
    """COCO-style greedy matching: detections by confidence, each takes the best unmatched GT."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["conf"])
    taken = set()
    tp = 0
    for i in order:
        best, best_iou = -1, iou_thr
        for j, g in enumerate(gt):
            if j in taken:
                continue
            v = iou_xyxy(dets[i]["box"], g)
            if v >= best_iou:
                best, best_iou = j, v
        if best >= 0:
            taken.add(best)
            tp += 1
    fp = len(dets) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def per_image_counts(records, yolo_conf, cls_thr=None, protect_conf=1.01, iou_thr=0.5):
    """Array (N, 5): tp, fp, fn, gt_positive_image, predicted_positive_image."""
    out = np.zeros((len(records), 5), dtype=np.int64)
    for k, r in enumerate(records):
        kept = [d for d in r["dets"] if keep(d, yolo_conf, cls_thr, protect_conf)]
        tp, fp, fn = match_image(r["gt"], kept, iou_thr)
        out[k] = (tp, fp, fn, int(len(r["gt"]) > 0), int(len(kept) > 0))
    return out


def metrics_from_counts(c):
    tp, fp, fn = int(c[:, 0].sum()), int(c[:, 1].sum()), int(c[:, 2].sum())
    gt_pos, pred_pos = c[:, 3], c[:, 4]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    itp = int(((gt_pos == 1) & (pred_pos == 1)).sum())
    ifp = int(((gt_pos == 0) & (pred_pos == 1)).sum())
    ifn = int(((gt_pos == 1) & (pred_pos == 0)).sum())
    itn = int(((gt_pos == 0) & (pred_pos == 0)).sum())
    n = len(c)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f2": f2,
        "fp_per_image": fp / n if n else 0.0,
        "img_tp": itp, "img_fp": ifp, "img_fn": ifn, "img_tn": itn,
        "img_recall": itp / (itp + ifn) if itp + ifn else 0.0,
        "img_precision": itp / (itp + ifp) if itp + ifp else 0.0,
        "img_accuracy": (itp + itn) / n if n else 0.0,
    }


def evaluate(records, yolo_conf, cls_thr=None, protect_conf=1.01, iou_thr=0.5):
    return metrics_from_counts(per_image_counts(records, yolo_conf, cls_thr, protect_conf, iou_thr))


def sweep_cls(records, yolo_conf, protect_conf=1.01, thresholds=None, iou_thr=0.5):
    thresholds = np.linspace(0.0, 1.0, 101) if thresholds is None else thresholds
    return [{"cls_thr": float(t), **evaluate(records, yolo_conf, float(t), protect_conf, iou_thr)}
            for t in thresholds]


def pick_cls_threshold(val_records, yolo_conf, protect_conf=1.01, max_recall_drop=0.005, iou_thr=0.5):
    """Largest classifier threshold whose recall stays within `max_recall_drop` of stage 1.
    Fewer false positives for a bounded recall cost. Choose on VAL, never on TEST."""
    base = evaluate(val_records, yolo_conf, None, protect_conf, iou_thr)["recall"]
    best = 0.0
    for row in sweep_cls(val_records, yolo_conf, protect_conf, iou_thr=iou_thr):
        if row["recall"] >= base - max_recall_drop:
            best = max(best, row["cls_thr"])
    return best


def yolo_only_at_recall(records, target_recall, iou_thr=0.5):
    """Fair baseline: the strictest YOLO-only confidence threshold that still reaches
    `target_recall`. Without this, two-stage 'gains' may just be a higher YOLO threshold."""
    best = None
    for t in np.linspace(0.0, 1.0, 201):
        m = evaluate(records, float(t), None, 1.01, iou_thr)
        if m["recall"] >= target_recall:
            best = {"yolo_conf": float(t), **m}
    return best


def bootstrap(records, yolo_conf, cls_thr=None, protect_conf=1.01, n_boot=1000, seed=0, iou_thr=0.5):
    """95% CI by resampling IMAGES. Counts are precomputed so this is fast."""
    counts = per_image_counts(records, yolo_conf, cls_thr, protect_conf, iou_thr)
    rng = np.random.default_rng(seed)
    n = len(counts)
    rec, prec, fpi = [], [], []
    for _ in range(n_boot):
        m = metrics_from_counts(counts[rng.integers(0, n, n)])
        rec.append(m["recall"]); prec.append(m["precision"]); fpi.append(m["fp_per_image"])
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"recall_ci": q(rec), "precision_ci": q(prec), "fp_per_image_ci": q(fpi)}



def paired_bootstrap_diff(records, cfg_a, cfg_b, n_boot=1000, seed=0, iou_thr=0.5):
    """95% CI of (A - B) for recall, precision and FP/image, resampling the SAME images for both arms.
    cfg = dict(yolo_conf=..., cls_thr=None|float, protect_conf=float).
    Claim a gain only if the interval excludes zero (for FP/image: A - B below zero means A has fewer false alarms)."""
    ca = per_image_counts(records, cfg_a["yolo_conf"], cfg_a.get("cls_thr"), cfg_a.get("protect_conf", 1.01), iou_thr)
    cb = per_image_counts(records, cfg_b["yolo_conf"], cfg_b.get("cls_thr"), cfg_b.get("protect_conf", 1.01), iou_thr)
    rng = np.random.default_rng(seed)
    n = len(ca)
    out = {"recall": [], "precision": [], "fp_per_image": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ma, mb = metrics_from_counts(ca[idx]), metrics_from_counts(cb[idx])
        for k in out:
            out[k].append(ma[k] - mb[k])
    return {k: {"mean_diff": float(np.mean(v)),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in out.items()}


def yolo_only_curve(records, iou_thr=0.5, n=101):
    """FP/image vs recall for YOLO alone, for plotting with the two-stage point on top."""
    return [{"yolo_conf": float(t), **evaluate(records, float(t), None, 1.01, iou_thr)}
            for t in np.linspace(0.0, 1.0, n)]

# ----------------------------------------------------------------------------- model glue
# NOT unit-tested here (needs weights, images, torch, ultralytics). Run it and sanity-check
# a few crops by eye (--save-crops) before trusting any number.

def _read_gt(label_path, w, h):
    boxes = []
    if os.path.exists(label_path):
        for line in open(label_path):
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(x) for x in p[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def build_records(yolo_weights, cls_weights, images_dir, labels_dir, conf=0.05, pad=0.2,
                  imgsz=640, crop_size=224, device=None, save_crops=None):
    """Model glue: YOLO proposals + MobileNetV3-Large crop scores.

    Uses the shared constructor and normalisation path (plan §Phase 5.2 /
    ground rule 10): mnv3_shared.build_mnv3_large + mnv3_shared.preprocess_uint8.
    pad=0.2 matches the training-crop rule (20% context).
    """
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO

    import mnv3_shared

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(yolo_weights)
    clf = mnv3_shared.build_mnv3_large(pretrained=False)
    clf.load_state_dict(torch.load(cls_weights, map_location=device))
    clf.eval().to(device)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    names = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(exts))
    if save_crops:
        os.makedirs(save_crops, exist_ok=True)
    records = []
    for name in names:
        img = cv2.imread(os.path.join(images_dir, name))
        h, w = img.shape[:2]
        gt = _read_gt(os.path.join(labels_dir, os.path.splitext(name)[0] + ".txt"), w, h)
        res = yolo.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        crops = []
        for (x1, y1, x2, y2) in boxes:
            bw, bh = x2 - x1, y2 - y1
            side = max(bw, bh) * (1 + 2 * pad)          # square crop with context
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
            crop = cv2.resize(img[b:d, a:c], (crop_size, crop_size))
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if save_crops and len(crops) <= 3:
                cv2.imwrite(os.path.join(save_crops, f"{os.path.splitext(name)[0]}_{len(crops)}.jpg"), crop)
        probs = np.zeros(0)
        if crops:
            x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).contiguous()
            x = mnv3_shared.preprocess_uint8(x).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(clf(x)).squeeze(1).cpu().numpy().reshape(-1)
        records.append({
            "name": name, "gt": gt,
            "dets": [{"box": [float(v) for v in bx], "conf": float(cf), "cls_p": float(p)}
                     for bx, cf, p in zip(boxes, confs, probs)],
        })
    return records


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--cls", required=True, help="state_dict of mobilenet_v3_large with 1-logit head")
    ap.add_argument("--val-images", required=True); ap.add_argument("--val-labels", required=True)
    ap.add_argument("--test-images", required=True); ap.add_argument("--test-labels", required=True)
    ap.add_argument("--conf", type=float, default=0.05, help="stage-1 confidence (keep LOW)")
    ap.add_argument("--protect-conf", type=float, default=0.60,
                    help="proposals at/above this skip stage 2 (recall-safe)")
    ap.add_argument("--max-recall-drop", type=float, default=0.005)
    ap.add_argument("--out", default="two_stage_report.json")
    ap.add_argument("--save-crops", default=None, help="dump a few crops per image to eyeball")
    a = ap.parse_args()

    val = build_records(a.yolo, a.cls, a.val_images, a.val_labels, a.conf, save_crops=a.save_crops)
    thr = pick_cls_threshold(val, a.conf, a.protect_conf, a.max_recall_drop)   # VAL only
    test = build_records(a.yolo, a.cls, a.test_images, a.test_labels, a.conf)

    s1 = evaluate(test, a.conf)
    s2 = evaluate(test, a.conf, thr, a.protect_conf)
    # arm B: YOLO alone, threshold chosen on VAL to match the two-stage VAL recall, then applied to TEST
    val_two = evaluate(val, a.conf, thr, a.protect_conf)
    b_val = yolo_only_at_recall(val, val_two["recall"])
    b_conf = b_val["yolo_conf"] if b_val else a.conf
    base = {"yolo_conf": b_conf, **evaluate(test, b_conf), **bootstrap(test, b_conf)}
    paired = paired_bootstrap_diff(
        test, {"yolo_conf": a.conf, "cls_thr": thr, "protect_conf": a.protect_conf}, {"yolo_conf": b_conf})
    report = {
        "chosen_on_val": {"cls_thr": thr, "yolo_conf": a.conf, "protect_conf": a.protect_conf},
        "test_stage1_only": {**s1, **bootstrap(test, a.conf)},
        "test_two_stage": {**s2, **bootstrap(test, a.conf, thr, a.protect_conf)},
        "test_yolo_only_armB_threshold_from_val": base,
        "paired_diff_twostage_minus_armB": paired,
        "test_yolo_only_curve": yolo_only_curve(test)[::5],
        "val_sweep": sweep_cls(val, a.conf, a.protect_conf)[::5],
    }
    json.dump(report, open(a.out, "w"), indent=2)
    print(json.dumps({k: report[k] for k in report if k != "val_sweep"}, indent=2))


if __name__ == "__main__":
    main()
