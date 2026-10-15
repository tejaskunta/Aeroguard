#!/usr/bin/env python3
"""Phase 6 E6 — Grad-CAM head-to-head: YOLO vs MobileNetV3 baseline vs stage-2
crops on the same positive v2 test images (plan §Phase 6, E6).

Enrichment here = share of heatmap energy inside the union of the image's GT
boxes, divided by that union's share of image area (same definition as E1, but
on the box union so the two models are compared per IMAGE).

Pairs: images where both YOLO (deployment arm C) and the MNv3 A+B baseline
(val-chosen threshold) are positive. Stage-2 crops: matched proposal crops at
the deployment conf, energy measured inside the GT-in-crop region.

Outputs:
  e6_pairs.csv, e6_crops.csv, e6_summary.json, e6_grid.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import two_stage_eval as ts  # noqa: E402
from phase6_gradcam import YoloCAM, ClsCAM, match_map, overlay  # noqa: E402
from mnv3_shared import build_mnv3_large, preprocess_uint8  # noqa: E402


def union_box(gt):
    x1 = min(g[0] for g in gt); y1 = min(g[1] for g in gt)
    x2 = max(g[2] for g in gt); y2 = max(g[3] for g in gt)
    return [x1, y1, x2, y2]


def energy_union(heat, boxes, scale_xy=(1.0, 1.0)):
    """energy share / area share for the union of boxes (heat: HxW 0..1)."""
    sx, sy = scale_xy
    h, w = heat.shape
    mask = np.zeros_like(heat, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        a = max(0, int(round(x1 * sx))); b = max(0, int(round(y1 * sy)))
        c = min(w, int(round(x2 * sx))); d = min(h, int(round(y2 * sy)))
        mask[b:d, a:c] = True
    total = heat.sum() + 1e-12
    es = float(heat[mask].sum() / total)
    a_share = float(mask.sum()) / heat.size
    return es, (es / a_share if a_share > 0 else float("nan"))


def cls_score(model, img_rgb_hw, device="cuda"):
    t = torch.from_numpy(img_rgb_hw).permute(2, 0, 1).unsqueeze(0)
    x = preprocess_uint8(t).to(device)
    with torch.no_grad():
        return float(torch.sigmoid(model(x)).view(-1)[0])


def proposal_crop(img, box, pad=0.2, size=224):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + 2 * pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = cv2.resize(img[b:d, a:c], (size, size))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # GT in crop coords (224 scale)
    sx, sy = size / max(c - a, 1), size / max(d - b, 1)
    return rgb, (a, b, sx, sy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--cls", required=True, help="stage-2 weights")
    ap.add_argument("--baseline", required=True, help="MNv3 A+B stage_final.pt")
    ap.add_argument("--baseline-metrics", required=True, help="baseline metrics.json")
    ap.add_argument("--val-report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    data = Path(args.data)
    images = data / "test" / "images"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vr = json.load(open(args.val_report, encoding="utf-8"))
    chosen = {"yolo_conf": vr["chosen_effective"]["yolo_conf"],
              "cls_thr": vr["chosen_effective"]["cls_thr"],
              "protect_conf": vr["chosen_effective"]["protect_conf"]}
    bm = json.load(open(args.baseline_metrics, encoding="utf-8"))
    base_thr = bm["chosen_on_val"]["recall_first_threshold"]
    print("chosen", chosen, "baseline_thr", base_thr, flush=True)

    records = ts.build_records(args.yolo, args.cls, images, data / "test" / "labels",
                               conf=chosen["yolo_conf"], crop_size=224)
    ycam = YoloCAM(args.yolo)
    base = build_mnv3_large(pretrained=False)
    base.load_state_dict(torch.load(args.baseline, map_location="cuda"))
    base.eval().to("cuda")
    bc = ClsCAM(args.baseline)      # hooks on the baseline model itself
    sc = ClsCAM(args.cls)           # stage-2 model

    pairs, crop_rows = [], []
    for r in records:
        if not r["gt"]:
            continue
        img = cv2.imread(str(images / r["name"]))
        if img is None:
            continue
        kept = [d for d in r["dets"] if ts.keep(d, chosen["yolo_conf"],
                                                 chosen["cls_thr"], chosen["protect_conf"])]
        d2g = match_map(r["gt"], kept)
        yolo_pos = any(g >= 0 for g in d2g)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        s224 = cv2.resize(rgb, (224, 224))
        h_img, w_img = img.shape[:2]
        base_s = cls_score(base, s224)
        base_pos = base_s >= base_thr

        y_en = m_en = np.nan
        if yolo_pos:
            heat, _ = ycam.cam(img)
            y_en = energy_union(heat, r["gt"])[1]
        if base_pos:
            mheat, _ = bc.cam(s224)
            m_en = energy_union(mheat, r["gt"], scale_xy=(224 / w_img, 224 / h_img))[1]
        if yolo_pos and base_pos:
            pairs.append({"image": r["name"], "yolo_enrichment": y_en,
                          "mnv3_enrichment": m_en, "baseline_score": base_s})
        elif yolo_pos:
            pairs.append({"image": r["name"], "yolo_enrichment": y_en,
                          "mnv3_enrichment": np.nan, "baseline_score": base_s})
        elif base_pos:
            pairs.append({"image": r["name"], "yolo_enrichment": np.nan,
                          "mnv3_enrichment": m_en, "baseline_score": base_s})

        # stage-2 crop CAMs for matched TPs
        for i, gi in enumerate(d2g):
            if gi < 0:
                continue
            rgb_c, (a, b, sx, sy) = proposal_crop(img, kept[i]["box"])
            gtc = []
            gx1, gy1, gx2, gy2 = r["gt"][gi]
            gtc = [(gx1 - a) * sx, (gy1 - b) * sy, (gx2 - a) * sx, (gy2 - b) * sy]
            heat, logit = sc.cam(rgb_c)
            es, en = energy_union(heat, [gtc])
            crop_rows.append({"image": r["name"], "det": i, "conf": kept[i]["conf"],
                              "energy_share": es, "enrichment": en,
                              "crop_logit": logit})
        if len(pairs) % 40 == 0 and pairs:
            print(f"  {len(pairs)} images", flush=True)

    with open(out / "e6_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        w.writeheader(); w.writerows(pairs)
    with open(out / "e6_crops.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(crop_rows[0].keys()))
        w.writeheader(); w.writerows(crop_rows)

    both = [p for p in pairs if not np.isnan(p["yolo_enrichment"])
            and not np.isnan(p["mnv3_enrichment"])]
    ye = np.array([p["yolo_enrichment"] for p in both])
    me = np.array([p["mnv3_enrichment"] for p in both])
    rng = np.random.default_rng(0)
    diffs = [float((ye[rng.integers(0, len(ye), len(ye))]
                    - me[rng.integers(0, len(me), len(me))]).mean()) for _ in range(2000)]
    ce = np.array([c["enrichment"] for c in crop_rows])
    summary = {
        "n_pairs_both_positive": len(both),
        "n_yolo_positive": int(sum(1 for p in pairs if not np.isnan(p["yolo_enrichment"]))),
        "n_base_positive": int(sum(1 for p in pairs if not np.isnan(p["mnv3_enrichment"]))),
        "yolo_enrichment": {"mean": float(np.nanmean([p["yolo_enrichment"] for p in pairs])),
                            "median": float(np.nanmedian([p["yolo_enrichment"] for p in pairs]))},
        "mnv3_enrichment": {"mean": float(np.nanmean([p["mnv3_enrichment"] for p in pairs])),
                            "median": float(np.nanmedian([p["mnv3_enrichment"] for p in pairs]))},
        "paired_diff_yolo_minus_mnv3": {
            "mean": float((ye - me).mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]},
        "stage2_crop_enrichment": {"n": len(ce), "mean": float(ce.mean()),
                                   "median": float(np.median(ce))},
        "baseline_thr": base_thr,
    }
    (out / "e6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    # example grid: 3 images (high / mid / low yolo enrichment among both-positive)
    pick = sorted(both, key=lambda p: p["yolo_enrichment"])
    picks = [pick[0], pick[len(pick) // 2], pick[-1]]
    panels = []
    for p in picks:
        rec = next(r for r in records if r["name"] == p["image"])
        img = cv2.imread(str(images / p["image"]))
        gt = overlay(ycam.cam(img)[0])
        vis = cv2.addWeighted(img, 0.55, gt, 0.45, 0)
        for x1, y1, x2, y2 in rec["gt"]:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
        mh, _ = bc.cam(rgb)
        mh_img = overlay(cv2.resize(mh, (img.shape[1], img.shape[0])))
        mv = cv2.addWeighted(img, 0.55, mh_img, 0.45, 0)
        for x1, y1, x2, y2 in rec["gt"]:
            cv2.rectangle(mv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        row = np.hstack([vis, mv])
        row = cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2))
        panels.append(row)
    hmin = min(p.shape[0] for p in panels)
    grid = np.hstack([p[:hmin] for p in panels])
    cv2.imwrite(str(out / "e6_grid.png"), grid)
    print("wrote", out / "e6_grid.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
