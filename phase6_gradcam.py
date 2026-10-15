#!/usr/bin/env python3
"""Phase 6 — Grad-CAM experiments E1 (energy-in-box), E2 (deletion),
E4 (failure gallery) on the v2 test set (plan §Phase 6).

Method (plan §6.0):
  * YOLOv8: target layer = the stride-32 neck feature fed into Detect (captured as
    Detect's forward input); score = top class score of the raw output before NMS.
  * MobileNetV3 (stage 2): last conv block (features[-1]) + output logit.
  * Heatmap = ReLU of the gradient-weighted sum of activations, bilinear-upsampled
    to image size, normalised per image (max 1).
Wording rule: Grad-CAM shows which regions influenced the prediction; it never
proves the prediction is right.

Outputs under --out:
  e1_energy.csv + e1_summary.json + e1_hist.png
  e2_deletion.csv + e2_summary.json + e2_hist.png
  e4_gallery/*.png + e4_failures.csv   (hand categorisation happens after)
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import two_stage_eval as ts  # noqa: E402
from mnv3_shared import build_mnv3_large, preprocess_uint8  # noqa: E402


# --------------------------------------------------------------------- Grad-CAM

class YoloCAM:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.yolo = YOLO(str(weights))
        self.model = self.yolo.model
        self.model.eval()
        self.model.requires_grad_(True)   # frozen on load; need a graph, never step
        self._act = None
        det = self.model.model[-1]
        det.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        # inp[0]: list of neck maps feeding Detect (stride 8/16/32 in bottom-up
        # order). Keep all with retain_grad; cam() picks the largest stride whose
        # gradient is actually non-zero (the top pre-NMS anchor often sits on the
        # stride-8 head, where stride-32 grads are exactly zero). Documented
        # against plan §6.0's "stride 16 or 32" hint in method notes.
        feats = inp[0] if isinstance(inp[0], (list, tuple)) else [inp[0]]
        for f in feats:
            if f.requires_grad:
                f.retain_grad()
        self._act = list(feats)

    def _defuse(self):
        """predict() moves the model to CUDA inside torch.inference_mode(), which
        turns tensors into inference tensors that break backward. This includes
        plain tensor attributes on Detect (anchors, strides) that are not
        registered buffers. Re-materialise any inference tensor as a normal one."""
        for m in self.model.modules():
            for k, v in list(m.__dict__.items()):
                if torch.is_tensor(v) and v.is_inference():
                    m.__dict__[k] = v.clone()

    @torch.enable_grad()
    def cam(self, img_bgr):
        """img_bgr HWC uint8 -> (heatmap HxW float 0..1, top_score float)."""
        p = next(self.model.parameters())
        if not p.requires_grad:
            self.model.requires_grad_(True)   # predict() strips grads; re-arm
        self._defuse()
        t = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        t = t.unsqueeze(0).float().div(255).to(p.device)
        self.model.zero_grad(set_to_none=True)
        out = self.model(t)
        raw = out[0] if isinstance(out, tuple) else out
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        score = raw[:, 4:].amax()          # top class score pre-NMS (single class here)
        score.backward(retain_graph=False)
        chosen = None
        for a in reversed(self._act):      # stride 32 -> 16 -> 8
            if a.grad is not None and float(a.grad.abs().max()) > 0:
                chosen = a
                break
        if chosen is None:
            chosen = self._act[0]
        act, grad = chosen, chosen.grad
        cam = (grad * act).sum(dim=1, keepdim=True).clamp(min=0)[0, 0]
        cam = F.interpolate(cam[None, None], size=img_bgr.shape[:2],
                            mode="bilinear", align_corners=False)[0, 0]
        m = cam.max()
        if m > 0:
            cam = cam / m
        return cam.detach().cpu().numpy(), float(score.detach())


class ClsCAM:
    def __init__(self, weights, device="cuda"):
        self.device = device
        self.model = build_mnv3_large(pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location=device))
        self.model.eval().to(device)
        self._act = None
        self.model.features[-1].register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        out.retain_grad()
        self._act = out

    @torch.enable_grad()
    def cam(self, crop_rgb_uint8_hw):
        """crop HWC uint8 -> (heatmap HxW float 0..1, logit float)."""
        t = torch.from_numpy(crop_rgb_uint8_hw).permute(2, 0, 1).unsqueeze(0)
        x = preprocess_uint8(t).to(self.device)
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x).view(-1)[0]
        logit.backward()
        act, grad = self._act, self._act.grad
        cam = (grad * act).sum(dim=1, keepdim=True).clamp(min=0)[0, 0]
        cam = F.interpolate(cam[None, None], size=crop_rgb_uint8_hw.shape[:2],
                            mode="bilinear", align_corners=False)[0, 0]
        m = cam.max()
        if m > 0:
            cam = cam / m
        return cam.detach().cpu().numpy(), float(logit.detach())


def match_map(gt, dets, iou_thr=0.5):
    """greedy match -> (gt_index or -1) per det, (det_index or -1) per gt."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["conf"])
    det2gt = [-1] * len(dets)
    taken = set()
    for i in order:
        best, bi = iou_thr, -1
        for j, g in enumerate(gt):
            if j in taken:
                continue
            v = ts.iou_xyxy(dets[i]["box"], g)
            if v >= best:
                best, bi = v, j
        if bi >= 0:
            det2gt[i] = bi
            taken.add(bi)
    return det2gt


# ------------------------------------------------------------------------- E1

def energy_stats(heat, box):
    """box: one (x1,y1,x2,y2) or a list/array of boxes -> union mask."""
    h, w = heat.shape
    boxes = np.asarray(box, dtype=float)
    if boxes.ndim == 1:
        boxes = boxes[None]
    mask = np.zeros_like(heat, dtype=bool)
    regions = []
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = True
        regions.append((x1, y1, x2, y2))
    total = heat.sum() + 1e-12
    e_share = float(heat[mask].sum() / total)
    a_share = float(mask.sum()) / heat.size
    py, px = np.unravel_index(np.argmax(heat), heat.shape)
    peak_in = any(x1 <= px < x2 and y1 <= py < y2 for x1, y1, x2, y2 in regions)
    return e_share, a_share, (e_share / a_share if a_share > 0 else np.nan), peak_in


def run_e1(cams_by_name, records, chosen, out: Path):
    """cams_by_name: callable(name, img_bgr) -> heat."""
    rows = []
    for r in records:
        kept = [d for d in r["dets"] if ts.keep(d, chosen["yolo_conf"],
                                                 chosen["cls_thr"], chosen["protect_conf"])]
        d2g = match_map(r["gt"], kept)
        img = cv2.imread(str(IMAGES / r["name"]))
        heat = cams_by_name(r["name"], img) if img is not None else None
        if heat is None:
            continue
        for i, gi in enumerate(d2g):
            if gi < 0:
                continue
            es, as_, en, pk = energy_stats(heat, r["gt"][gi])
            rows.append({"image": r["name"], "det": i, "conf": kept[i]["conf"],
                         "energy_share": es, "area_share": as_, "enrichment": en,
                         "peak_inside": int(pk)})
    with open(out / "e1_energy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    es = np.array([x["energy_share"] for x in rows])
    en = np.array([x["enrichment"] for x in rows])
    pk = np.array([x["peak_inside"] for x in rows])
    summary = {"n": len(rows),
               "energy_share": {"mean": float(es.mean()), "median": float(np.median(es)),
                                "q1": float(np.percentile(es, 25)),
                                "q3": float(np.percentile(es, 75))},
               "enrichment": {"mean": float(en.mean()), "median": float(np.median(en))},
               "peak_inside_rate": float(pk.mean())}
    (out / "e1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        ax[0].hist(es, bins=30, color="#3b7dd8")
        ax[0].set_xlabel("energy share in GT box")
        ax[0].set_ylabel("TP count")
        ax[1].hist(np.clip(en, 0, 20), bins=30, color="#d87f3b")
        ax[1].set_xlabel("enrichment (share / area share)")
        fig.tight_layout()
        fig.savefig(out / "e1_hist.png", dpi=200)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("histogram skipped:", e)
    print("E1", json.dumps(summary, indent=2))
    return rows


# ------------------------------------------------------------------------- E2

def top_conf(yolo, img, conf=0.001):
    res = yolo.predict(img, conf=conf, imgsz=640, verbose=False)[0]
    if not len(res.boxes):
        return 0.0
    return float(res.boxes.conf.max())


def bg_box(gt_all, w, h, size_box, rng, tries=50):
    x1, y1, x2, y2 = size_box
    bw, bh = x2 - x1, y2 - y1
    for _ in range(tries):
        cx, cy = rng.uniform(bw / 2, w - bw / 2), rng.uniform(bh / 2, h - bh / 2)
        cand = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
        if all(ts.iou_xyxy(cand, g) < 0.1 for g in gt_all):
            return [int(max(0, cand[0])), int(max(0, cand[1])),
                    int(min(w, cand[2])), int(min(h, cand[3]))]
    return None


def run_e2(records, chosen, out: Path, seed=0):
    """Deletion: mask mine box vs same-size background box (mean fill)."""
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_W))
    rng = np.random.default_rng(seed)
    rows = []
    for r in records:
        kept = [d for d in r["dets"] if ts.keep(d, chosen["yolo_conf"],
                                                 chosen["cls_thr"], chosen["protect_conf"])]
        d2g = match_map(r["gt"], kept)
        if not r["gt"] or all(g < 0 for g in d2g):
            continue
        img = cv2.imread(str(IMAGES / r["name"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        gi = next(g for g in d2g if g >= 0)
        mine = [int(v) for v in r["gt"][gi]]
        t0 = top_conf(yolo, img)
        m_img = img.copy()
        mx1, my1, mx2, my2 = mine
        m_img[my1:my2, mx1:mx2] = int(img[my1:my2, mx1:mx2].mean())
        t1 = top_conf(yolo, m_img)
        bb = bg_box(r["gt"], w, h, mine, rng)
        t2 = None
        if bb is not None:
            b_img = img.copy()
            bx1, by1, bx2, by2 = bb
            b_img[by1:by2, bx1:bx2] = int(img[by1:by2, bx1:bx2].mean())
            t2 = top_conf(yolo, b_img)
        rows.append({"image": r["name"], "conf_orig": t0, "conf_mine_masked": t1,
                     "conf_bg_masked": t2 if t2 is not None else "",
                     "drop_mine": t0 - t1,
                     "drop_bg": (t0 - t2) if t2 is not None else ""})
        if len(rows) % 25 == 0:
            print(f"  e2 {len(rows)} images", flush=True)
    with open(out / "e2_deletion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(rows[0].keys()))
        w.writerows(rows)
    dm = np.array([x["drop_mine"] for x in rows])
    db = np.array([x["drop_bg"] for x in rows if x["drop_bg"] != ""])
    summary = {"n": len(rows),
               "drop_mine": {"mean": float(dm.mean()), "median": float(np.median(dm)),
                             "q1": float(np.percentile(dm, 25)),
                             "q3": float(np.percentile(dm, 75))},
               "drop_bg": {"mean": float(db.mean()), "median": float(np.median(db)),
                           "q1": float(np.percentile(db, 25)),
                           "q3": float(np.percentile(db, 75))},
               "mine_drop_gt_bg": bool(dm.mean() > db.mean())}
    (out / "e2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dm, bins=30, alpha=0.6, label="mine masked", color="#d83f3f")
        ax.hist(db, bins=30, alpha=0.6, label="background masked", color="#3b7dd8")
        ax.set_xlabel("top-confidence drop")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "e2_hist.png", dpi=200)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("histogram skipped:", e)
    print("E2", json.dumps(summary, indent=2))
    return rows


# ------------------------------------------------------------------------- E4

def overlay(heat, color_bgr=(0, 0, 255), alpha=0.45):
    hcv = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    hcv = cv2.applyColorMap(hcv, cv2.COLORMAP_JET)
    return cv2.addWeighted(hcv, alpha, np.zeros_like(hcv), 1 - alpha, 0)


def run_e4(cams_by_name, records, chosen, out: Path, max_fp=50):
    gal = out / "e4_gallery"
    gal.mkdir(parents=True, exist_ok=True)
    rows = []
    n_fn = n_fp = 0
    for r in records:
        kept = [d for d in r["dets"] if ts.keep(d, chosen["yolo_conf"],
                                                 chosen["cls_thr"], chosen["protect_conf"])]
        d2g = match_map(r["gt"], kept)
        fns = [j for j in range(len(r["gt"])) if j not in [g for g in d2g if g >= 0]]
        fps = [i for i, g in enumerate(d2g) if g < 0]
        if not fns and not fps:
            continue
        img = cv2.imread(str(IMAGES / r["name"]))
        if img is None:
            continue
        heat = cams_by_name(r["name"], img)
        hm = overlay(heat)
        vis = cv2.addWeighted(img, 0.55, hm, 0.45, 0)
        stem = Path(r["name"]).stem
        for j in fns:
            x1, y1, x2, y2 = [int(v) for v in r["gt"][j]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)  # GT = yellow
            cv2.putText(vis, "FN gt", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            p = gal / f"{stem}__FN{j}.png"
            cv2.imwrite(str(p), vis)
            rows.append({"file": p.name, "image": r["name"], "kind": "FN", "conf": "",
                         "gt": j, "category": ""})
            n_fn += 1
        for i in fps:
            if n_fp >= max_fp:
                break
            x1, y1, x2, y2 = [int(v) for v in kept[i]["box"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"FP {kept[i]['conf']:.2f}", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            p = gal / f"{stem}__FP{i}.png"
            cv2.imwrite(str(p), vis)
            rows.append({"file": p.name, "image": r["name"], "kind": "FP",
                         "conf": kept[i]["conf"], "gt": "", "category": ""})
            n_fp += 1
    with open(out / "e4_failures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"E4 gallery: {n_fn} FN + {n_fp} FP images -> {gal}")
    return rows


# --------------------------------------------------------------------------

IMAGES: Path = None
YOLO_W: Path = None


def main() -> int:
    global IMAGES, YOLO_W
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--cls", required=True, help="stage-2 weights (build_records needs it)")
    ap.add_argument("--val-report", required=True, help="twostage val_report.json (chosen)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp", default="E1,E2,E4")
    ap.add_argument("--max-fp", type=int, default=50)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    data = Path(args.data)
    IMAGES = data / "test" / "images"
    YOLO_W = Path(args.yolo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vr = json.load(open(args.val_report, encoding="utf-8"))
    chosen = {"yolo_conf": vr["chosen_effective"]["yolo_conf"],
              "cls_thr": vr["chosen_effective"]["cls_thr"],
              "protect_conf": vr["chosen_effective"]["protect_conf"]}
    print("chosen:", chosen, flush=True)

    print("building test records...", flush=True)
    records = ts.build_records(args.yolo, args.cls, IMAGES, data / "test" / "labels",
                               conf=chosen["yolo_conf"], crop_size=224)
    cam = YoloCAM(args.yolo)
    cache = {}

    def cams_by_name(name, img):
        if name not in cache:
            cache[name], _ = cam.cam(img)
        return cache[name]

    exps = [e.strip().upper() for e in args.exp.split(",")]
    if "E1" in exps:
        run_e1(cams_by_name, records, chosen, out)
    if "E2" in exps:
        run_e2(records, chosen, out)
    if "E4" in exps:
        run_e4(cams_by_name, records, chosen, out, max_fp=args.max_fp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
