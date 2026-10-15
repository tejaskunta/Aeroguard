#!/usr/bin/env python3
"""Phase 6 E3 — shortcut/paste test (plan §Phase 6, E3).

Conditions (~100 composites each, rng seed 0):
  paste   : test-set mine crops (feathered soft edges) pasted onto other test
            backgrounds at random non-GT locations -> detection must HOLD at
            the pasted box (IoU>=0.5).
  remove  : the mine is inpainted out of (originally deploy-detected) test
            images -> detection at the original box must DISAPPEAR.

Also renders e3_marker_check.png: 3x-box crops around random test GT for a
hand look at markers/paint/flags next to mines (plan wording).

Outputs under --out: e3_summary.json, e3_paste.csv, e3_remove.csv,
e3_paste_examples.png, e3_remove_examples.png, e3_marker_check.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import two_stage_eval as ts  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def feather_mask(h, w, feather=9):
    m = np.ones((h, w), np.float32)
    m[: feather // 2, :] = 0
    m[-(feather // 2):, :] = 0
    m[:, : feather // 2] = 0
    m[:, -(feather // 2):] = 0
    m = cv2.GaussianBlur(m, (0, 0), feather / 2.0)
    return m[..., None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.02, help="deployment conf")
    ap.add_argument("--conf2", type=float, default=0.125, help="arm-B conf (extra column)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    yolo = YOLO(args.yolo)
    rng = np.random.default_rng(args.seed)

    tdir = data / "test"
    names = sorted(p.name for p in (tdir / "images").iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    imgs, gts = {}, {}
    for n in names:
        im = cv2.imread(str(tdir / "images" / n))
        if im is None:
            continue
        imgs[n] = im
        h, w = im.shape[:2]
        gts[n] = ts._read_gt(str(tdir / "labels" / (Path(n).stem + ".txt")), w, h)

    # ---------------- paste condition: mine crop -> other test background
    pos = [(n, g) for n in names if gts.get(n) for g in gts[n]]
    pos = [pos[i] for i in rng.permutation(len(pos))]
    paste_rows = []
    example_tiles = []
    for src, box in pos[: args.n]:
        # random destination different from source
        dest_cands = [n for n in names if n != src and gts.get(n)]
        dst = dest_cands[rng.integers(len(dest_cands))]
        di = imgs[dst]
        dh, dw = di.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        crop = imgs[src][max(0, y1):y2, max(0, x1):x2].copy()
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            continue
        ch, cw = crop.shape[:2]
        if ch >= dh or cw >= dw:
            continue
        # random location away from destination GT
        ok, tries = False, 0
        px = py = 0
        while ok is False and tries < 60:
            tries += 1
            px = int(rng.integers(0, dw - cw))
            py = int(rng.integers(0, dh - ch))
            pb = [px, py, px + cw, py + ch]
            if all(ts.iou_xyxy(pb, g) < 0.05 for g in gts[dst]):
                ok = True
        if not ok:
            continue
        alpha = feather_mask(ch, cw)
        roi = di[py:py + ch, px:px + cw].astype(np.float32)
        di[py:py + ch, px:px + cw] = (crop.astype(np.float32) * alpha +
                                      roi * (1 - alpha)).astype(np.uint8)
        pbox = [float(px), float(py), float(px + cw), float(py + ch)]
        res = yolo.predict(di, conf=args.conf, imgsz=640, verbose=False)[0]
        dets = (res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()) if len(res.boxes) \
            else (np.zeros((0, 4)), np.zeros(0))
        best = max((ts.iou_xyxy(list(b), pbox) for b in dets[0]), default=0.0)
        kept2 = [b for b, c in zip(dets[0], dets[1]) if c >= args.conf2]
        best2 = max((ts.iou_xyxy(list(b), pbox) for b in kept2), default=0.0)
        paste_rows.append({"src": src, "dst": dst, "paste_iou": best,
                           "detected": int(best >= 0.5),
                           "detected_arm_b": int(best2 >= 0.5),
                           "paste_box": ",".join(f"{v:.0f}" for v in pbox)})
        if len(example_tiles) < 8:
            vis = di.copy()
            cv2.rectangle(vis, (px, py), (px + cw, py + ch), (0, 255, 255), 2)
            cv2.putText(vis, f"iou={best:.2f}", (4, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            example_tiles.append(cv2.resize(vis, (360, int(360 * dh / dw))[:2]))

    # ---------------- remove condition: inpaint the mine out
    # candidates: images where deploy conf detects at least one GT box
    rem_rows = []
    rem_tiles = []
    for n in names:
        if len(rem_rows) >= args.n:
            break
        if not gts.get(n):
            continue
        im = imgs[n].copy()
        h, w = im.shape[:2]
        res = yolo.predict(im, conf=args.conf, imgsz=640, verbose=False)[0]
        if not len(res.boxes):
            continue
        dets = [list(map(float, b)) for b in res.boxes.xyxy.cpu().numpy()]
        confs = [float(c) for c in res.boxes.conf.cpu().numpy()]
        # pick the GT box that is currently detected (highest-conf match)
        hit = None
        for gi, g in enumerate(gts[n]):
            for d, c in zip(dets, confs):
                if ts.iou_xyxy(d, g) >= 0.5:
                    if hit is None or c > hit[2]:
                        hit = (gi, d, c)
        if hit is None:
            continue
        gi, d, c = hit
        g = gts[n][gi]
        mask = np.zeros((h, w), np.uint8)
        x1, y1, x2, y2 = [int(round(v)) for v in g]
        pad = max(2, int(0.15 * max(x2 - x1, y2 - y1)))
        mask[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] = 255
        cleaned = cv2.inpaint(im, mask, 5, cv2.INPAINT_TELEA)
        res2 = yolo.predict(cleaned, conf=args.conf, imgsz=640, verbose=False)[0]
        dets2 = [list(map(float, b)) for b in res2.boxes.xyxy.cpu().numpy()] if len(res2.boxes) else []
        confs2 = [float(cc) for cc in res2.boxes.conf.cpu().numpy()] if len(res2.boxes) else []
        best2 = max((ts.iou_xyxy(d, g) for d in dets2), default=0.0)
        # any remaining high-conf det near the hole?
        near = max((cc for d, cc in zip(dets2, confs2)
                    if ts.iou_xyxy(d, g) >= 0.3), default=0.0)
        rem_rows.append({"image": n, "gt": ",".join(f"{v:.0f}" for v in g),
                         "was_conf": c, "residual_iou": best2,
                         "disappeared": int(best2 < 0.5),
                         "residual_conf_iou30": near})
        if len(rem_tiles) < 8:
            vis = cleaned.copy()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, f"res={best2:.2f}", (4, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            rem_tiles.append(cv2.resize(vis, (360, int(360 * h / w))[:2]))

    # ---------------- marker/paint/flags hand-look crops
    pos2 = [(n, g) for n in names if gts.get(n) for g in gts[n]]
    pos2 = [pos2[i] for i in rng.permutation(len(pos2))]
    mk_tiles = []
    for n, g in pos2[:12]:
        im = imgs[n].copy()
        h, w = im.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in g]
        bw, bh = max(x2 - x1, 12), max(y2 - y1, 12)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        R = int(max(bw, bh) * 1.5)
        a, b2 = max(0, cx - R), max(0, cy - R)
        c2, d2 = min(w, cx + R), min(h, cy + R)
        crop = im[b2:d2, a:c2].copy()
        cv2.rectangle(crop, (x1 - a, y1 - b2), (x2 - a, y2 - b2), (0, 255, 255), 2)
        mk_tiles.append(cv2.resize(crop, (240, 240)))

    def sheet(tiles, path, cols=4):
        if not tiles:
            return
        th = max(t.shape[0] for t in tiles)
        tw = max(t.shape[1] for t in tiles)
        rows = (len(tiles) + cols - 1) // cols
        s = np.zeros((rows * th, cols * tw, 3), np.uint8)
        for i, t in enumerate(tiles):
            s[(i // cols) * th:(i // cols) * th + t.shape[0],
              (i % cols) * tw:(i % cols) * tw + t.shape[1]] = t
        cv2.imwrite(str(path), s)

    sheet(example_tiles, out / "e3_paste_examples.png")
    sheet(rem_tiles, out / "e3_remove_examples.png")
    sheet(mk_tiles, out / "e3_marker_check.png")

    with open(out / "e3_paste.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(paste_rows[0].keys()))
        wtr.writeheader(); wtr.writerows(paste_rows)
    with open(out / "e3_remove.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rem_rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rem_rows)

    n = len(paste_rows)
    summary = {
        "paste": {"n": n,
                  "detected_rate_conf002": sum(r["detected"] for r in paste_rows) / max(n, 1),
                  "detected_rate_conf0125": sum(r["detected_arm_b"] for r in paste_rows) / max(n, 1)},
        "remove": {"n": len(rem_rows),
                   "disappeared_rate": sum(r["disappeared"] for r in rem_rows) / max(len(rem_rows), 1)},
        "conf": args.conf, "conf2": args.conf2, "seed": args.seed,
    }
    import json
    (out / "e3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
