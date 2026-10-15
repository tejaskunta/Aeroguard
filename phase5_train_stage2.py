#!/usr/bin/env python3
"""Phase 5.2 — stage-2 crop classifier (plan §Phase 5.2).

Same constructor + normalisation as Phase 4 (mnv3_shared). Whole network
fine-tuned at a small learning rate. Mild augmentation (flip, rotation, light
colour jitter). Early stopping on crop-val loss. Seeds 0/1/2 via --seed.
Neg:pos ratio sampled per epoch (1 or 3; choose on val, plan §Phase 5.1).

Outputs under --out:
  best.pt            MineClassifier state_dict (loads into mnv3_shared.build_mnv3_large)
  train_log.csv      epoch, loss, val_loss, val_auc, val_f1
  summary.json       config + best epoch + final val metrics
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mnv3_shared import build_mnv3_large, preprocess_uint8  # noqa: E402


class CropDS(Dataset):
    def __init__(self, paths, labels, train: bool, seed: int = 0):
        self.paths = paths
        self.labels = labels
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.paths)

    def _aug(self, img: Image.Image) -> Image.Image:
        from torchvision.transforms import functional as TF
        if self.rng.random() < 0.5:
            img = TF.hflip(img)
        img = TF.rotate(img, self.rng.uniform(-15, 15), fill=0)
        img = TF.adjust_brightness(img, self.rng.uniform(0.85, 1.15))
        img = TF.adjust_contrast(img, self.rng.uniform(0.85, 1.15))
        return img

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.train:
            img = self._aug(img)
        t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0)
        x = preprocess_uint8(t).squeeze(0)
        return x, torch.tensor([float(self.labels[i])], dtype=torch.float32)


def auc(scores, labels) -> float:
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    all_s = np.concatenate([pos, neg])
    for v in np.unique(all_s):
        m = all_s == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def f1_at(scores, labels, thr=0.5):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=int)
    pred = (scores >= thr).astype(int)
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0, prec, rec


def load_split(crop_root: Path, split: str, ratio: float, seed: int,
               max_pos: int | None = None):
    rng = random.Random(seed)
    pos = sorted(p for p in (crop_root / split / "pos").glob("*.jpg"))
    neg = sorted(p for p in (crop_root / split / "neg").glob("*.jpg"))
    if max_pos is not None and len(pos) > max_pos:
        pos = rng.sample(pos, max_pos)
    if ratio is not None and ratio > 0:
        n_neg = min(len(neg), max(1, int(round(len(pos) * ratio))))
        neg = rng.sample(neg, n_neg)
    paths = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    idx = list(range(len(paths)))
    rng.shuffle(idx)
    return [paths[i] for i in idx], [labels[i] for i in idx]


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    ys, ss, loss_sum, n = [], [], 0.0, 0
    lossf = nn.BCEWithLogitsLoss()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).view(-1)
        logits = model(xb).view(-1)
        loss_sum += float(lossf(logits, yb)) * len(xb)
        n += len(xb)
        ys.extend(yb.cpu().numpy().tolist())
        ss.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return loss_sum / max(n, 1), ss, ys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, help="crop dataset root (phase5_build_crops --out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=1.0, help="neg:pos per epoch (1 or 3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5, help="whole-net fine-tune lr (small)")
    ap.add_argument("--max-pos-per-epoch", type=int, default=2048,
                    help="positives per epoch (11.5k GT crops exist); neg = ratio x this")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.crops)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr_pos = sorted(p for p in (root / "train" / "pos").glob("*.jpg"))
    tr_neg = sorted(p for p in (root / "train" / "neg").glob("*.jpg"))
    va_p, va_y = load_split(root, "valid", None, args.seed)
    n_pos_ep = min(args.max_pos_per_epoch, len(tr_pos))
    n_neg_ep = min(len(tr_neg), int(round(n_pos_ep * args.ratio)))
    print(f"train pool pos={len(tr_pos)} neg={len(tr_neg)} | "
          f"per-epoch pos={n_pos_ep} neg={n_neg_ep} | "
          f"val crops pos={sum(va_y)} neg={len(va_y) - sum(va_y)}", flush=True)

    va_ds = CropDS(va_p, va_y, train=False)
    va_ld = DataLoader(va_ds, batch_size=args.batch * 2, shuffle=False,
                       num_workers=0, pin_memory=True)

    model = build_mnv3_large(pretrained=True).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()

    ep_rng = random.Random(args.seed)
    best_val, best_state, best_ep, bad = float("inf"), None, 0, 0
    log_rows = []
    for ep in range(1, args.epochs + 1):
        # per-epoch neg:pos resampling (plan §5.1 ratio choice, 1:1 or 1:3)
        pos_s = ep_rng.sample(tr_pos, n_pos_ep)
        neg_s = ep_rng.sample(tr_neg, n_neg_ep)
        paths = pos_s + neg_s
        labels = [1] * len(pos_s) + [0] * len(neg_s)
        tr_ld = DataLoader(CropDS(paths, labels, train=True, seed=args.seed + ep),
                           batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, pin_memory=True)
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(args.device), yb.to(args.device).view(-1)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb).view(-1), yb)
            loss.backward()
            opt.step()
            run += float(loss) * len(xb)
            n += len(xb)
        tr_loss = run / max(n, 1)
        va_loss, va_s, va_yt = eval_loader(model, va_ld, args.device)
        a = auc(va_s, va_yt)
        f1, _, _ = f1_at(va_s, va_yt, 0.5)
        log_rows.append({"epoch": ep, "train_loss": round(tr_loss, 5),
                         "val_loss": round(va_loss, 5), "val_auc": round(a, 5),
                         "val_f1": round(f1, 5), "sec": round(time.time() - t0, 1)})
        print(f"  ep {ep}/{args.epochs} tr {tr_loss:.4f} va {va_loss:.4f} "
              f"auc {a:.4f} f1 {f1:.4f} ({log_rows[-1]['sec']}s)", flush=True)
        if va_loss < best_val - 1e-5:
            best_val, best_ep, bad = va_loss, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at {ep} (best ep {best_ep})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out / "best.pt")
    _, best_s, best_y = eval_loader(model, va_ld, args.device)
    f1, prec, rec = f1_at(best_s, best_y, 0.5)
    summary = {"config": {k: getattr(args, k) for k in
                          ("crops", "ratio", "seed", "epochs", "patience", "batch", "lr",
                           "max_pos_per_epoch")},
               "best_epoch": best_ep, "best_val_loss": best_val,
               "val_auc": auc(best_s, best_y),
               "val_f1_at_0.5": f1, "val_precision_at_0.5": prec,
               "val_recall_at_0.5": rec,
               "train_crops": {"pos_pool": len(tr_pos), "neg_pool": len(tr_neg),
                               "pos_per_epoch": n_pos_ep, "neg_per_epoch": n_neg_ep},
               "val_crops": {"pos": int(sum(va_y)), "neg": int(len(va_y) - sum(va_y))}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
