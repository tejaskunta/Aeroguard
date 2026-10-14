#!/usr/bin/env python3
"""Phase 4 — base-paper baseline: MobileNetV3-Large image-level classifier.

Re-trains the Heuschmid et al. (IEEE Access 2025) recipe on the frozen v2 split
(plan v2 §Phase 4, Appendix A):

  Stage A:  ImageNet weights, head = GAP -> Dense128+ReLU -> Dropout0.3 -> Dense1,
            base frozen, BCE + class weights from train, Adam lr 1e-4,
            20 epochs x 25 steps/epoch x batch 32, paper augmentation.
  Stage B:  unfreeze last ~50 layers (mapped to block boundary, see
            mnv3_shared.unfreeze_last_layers), lr 2e-5, 10 more epochs.
  Extra rows: Stage A at 448 and 640; batch 64 sensitivity.

Deviations vs the Keras paper (documented in deviations.md):
  - PyTorch/torchvision instead of TensorFlow (plan §Phase 4.6).
  - ImageNet mean/std normalisation instead of raw rescale 1/255
    (plan §Phase 4.6: "use ImageNet mean/std consistently").
  - RGB (dataset is RGB) instead of greyscale-converted thermal.

Outputs (plan §9): --out dir with args.json, per_image_{val,test}.csv,
metrics.json, stage_{A,AB}.pt.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnv3_shared import build_mnv3_large, preprocess_uint8, unfreeze_last_layers

from torchvision.transforms import functional as TF


# ------------------------------------------------------------------ data

def read_split(root: Path, split: str):
    """-> list of (image_path, label 0/1) from YOLO labels (label=1 iff >=1 box)."""
    items = []
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        lbl = lbl_dir / (p.name.rsplit(".", 1)[0] + ".txt")
        y = 0
        if lbl.exists():
            txt = lbl.read_text(encoding="utf-8", errors="replace").strip()
            y = int(bool(txt))
        items.append((p, y))
    return items


class ImageLevelDS(Dataset):
    def __init__(self, items, imgsz: int, train: bool, seed: int = 0):
        self.items = items
        self.imgsz = imgsz
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def _augment(self, img: Image.Image) -> Image.Image:
        # paper: rotation +-10deg, zoom <=10%, brightness 0.7-1.0, h+v flips,
        # nearest fill. Implemented with PIL; nearest fill ~ border replicate is
        # approximated with constant fill (see deviations.md).
        if self.rng.random() < 0.5:
            img = TF.hflip(img)
        if self.rng.random() < 0.5:
            img = TF.vflip(img)
        angle = self.rng.uniform(-10, 10)
        scale = self.rng.uniform(0.9, 1.1)
        img = TF.affine(img, angle=angle, translate=(0, 0), scale=scale, shear=0,
                        fill=0)
        b = self.rng.uniform(0.7, 1.0)
        img = TF.adjust_brightness(img, b)
        return img

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        if self.train:
            img = self._augment(img)
        t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0)
        x = preprocess_uint8(t).squeeze(0)
        return x, torch.tensor([float(y)], dtype=torch.float32)


def make_loader(ds, batch, shuffle, workers):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=False)


# ------------------------------------------------------------------ metrics

def image_metrics(y, s, thr):
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(s) >= thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    n = len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    acc = (tp + tn) / n if n else 0.0
    pos, neg = (y == 1).sum(), (y == 0).sum()
    bal = 0.5 * ((tp / pos) if pos else 0.0) + 0.5 * ((tn / neg) if neg else 0.0)
    po = acc
    pe = ((pos / n) * ((tp + fp) / n) + (neg / n) * ((tn + fn) / n)) if n else 0.0
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec,
            "recall": rec, "f2": f2, "accuracy": acc, "balanced_accuracy": bal,
            "kappa": kappa}


def bootstrap_ci(y, s, thr, n_boot=1000, seed=0, keys=("accuracy", "balanced_accuracy", "recall", "precision")):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = image_metrics(y[idx], s[idx], thr)
        for k in keys:
            acc[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in acc.items()}


def recall_first_threshold(y, s, targets=(0.97, 0.95, 0.98)):
    """Largest score threshold whose recall >= target (plan §Phase 3.4 rule,
    applied to the classifier on VAL only)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    out = {}
    for t in targets:
        best = None
        for thr in np.linspace(0.0, 1.0, 1001):
            m = image_metrics(y, s, float(thr))
            if m["recall"] >= t:
                best = float(thr)
        out[str(t)] = best
    return out


# ------------------------------------------------------------------ training

def train_one_stage(model, train_items, args, epochs, lr, stage, seed,
                    class_w, init_weights=None):
    device = args.device
    if init_weights is not None:
        model.load_state_dict(init_weights)
    model.to(device)
    ds = ImageLevelDS(train_items, args.imgsz, train=True, seed=seed)
    loader = make_loader(ds, args.batch, True, args.workers)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    w = torch.tensor(class_w, dtype=torch.float32, device=device)  # [w_neg, w_pos]

    history = []
    model.train()
    for ep in range(epochs):
        # paper: fixed steps per epoch over a shuffled stream (fresh subset each epoch)
        steps = args.steps
        t0, ep_loss, n_seen = time.time(), 0.0, 0
        it = iter(loader)
        for _ in range(steps):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            per = lossf(logits, yb)
            weights = w[yb.long().squeeze(1).long()]
            loss = (per * weights).mean()
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(xb)
            n_seen += len(xb)
        row = {"stage": stage, "epoch": ep + 1, "loss": ep_loss / max(n_seen, 1),
               "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(f"  [{stage}] ep {ep + 1}/{epochs} loss {row['loss']:.4f} ({row['sec']}s)",
              flush=True)
    return model, history


@torch.no_grad()
def infer(model, items, args):
    device = args.device
    model.eval()
    scores, labels, names = [], [], []
    ds = ImageLevelDS(items, args.imgsz, train=False)
    loader = make_loader(ds, args.batch, False, 0)
    for xb, yb in loader:
        logits = model(xb.to(device)).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        scores.extend(float(v) for v in probs)
        labels.extend(int(v) for v in yb.squeeze(1).numpy())
    names = [p.name for p, _ in items]
    return names, labels, scores


def evaluate_split(names, labels, scores, thr, n_boot=1000, seed=0):
    m = image_metrics(labels, scores, thr)
    m["threshold"] = thr
    m["bootstrap95"] = bootstrap_ci(labels, scores, thr, n_boot, seed)
    return m


def save_per_image(path: Path, names, labels, scores):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "label", "score"])
        for n, y, s in zip(names, labels, scores):
            w.writerow([n, y, f"{s:.6f}"])


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="v2 dataset root (contains train/valid/test)")
    ap.add_argument("--stage", choices=["A", "AB"], default="A")
    ap.add_argument("--imgsz", type=int, default=224, choices=[224, 448, 640])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20, help="Stage A epochs")
    ap.add_argument("--ft-epochs", type=int, default=10, help="Stage B epochs")
    ap.add_argument("--steps", type=int, default=25, help="steps per epoch (paper: 25)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-b", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_items = read_split(root, "train")
    val_items = read_split(root, "valid")
    test_items = read_split(root, "test")
    n_pos = sum(y for _, y in train_items)
    n_neg = len(train_items) - n_pos
    n = len(train_items)
    class_w = [n / (2 * n_neg), n / (2 * n_pos)]
    print(f"train n={n} pos={n_pos} neg={n_neg} class_w={class_w}", flush=True)

    model = build_mnv3_large(pretrained=not args.no_pretrained)
    for p in model.features.parameters():
        p.requires_grad = False
    frozen = not any(p.requires_grad for p in model.features.parameters())
    print(f"Stage A: base frozen={frozen}", flush=True)

    model, hist_a = train_one_stage(model, train_items, args, args.epochs,
                                    args.lr, "A", args.seed, class_w)
    torch.save(model.state_dict(), out / "stage_A.pt")
    state_a = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    hist_b = []
    if args.stage == "AB":
        n_unfrozen, first_idx = unfreeze_last_layers(model)
        print(f"Stage B: unfreeze features[{first_idx}:] + head, "
              f"{n_unfrozen:,} params, lr={args.lr_b}", flush=True)
        model, hist_b = train_one_stage(model, train_items, args, args.ft_epochs,
                                        args.lr_b, "B", args.seed, class_w,
                                        init_weights=state_a)
    torch.save(model.state_dict(), out / "stage_final.pt")

    # ---- val: thresholds chosen here only
    v_names, v_y, v_s = infer(model, val_items, args)
    op = recall_first_threshold(v_y, v_s)
    thr_main = op["0.97"] if op["0.97"] is not None else op["0.95"]
    if thr_main is None:
        thr_main = 0.5
    val_metrics = {
        "thr_0.5": evaluate_split(v_names, v_y, v_s, 0.5),
        "thr_recall_first": evaluate_split(v_names, v_y, v_s, thr_main),
        "recall_first_thresholds": op,
    }
    save_per_image(out / "per_image_val.csv", v_names, v_y, v_s)

    # ---- test: once
    t_names, t_y, t_s = infer(model, test_items, args)
    test_metrics = {
        "thr_0.5": evaluate_split(t_names, t_y, t_s, 0.5),
        "thr_recall_first": evaluate_split(t_names, t_y, t_s, thr_main),
    }
    save_per_image(out / "per_image_test.csv", t_names, t_y, t_s)

    summary = {
        "config": {k: getattr(args, k) for k in
                   ("data", "stage", "imgsz", "batch", "epochs", "ft_epochs",
                    "steps", "lr", "lr_b", "seed", "device")},
        "class_weights_neg_pos": class_w,
        "train_counts": {"n": n, "pos": n_pos, "neg": n_neg},
        "chosen_on_val": {"recall_first_threshold": thr_main, **op},
        "val": val_metrics,
        "test": test_metrics,
        "history": hist_a + hist_b,
        "torch_version": torch.__version__,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"val_0.5": {k: round(v, 4) for k, v in val_metrics["thr_0.5"].items()
                                  if isinstance(v, float)},
                      "test_0.5": {k: round(v, 4) for k, v in test_metrics["thr_0.5"].items()
                                   if isinstance(v, float)},
                      "chosen_thr": thr_main}, indent=2))
    print(f"wrote {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
