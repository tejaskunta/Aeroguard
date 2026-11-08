#!/usr/bin/env python3
"""A3 — independent near-duplicate check with pretrained ImageNet embeddings.

Purpose
-------
The frozen v2 split removed 30 eval images with a pHash (Hamming<=2) +
32x32-RGB MSE<=900 rule, and declared the remaining 99 pHash candidates
"terrain texture false positives". That verdict rests on the same two
hand-tuned features used to build the filter. This script checks it with a
completely different representation:

  * ResNet-50 (torchvision IMAGENET1K_V2, ImageNet weights, never trained
    on this dataset), 2048-d global-average-pooled embedding, L2-normed
  * cosine nearest neighbour of every valid/test image against every
    v2-train image (3907 x 370 exact search, no ANN approximation)

Populations scored
------------------
  30 eval images removed during split construction
        recovered from the v1 store (they are absent from v2); we check
        whether the embedding still sees a near-duplicate, and whether the
        pHash-proposed partner is even still in the v2 train split.
  99 residual pHash candidates that survived
        does the embedding agree they are unrelated (texture lookalikes)?
  all 370 eval images
        top-1 pair whose 32x32 MSE is > 900, i.e. not caught by the filter.

Counting rule (declared before looking at survivor cosines)
-----------------------------------------------------------
Leak-similarity threshold T = minimum cosine between a *known* removed leak
and its pHash-proposed train partner. A survivor is "flagged by the
embedding method" if its cosine >= T. T comes from known leaks only; no
detector metric is consulted and no threshold is tuned on test.

Outputs (paper_results/audit_v2/):
  embedding_near_dup_pairs.csv     all 370 top-1 pairs + MSE + pHash info
  embedding_removed_recheck.csv    the 30 removed images re-scored
  embedding_near_dup_summary.json
  embedding_near_dup.md
  embedding_top50_montage.jpg      top-50 MSE-uncaught pairs side by side

Reads the test split. No detector inference, no model threshold selection,
no performance metric computed from it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parents[2]
V1 = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined")
V2 = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")
PRE_REMOVAL = ROOT / "paper_results" / "audit_v2" / "near_dup_pairs.csv"
OUT = ROOT / "paper_results" / "audit_v2"

MSE_THRESHOLD = 900.0
EMBED_SIZE = 256
BATCH = 48

MODEL: torch.nn.Module | None = None


# ---------------------------------------------------------------- embeddings
def build_model(device: torch.device) -> torch.nn.Module:
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    return m.eval().to(device)


@torch.no_grad()
def embed_paths(paths: list[Path], device: torch.device, verbose: bool = True) -> np.ndarray:
    global MODEL
    if MODEL is None:
        MODEL = build_model(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(paths), BATCH):
        arr = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((EMBED_SIZE, EMBED_SIZE), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            for p in paths[i : i + BATCH]
        ])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
        x = (x - mean) / std
        out.append(F.normalize(MODEL(x).flatten(1), dim=1).cpu().numpy())
        if verbose and (i // BATCH) % 10 == 0:
            print(f"  embedded {min(i + BATCH, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------- MSE (same implementation as make_mse_sensitivity.py)
def small_rgb(path: Path, n: int = 32) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((n, n), Image.BILINEAR)
        return np.asarray(im, dtype=np.float64)


def mse32(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def auroc(pos: list[float], neg: list[float]) -> float:
    """Probability a random positive scores above a random negative (ties=0.5)."""
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else (0.5 if p == n else 0.0)
    return s / (len(pos) * len(neg))


# ---------------------------------------------------------------- data index
def index_split(root: Path, split: str) -> dict[str, Path]:
    d = root / split / "images"
    return {p.name: p for p in d.iterdir() if p.is_file()} if d.is_dir() else {}


def v1_lookup() -> dict[str, tuple[str, Path]]:
    out: dict[str, tuple[str, Path]] = {}
    for s in ("train", "valid", "test"):
        for name, p in index_split(V1, s).items():
            out.setdefault(name, (s, p))
    return out


def phash_candidates() -> dict[str, dict]:
    """Best (lowest-hamming) pre-removal pHash candidate per eval query."""
    best: dict[str, dict] = {}
    with open(PRE_REMOVAL, newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e["query_split"] not in ("valid", "test"):
                continue
            h = int(e["min_hamming"])
            if h > 2:
                continue
            q = e["query_name"]
            if q not in best or h < best[q]["hamming"]:
                best[q] = {"hamming": h, "train_name": e["train_name"]}
    return best


# ---------------------------------------------------------------- montage
def montage(pairs: list[dict], path: Path, cell: int = 160, cols: int = 5) -> None:
    rows = (len(pairs) + cols - 1) // cols
    pad, label_h = 6, 30
    canvas = Image.new(
        "RGB",
        (cols * (cell * 2 + pad) + pad, rows * (cell + label_h + pad) + pad),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for k, r in enumerate(pairs):
        c, row = k % cols, k // cols
        x = pad + c * (cell * 2 + pad)
        y = pad + row * (cell + label_h + pad)
        for j, key in enumerate(("query_path", "train_path")):
            try:
                im = Image.open(r[key]).convert("RGB").resize((cell, cell), Image.BILINEAR)
            except Exception:
                im = Image.new("RGB", (cell, cell), (200, 200, 200))
            canvas.paste(im, (x + j * cell, y))
        draw.text(
            (x, y + cell + 2),
            f"{k + 1}. {r['query_name'][:20]} | cos {r['cosine']:.3f} | mse {r['mse32']:.0f}",
            fill=(0, 0, 0),
            font=font,
        )
    canvas.save(path, quality=90)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    tr = index_split(V2, "train")
    ev: dict[str, tuple[str, Path]] = {}
    for s in ("valid", "test"):
        for name, p in index_split(V2, s).items():
            ev[name] = (s, p)
    train_names = sorted(tr)
    eval_names = sorted(ev)
    print(f"v2 train={len(train_names)} v2 eval={len(eval_names)}")

    cand = phash_candidates()
    removed_names = sorted(q for q in cand if q not in ev)
    survivor_names = sorted(q for q in cand if q in ev)
    print(
        f"pHASH candidates={len(cand)} removed={len(removed_names)} "
        f"survivors={len(survivor_names)}"
    )

    print("[1/3] embedding v2 train ...")
    Xtr = embed_paths([tr[n] for n in train_names], device)
    print("[2/3] embedding v2 eval ...")
    Xev = embed_paths([ev[n][1] for n in eval_names], device)

    print("[3/3] embedding removed eval images (from v1 store) ...")
    v1 = v1_lookup()
    have = [n for n in removed_names if n in v1]
    Xrm = embed_paths([v1[n][1] for n in have], device, verbose=False)
    print(f"  found {len(have)}/{len(removed_names)} in v1 store")

    # ---- top-1 pair for every eval image
    S = Xev @ Xtr.T
    best_j = S.argmax(axis=1)
    best_cos = S[np.arange(len(eval_names)), best_j]

    rows = []
    for i, q in enumerate(eval_names):
        t = train_names[int(best_j[i])]
        rows.append(
            {
                "query_split": ev[q][0],
                "query_name": q,
                "train_name": t,
                "cosine": round(float(best_cos[i]), 6),
                "mse32": round(mse32(small_rgb(ev[q][1]), small_rgb(tr[t])), 2),
                "is_phash_candidate": int(q in cand),
                "phash_train_name": cand[q]["train_name"] if q in cand else "",
                "phash_hamming": cand[q]["hamming"] if q in cand else "",
                "agrees_with_phash": int(q in cand and cand[q]["train_name"] == t),
                "query_path": str(ev[q][1]),
                "train_path": str(tr[t]),
            }
        )
    by_name = {r["query_name"]: r for r in rows}

    uncaught = sorted([r for r in rows if r["mse32"] > MSE_THRESHOLD], key=lambda r: -r["cosine"])
    montage(uncaught[: args.top], out / "embedding_top50_montage.jpg")

    # ---- the removed eval images, re-checked against the v2 train split
    idx = {n: i for i, n in enumerate(train_names)}
    S_rm = Xrm @ Xtr.T
    removed_rows = []
    for i, name in enumerate(have):
        j = int(S_rm[i].argmax())
        ph = cand[name]["train_name"]
        removed_rows.append(
            {
                "name": name,
                "v1_split": v1[name][0],
                "nn_v2_train": train_names[j],
                "cosine_nn": round(float(S_rm[i, j]), 6),
                "phash_train_name": ph,
                "phash_partner_still_in_v2_train": int(ph in idx),
                "cosine_phash_partner": (
                    round(float(S_rm[i, idx[ph]]), 6) if ph in idx else ""
                ),
                "mse_nn": round(mse32(small_rgb(v1[name][1]), small_rgb(tr[train_names[j]])), 2),
            }
        )

    # ---- leak-similarity threshold from known leaks only
    known_cos = [r["cosine_phash_partner"] for r in removed_rows if r["cosine_phash_partner"] != ""]
    T = float(min(known_cos)) if known_cos else None

    survivor_rows = [by_name[q] for q in survivor_names if q in by_name]
    flagged_survivors = (
        [r for r in survivor_rows if T is not None and r["cosine"] >= T] if T is not None else []
    )
    survivors_agreeing = sum(r["agrees_with_phash"] for r in survivor_rows)
    S_cos = [r["cosine"] for r in survivor_rows]

    summary = {
        "method": "ResNet-50 IMAGENET1K_V2 GAP embedding, cosine NN over the 3907 v2 train images",
        "embed_input": f"{EMBED_SIZE}x{EMBED_SIZE}",
        "eval_images": len(eval_names),
        "train_images": len(train_names),
        "mse_threshold_reused": MSE_THRESHOLD,
        "pairs_mse_above_900": len(uncaught),
        "pairs_mse_below_900": len(rows) - len(uncaught),
        "max_cosine_among_mse_uncaught": uncaught[0]["cosine"] if uncaught else None,
        "mse_uncaught_at_cosine_ge_0_95": sum(1 for r in uncaught if r["cosine"] >= 0.95),
        "removed_total": len(removed_names),
        "removed_found_in_v1": len(have),
        "removed_missing_from_v1": len(removed_names) - len(have),
        "removed_partner_still_in_v2_train": sum(
            r["phash_partner_still_in_v2_train"] for r in removed_rows
        ),
        "leak_similarity_threshold_from_known_leaks": T,
        "survivors_total": len(survivor_names),
        "survivors_flagged_by_embedding": len(flagged_survivors),
        "survivors_where_embedding_agrees_with_phash_choice": survivors_agreeing,
        "survivor_cosine_min_median_max": (
            [
                round(float(np.min(S_cos)), 6),
                round(float(np.median(S_cos)), 6),
                round(float(np.max(S_cos)), 6),
            ]
            if S_cos
            else None
        ),
    }

    with open(out / "embedding_near_dup_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with open(out / "embedding_removed_recheck.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(removed_rows[0]))
        w.writeheader()
        w.writerows(removed_rows)

    with open(out / "embedding_near_dup_summary.json", "w", encoding="utf-8") as f:
        json.dump({**summary, "removed_rows": removed_rows}, f, indent=2)

    md = [
        "# A3 — independent near-duplicate check with ImageNet embeddings",
        "",
        "Model: ResNet-50 `IMAGENET1K_V2` (frozen ImageNet weights), 2048-d GAP embedding,",
        f"{EMBED_SIZE}x{EMBED_SIZE} RGB input, cosine similarity, exact search over the",
        f"{summary['train_images']} v2 **train** images. 32x32 RGB MSE uses the same code path as",
        "`src/analysis/make_mse_sensitivity.py`.",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Eval images checked | {summary['eval_images']} |",
        f"| Train images searched | {summary['train_images']} |",
        f"| Top-1 pairs with MSE > 900 (frozen filter would not catch) | {summary['pairs_mse_above_900']} |",
        f"| Top-1 pairs with MSE <= 900 | {summary['pairs_mse_below_900']} |",
        f"| Highest cosine among MSE-uncaught pairs | {summary['max_cosine_among_mse_uncaught']} |",
        f"| MSE-uncaught pairs at cosine >= 0.95 | {summary['mse_uncaught_at_cosine_ge_0_95']} |",
        "",
        "### The 30 eval images removed during split construction",
        "",
        f"- Recovered from the v1 store: **{summary['removed_found_in_v1']}/{summary['removed_total']}**.",
        f"- pHash-proposed partner still in the v2 train split: **{summary['removed_partner_still_in_v2_train']}/{summary['removed_found_in_v1']}**.",
        f"- Minimum cosine to the pHash-proposed partner: **{summary['leak_similarity_threshold_from_known_leaks']}**.",
        "",
        "### The 99 residual pHash candidates (declared texture false positives)",
        "",
        f"- Embedding cosine to own top-1 v2-train image, min / median / max: {summary['survivor_cosine_min_median_max']}",
        f"- Flagged at the leak-derived threshold: **{summary['survivors_flagged_by_embedding']}/{summary['survivors_total']}**",
        f"- Embedding top-1 equals the pHash-proposed train image: **{summary['survivors_where_embedding_agrees_with_phash_choice']}/{summary['survivors_total']}**",
        "",
        f"Montage of the top {args.top} MSE-uncaught pairs by cosine: `embedding_top50_montage.jpg`.",
        "",
        "## Top 15 MSE-uncaught pairs",
        "",
        "| # | query | split | train NN | cosine | MSE |",
        "|---:|---|---|---|---:|---:|",
    ]
    for k, r in enumerate(uncaught[:15], 1):
        md.append(
            f"| {k} | `{r['query_name']}` | {r['query_split']} | `{r['train_name']}` | "
            f"{r['cosine']:.4f} | {r['mse32']:.1f} |"
        )
    (out / "embedding_near_dup.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())