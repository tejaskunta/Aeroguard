#!/usr/bin/env python3
"""MSE-threshold sensitivity for the Phase-2 near-duplicate filter.

Recomputes downsampled-RGB MSE for every pHash candidate from the
pre-removal audit list, using the original v1 image store (which still
contains the 30 eval images that were removed from v2). Reports how many
eval images would be dropped at alternative MSE thresholds.

Outputs paper_results/analysis/mse_sensitivity.{csv,json,md}
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
V1 = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined")
V2 = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")
NEAR = ROOT / "paper_results" / "audit_v2" / "near_dup_pairs.csv"
OUT = ROOT / "paper_results" / "analysis"
THRESHOLDS = [250, 500, 750, 900, 1100, 1300, 1500, 2000]


def small_rgb(path: Path, n: int = 32) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((n, n), Image.BILINEAR)
        return np.asarray(im, dtype=np.float64)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def index_split(root: Path, split: str) -> dict[str, Path]:
    d = root / split / "images"
    if not d.is_dir():
        return {}
    return {p.name: p for p in d.iterdir() if p.is_file()}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    v1 = {}
    for s in ("train", "valid", "test"):
        v1.update(index_split(V1, s))
    current_eval = set(index_split(V2, "valid")) | set(index_split(V2, "test"))

    best: dict[str, dict] = {}
    with open(NEAR, newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if int(e["min_hamming"]) > 2:
                continue
            if e["query_split"] not in ("valid", "test"):
                continue
            qn, tn = e["query_name"], e["train_name"]
            if qn not in v1 or tn not in v1:
                continue
            m = mse(small_rgb(v1[qn]), small_rgb(v1[tn]))
            cur = best.get(qn)
            if cur is None or m < cur["mse32"]:
                best[qn] = {
                    "query_split": e["query_split"],
                    "query_name": qn,
                    "train_name": tn,
                    "hamming": int(e["min_hamming"]),
                    "mse32": round(m, 2),
                    "removed": int(qn not in current_eval),
                }

    rows = sorted(best.values(), key=lambda r: r["mse32"])
    n_removed = sum(r["removed"] for r in rows)
    removed_mse = [r["mse32"] for r in rows if r["removed"]]
    survivors = [r["mse32"] for r in rows if not r["removed"]]

    table = []
    for t in THRESHOLDS:
        flagged = [r for r in rows if r["mse32"] <= t]
        tp = sum(r["removed"] for r in flagged)
        table.append({
            "mse_threshold": t,
            "flagged_total": len(flagged),
            "true_leaks_captured": tp,
            "removed_still_flagged": tp,
            "survivors_flagged": len(flagged) - tp,
            "recall_of_30_leaks": round(tp / max(n_removed, 1), 3),
        })

    summary = {
        "candidates_ham_le_2_eval": len(rows),
        "eval_images_removed_in_v2": n_removed,
        "removed_by_split": {
            sp: sum(1 for r in rows if r["removed"] and r["query_split"] == sp)
            for sp in ("valid", "test")},
        "candidates_by_split": {
            sp: sum(1 for r in rows if r["query_split"] == sp)
            for sp in ("valid", "test")},
        "frozen_threshold": 900,
        "max_mse_among_removed": max(removed_mse) if removed_mse else None,
        "nearest_surviving_mse": min(survivors) if survivors else None,
        "margin_ratio_900_to_nearest_survivor": (
            round(min(survivors) / 900.0, 3) if survivors else None),
        "sweep": table,
    }

    with open(OUT / "mse_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    with open(OUT / "mse_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Near-duplicate MSE threshold sensitivity",
        "",
        f"Candidates (pHash Hamming <= 2, eval splits): **{len(rows)}**",
        f"Eval images actually removed at MSE <= 900: **{n_removed}**",
        f"Largest MSE among removed leaks: **{summary['max_mse_among_removed']}**",
        f"Smallest MSE among surviving candidates: **{summary['nearest_surviving_mse']}**",
        f"Margin (nearest survivor / 900): **{summary['margin_ratio_900_to_nearest_survivor']}x**",
        "",
        "| MSE <= | flagged | true leaks captured | survivors flagged |",
        "|---:|---:|---:|---:|",
    ]
    for r in table:
        lines.append(f"| {r['mse_threshold']} | {r['flagged_total']} | "
                     f"{r['removed_still_flagged']} | {r['survivors_flagged']} |")
    (OUT / "mse_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[2:]))
    print(f"wrote {OUT / 'mse_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
