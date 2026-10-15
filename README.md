# AeroGuard — Landmine Detection Research (SafeMine v2)

Research code + models + reports for the SafeMine / AeroGuard landmine-detection
project. YOLOv8s detector + MobileNetV3 two-stage pipeline, trained and evaluated
under a frozen leakage-free split with a strict val-only threshold protocol.

**Scope:** RGB/thermal imagery detects **visible landmine-like objects and surface
indicators only** — buried objects are invisible to the camera (design constraint).

## Results (frozen v2 test, n=189)

| System | Metric | Value |
|---|---|---|
| YOLOv8s (3 seeds) | test mAP50 | **0.977 ± 0.003** |
| Two-stage arm C (seed 0) | FP/img @ R≈0.96 | **0.153** (vs YOLO-only 0.407) |
| Two-stage C−B bootstrap | Δprecision 95% CI | [0.065, 0.186] (excludes 0) |
| Grad-CAM saliency | energy enrichment | 86× YOLO vs 1.8× MNv3 baseline |

## Repository map

| Path | Contents |
|---|---|
| `models/` | **All trained weights** (YOLO 3 seeds, stage-2 3 seeds, original v1) + SHA256 + README |
| `reports/` | **PDF reports** — per-seed training reports + multi-phase research summary |
| `paper_results/` | Tables (T1–T6 .md/.tex), figures, per-seed eval JSONs, Grad-CAM, visualizations |
| `RUN_LOG.md` | Every run + every test-set touch (numbered #1–#14) |
| `DECISIONS.md` | D001–D023 research decisions with rationale |
| `ENVIRONMENT.txt` | Pinned deps, GPU, commit, eval protocol |
| Training/eval scripts | `week3_train_yolov8.py`, `phase3_eval.py`, `phase4_mnv3_baseline.py`, `phase5_pipeline.py`, `phase6_*.py` |
| Table/figure/PDF generators | `make_tables.py`, `make_report_pdfs.py`, `make_test_vis.py` |

## Quick start (inference)

```bash
git clone https://github.com/tejaskunta/Aeroguard.git && cd Aeroguard
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install ultralytics==8.4.117 torch==2.6.0

python -c "
from ultralytics import YOLO
m = YOLO('models/yolo_s_seed0/best.pt')
r = m.predict('path/to/image.jpg', conf=0.02)  # frozen OP conf
print(r[0].boxes)
"
```

Two-stage arm C: run YOLO at the seed's OP conf, crop each box + 20% context to
224×224, score with `models/stage2_s0/best.pt` (sigmoid), keep if score ≥ τ
(0.27 for seed 0; val-chosen). See `phase5_pipeline.py`.

## Regenerating paper artifacts

```bash
python make_tables.py         # T1–T6 (.md/.tex) + figures/
python make_report_pdfs.py    # reports/*.pdf (reads committed JSON only)
```

## Dataset

Not in git (size/licensing). Trained on `landmine_combined_v2`:
train 3907 / valid 181 / test 189 images, 0 true cross-split leaks.
Split manifest sha256: see `paper_results/split_v2/sha256.txt`.
Original sources: Roboflow landmine datasets (ds0, ds1, Jan LWIR, May, …).

## Protocol (non-negotiable)

1. Thresholds chosen on **val only**; test evaluated once under frozen protocol.
2. Every test touch logged in `RUN_LOG.md`.
3. Every paper number comes from a committed script + JSON.
4. Original dataset / v1 runs never modified.
5. Labels fixed only if objectively wrong (2 test boxes; before/after reported).

## History

Full git history (Phase 0 inventory → Phase 6 Grad-CAM → models/PDF release)
is preserved. Tags: `pre-audit-v2`, `phase1-audit-fail`, `phase2-v2-freeze`.

Older v1 report: `safemine_report.pdf` (yolov8m, pre-v2-split — reference only).
