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
| `paper/` | **Manuscript** (LaTeX, IEEEtran two-column conference format) + `refs.bib` + build `Makefile` |
| `models/` | **All trained weights** (YOLO 3 seeds, stage-2 3 seeds, original v1) + SHA256 + README + `NOTICE.md` |
| `reports/` | **PDF reports** — per-seed training reports + multi-phase research summary |
| `paper_results/` | Tables (T1–T6 .md/.tex), figures, per-seed eval JSONs, Grad-CAM, visualizations |
| `docs/reports/` | Pre-v2 SafeMine report (historical) |
| `RUN_LOG.md` | Every run + every test-set touch (numbered #1–#18) |
| `DECISIONS.md` | D001–D023 research decisions with rationale |
| `ENVIRONMENT.txt` | Pinned deps, GPU, commit, eval protocol |
| `src/common/` | Shared library: `two_stage_eval.py`, `mnv3_shared.py` |
| `src/training/` | `week3_train_yolov8.py`, `phase5_train_stage2.py` |
| `src/evaluation/` | `phase3_eval.py`, `per_source_report.py`, `label_fix_report.py` |
| `src/pipeline/` | `phase5_build_crops.py`, `phase5_pipeline.py` |
| `src/gradcam/` | `phase6_gradcam.py`, `phase6_e3_paste.py`, `phase6_e5_flags.py`, `phase6_e6_headtohead.py` |
| `src/baseline/` | `phase4_mnv3_baseline.py` |
| `src/analysis/` | `analysis_stats.py`, `dump_raw_stage2.py`, `make_ablation_matrix.py`, `make_leaky_vs_clean.py`, `make_mse_sensitivity.py` |
| `src/reporting/` | `make_tables.py`, `make_report_pdfs.py`, `make_test_vis.py`, `make_master_report.py` |
| `scripts/` | Dataset audit/resplit (`phase1_audit.py`, `phase2_*.py`), export/predict |
| `scripts/inspect/` | Ad-hoc JSON/crop inspection helpers (`dump_*.py`) |
| `tests/` | `pytest` suite (`conftest.py` puts `src/*` on the path) |
| `backend/`, `jetson/` | **Stale Week-1/2 AeroShield snapshot** — do not use |

> **`backend/` and `jetson/` are a frozen snapshot, not this project.** They were
> merged in at `9c1d287` from an early AeroShield state and have not been updated
> since. The real deployment system — FastAPI service with API-key auth,
> PostgreSQL/PostGIS, a RAG copilot, a React mission console, and an 11-module
> TensorRT edge pipeline — lives in the separate repository
> [tejaskunta/AeroShield](https://github.com/tejaskunta/AeroShield).
>
> The two projects are **different things**: AeroGuard (this repo) is the
> detection research paper; AeroShield is the deployment system built on top of
> it. Both serve the **same single-class** model (`Landmine`); AeroShield's
> `jetson/` layer exports this repo's `best.pt` to ONNX, builds a TensorRT engine
> on the Nano, and runs it. The only three-class labelling anywhere is
> `configs/data.yaml.example`, which is an unedited template, and the default
> label list of `MockDetector` (the GPU-less pipeline test double) — **not** the
> deployed detector.

Every script resolves the repo root from `Path(__file__)`, so all commands below
run from the repository root regardless of where the script lives.

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
(0.27 for seed 0; val-chosen). See `src/pipeline/phase5_pipeline.py`.

## Regenerating paper artifacts

```bash
python src/reporting/make_tables.py       # T1–T6 (.md/.tex) + figures/
python src/reporting/make_report_pdfs.py  # reports/*.pdf (reads committed JSON only)
cd paper && make                          # paper/main.pdf
```

Tests (no GPU or dataset needed for the pure-metric cases):

```bash
python -m pytest tests/
```

## Dataset

Not in git (size/licensing). Trained on `landmine_combined_v2`:
train 3907 / valid 181 / test 189 images, 0 true cross-split leaks.
Split manifest sha256: see `paper_results/split_v2/sha256.txt`.
Original sources: Roboflow landmine datasets (ds0, ds1, Jan LWIR, May, …).
Full provenance, licence status, and the manifest-only redistribution policy are
in [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

## License

Code, split manifest, tables and figures: **AGPL-3.0** (see `LICENSE`).
The released YOLO checkpoints are derivative works of Ultralytics YOLOv8 and are
also AGPL-3.0; the stage-2 checkpoints are fine-tuned torchvision MobileNetV3-Large.
See [`models/NOTICE.md`](models/NOTICE.md). **No dataset images are redistributed.**

## Protocol (non-negotiable)

1. Thresholds chosen on **val only**; test evaluated once under frozen protocol.
2. Every test touch logged in `RUN_LOG.md`.
3. Every paper number comes from a committed script + JSON.
4. Original dataset / v1 runs never modified.

> The rules above govern *what* is run, not *where* the script lives, so the
> `src/<subpackage>/` layout does not affect the protocol. Scripts were later
> collected under `src/` and `scripts/`; `RUN_LOG.md` entries have been updated
> to the current paths (see the note at the top of that file).
5. Labels fixed only if objectively wrong (2 test boxes; before/after reported).

## History

Full git history (Phase 0 inventory → Phase 6 Grad-CAM → models/PDF release)
is preserved. Tags: `pre-audit-v2`, `phase1-audit-fail`, `phase2-v2-freeze`.

Older v1 report: `safemine_report.pdf` (yolov8m, pre-v2-split — reference only).
