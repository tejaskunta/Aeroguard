# Phase 0 inventory — 2026-10-14

## Dataset

| Item | Value |
|---|---|
| Root | `C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined` |
| data.yaml | `path` absolute; `train: train/images`; `val: valid/images`; `test: test/images`; nc=1; names=['Landmine'] |
| train | 5723 images, 5723 labels, 11691 box lines, 48 empty labels, 0 missing labels |
| valid | 455 images, 455 labels, 2053 box lines, 1 empty label, 0 missing labels |
| test | 221 images, 221 labels, 966 box lines, **0 empty labels**, 0 missing labels |
| quality_reports/ | `dataset_statistics_report.md` (6399 imgs, 14710 instances), `duplicate_files.csv` (within-split only), plots |

### Naming patterns

| Prefix | train | valid | test | Notes |
|---|---:|---:|---:|---|
| `ds0_*` | 2787 | 265 | 134 | Roboflow export `ds0_####_jpg.rf.<md5>.jpg` |
| `ds1_*` | 2280 | 0 | 0 | Roboflow export; **train only** in this split |
| `May_*_lwir_*` | 549 | 157 | 70 | Name contains `lwir` — thermal? See Phase 1 |
| `Jan_*_lwir_*` | 107 | 33 | 17 | Name contains `lwir` — thermal? See Phase 1 |

Cross-split shared Roboflow base names (before `.rf.`): train∩valid=52, train∩test=37, valid∩test=12.

## Original run (v1)

| Item | Value |
|---|---|
| Folder | `SafeMineAI-main\runs\detect\runs\detect\landmine_yolov8m_75ep\` |
| Model | yolov8m.pt, 75 epochs, imgsz 640, batch 8, seed 0, patience 20, save_period **-1** |
| Finished | 2026-08-15T10:52:57Z, 15h 16m 42s |
| Val @ conf 0.25 | P 0.9414, R 0.9537, mAP50 0.9583, mAP50-95 0.6239 |
| Test @ conf 0.25 | P 0.9655, R 0.9572, mAP50 0.9718, mAP50-95 0.6360 |
| Weights | best.pt, last.pt, best.onnx under `weights/`; copies of best.* also under repo `weights/` |

## Week 2 pipeline (on disk; plan said “not in repo”)

`C:\Users\Delta\Downloads\Landmine_project\safemine_week2_pipeline\scripts\`

0. `00_merge_sources.py` — pools Roboflow train/valid/test; prefixes source names  
1. `01_dedup_and_clean.py` — MD5 + 8×8 average-hash **within** the merged pool  
2. `02_convert_annotations.py`  
3. `03_augment.py` — rot90/180/270, flip, brightness, zoom  
4. `04_split_dataset.py` — **random image-level** split seed 42 **after** augment  
5. `05_generate_stats_report.py`

Confirmed: augment runs **before** split; split is not group-aware → leakage risk is structural.

## SafeMineAI-main scripts

| File | Role |
|---|---|
| `week3_train_yolov8.py` | train/val/export; has --model --seed --conf --name; **no save_period flag** |
| `scripts/verify_dataset.py` | label/path checks; no perceptual dedup |
| `jetson/infer_trt.py` | letterboxes to **engine** in_h/in_w (not hardcoded 320 in this tree) |
| `jetson/gradcam.py` | **MISSING** |
| `two_stage_eval.py`, `test_two_stage.py` | **MISSING** (Phase 5) |

## Hardware / software

- RTX 4070 Laptop 8188 MiB, driver 581.42  
- Python 3.11.9, torch 2.6.0+cu124, CUDA 12.4 available, Ultralytics 8.4.117  
- Full freeze: `paper_results/ENVIRONMENT.txt`

## Backup verification

`paper_results/original_run/backup_verify.json` — SHA-256 match for best.pt, last.pt, best.onnx, args.yaml, results.csv, safemine_summary.json.
