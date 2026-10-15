# Stage-2 crop dataset recipe (plan §Phase 5.1)

Script: `phase5_build_crops.py` (single source of truth; rerun regenerates the
dataset bit-for-bit with `--seed 0`). Weights/images themselves are gitignored
(`*.pt`, `crops/**/*.jpg`); `manifest.csv`, `stats.json` and this recipe are
committed.

## Crop geometry (positives and negatives alike)

- Square crop, side = max(box_w, box_h) × (1 + 2×0.2) → **20 % context padding**
  (same `pad=0.2` used at inference in `two_stage_eval.build_records`).
- Positives: every GT box with mild jitter — shift ≤10 % of box size,
  scale U(0.9, 1.2).
- Resized to **224×224**.

## Composition (seed 0, frozen v2 split)

| split | positives | neg random-bg | neg YOLO-FP | neg empty-image | neg total |
|---|---|---|---|---|---|
| train | 11550 | 11661 | 3730 | 20 | 15411 |
| valid | 419 | 522 | 0 | 7 | 529 |

- Negative kinds (plan): (a) random background crops with IoU < 0.1 to any GT;
  (b) **YOLO false positives** on train images at conf 0.05 from the *early*
  checkpoint `epoch20.pt` (needs `save_period=5` from Phase 3 — FP diversity
  without leaking the final model); (c) one crop per empty-label image
  (20 train / 7 valid).
- Total: 27909 crops; `manifest.csv` columns
  `path,label,split,kind,source_image,box`.

## Training (see `phase5_train_stage2.py`, plan §Phase 5.2)

Whole-net fine-tune of `mnv3_shared.build_mnv3_large`, lr 2e-5, batch 32,
per-epoch ratio resampling to 1:1 (chosen on **val**: AUC 0.998 / F1 0.938 vs
ratio-3 0.997 / 0.918), patience 6, mild augmentation (flip, rotation, light
brightness). Same RGB /255 / ImageNet mean-std path as the baseline
(`mnv3_shared.preprocess_uint8`).
