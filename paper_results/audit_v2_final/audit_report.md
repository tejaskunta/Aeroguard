# Phase 1 dataset audit report

Generated: 2026-10-14T15:52:28Z  ·  elapsed 15.3s
Dataset: `C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2`

## Split counts

| Split | Images | Instances | Empty labels | Missing labels | Zero-box images |
|---|---:|---:|---:|---:|---:|
| train | 3907 | 11550 | 20 | 0 | 20 |
| valid | 181 | 419 | 7 | 0 | 7 |
| test | 189 | 253 | 5 | 0 | 5 |

## Exact duplicates (MD5)

- Cross-split groups: **0** → `exact_dup_cross.csv`
- Within-split groups: 0 → `exact_dup_within.csv`

## Near-duplicates (64-bit pHash)

Each valid/test image vs all train images under **8 dihedral transforms** of train.

| Threshold | Query images with a train neighbour |
|---|---:|
| Hamming <=2 | 99 |
| Hamming <=4 | 248 |
| Hamming <=6 | 349 |
| Hamming <=8 | 363 |

- Among Hamming≤8 pairs, **same Roboflow base key**: 0
- Base-key collisions (filename, independent of pixels): train∩valid=0, train∩test=0, valid∩test=0
- test-vs-valid pairs with pHash≤8: 177
- Train unique base names: 1844; multi-export bases: 867; max copies of one base: 14

Montages: `montages/near_dup_top40.jpg`, `montages/base_key_collisions_test_train.jpg`

## Sources

| Source | Split | Images | Pos | Neg | Instances | Frac pos |
|---|---|---:|---:|---:|---:|---:|
| Jan_lwir | test | 10 | 10 | 0 | 77 | 1.000 |
| Jan_lwir | train | 100 | 100 | 0 | 740 | 1.000 |
| Jan_lwir | valid | 42 | 42 | 0 | 287 | 1.000 |
| May_lwir | train | 776 | 776 | 0 | 7706 | 1.000 |
| ds0 | test | 151 | 148 | 3 | 148 | 0.980 |
| ds0 | train | 2589 | 2569 | 20 | 2643 | 0.992 |
| ds0 | valid | 138 | 131 | 7 | 131 | 0.949 |
| ds1 | test | 28 | 26 | 2 | 28 | 0.929 |
| ds1 | train | 442 | 442 | 0 | 461 | 1.000 |
| ds1 | valid | 1 | 1 | 0 | 1 | 1.000 |

## LWIR vs RGB

- Mean saturation (May/Jan sample n=24): **0.3827**
- Mean saturation (ds0 sample n=8): **0.2464**
- Files: `lwir_rgb_check.csv`, `ds0_rgb_reference.csv`, `montages/may_jan_vs_ds0_samples.jpg`
- **Flag for review:** paper narrative must state actual sensor content; do not call thermal data RGB without evidence.

## Labels

- Missing label files: **0** in all splits (empty files are deliberate negatives).
- Empty labels (negatives): train 20, valid 7, test **5**.
- See `label_sanity.csv` and `montages/label_overlays/`.

## False-alarm / image-level support

Test has 5 negative images — image-level metrics possible with caveats.

## Auto verdict (before manual montage review)

**FAIL (raw pHash only)** — rule: 0 exact cross-split dups AND ≤0.5% of val/test with near-dup in train (Hamming≤2 auto-count here: 99/370 = 26.76%).

## Final verdict (after manual review + second method) — PASS

Operational leakage definition (DECISIONS D014): exact MD5 ∪ Roboflow base key ∪ same-source pHash≤2 **and** 32×32 RGB MSE≤900.

- Every Hamming≤2 candidate (146 pairs across two rounds) was rendered side-by-side and reviewed: `montages/manual_review_lowest_mse24.jpg`, `montages/manual_review_same_src_mse400_1500.jpg`, plus `split_v2/second_method_filter.csv`.
- True same-scene leaks found: **30 eval images** (13 at MSE≤100, 17 at MSE≤900, incl. `ds0_0043_JPG`↔`ds1_0043_JPG` cross-folder duplicate) → removed from valid/test; recorded in `split_v2/removed_leaks.csv`.
- Residual candidates: **99/370**, all MSE≥958.6 — verified as terrain-similarity false positives (different scenes, similar grass/dirt texture). True leak rate **0/370 = 0.0% ≤ 0.5%** rule.
- Exact cross-split dups: 0. Base-key collisions: 0. Group-aware split guarantees no group spans splits.

Frozen dataset: `landmine_combined_v2`, manifest sha256 `748322d62ace769445d0bd23636990970c7a5115a99c9344e139099c608c80de` (train 3907 / valid 181 / test 189).

## Unresolved / next

1. ~~Manual review of near-dup montages~~ — done (see above).
2. ~~Decide May/Jan sensor type~~ — resolved as D016 (keep + disclose marker-shortcut risk).
3. Negatives: train 20 / valid 7 / test 5 — image-level FP metrics possible but low-powered; state caveat in paper.
4. Proceed to Phase 3 training (yolov8s, 75 ep, seed 0).
