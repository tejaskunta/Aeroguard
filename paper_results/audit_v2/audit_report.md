# Phase 1 dataset audit report

Generated: 2026-10-14T15:38:12Z  ·  elapsed 15.7s
Dataset: `C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2`

## Split counts

| Split | Images | Instances | Empty labels | Missing labels | Zero-box images |
|---|---:|---:|---:|---:|---:|
| train | 3907 | 11550 | 20 | 0 | 20 |
| valid | 200 | 456 | 7 | 0 | 7 |
| test | 200 | 264 | 5 | 0 | 5 |

## Exact duplicates (MD5)

- Cross-split groups: **0** → `exact_dup_cross.csv`
- Within-split groups: 0 → `exact_dup_within.csv`

## Near-duplicates (64-bit pHash)

Each valid/test image vs all train images under **8 dihedral transforms** of train.

| Threshold | Query images with a train neighbour |
|---|---:|
| Hamming <=2 | 129 |
| Hamming <=4 | 278 |
| Hamming <=6 | 379 |
| Hamming <=8 | 393 |

- Among Hamming≤8 pairs, **same Roboflow base key**: 0
- Base-key collisions (filename, independent of pixels): train∩valid=0, train∩test=0, valid∩test=0
- test-vs-valid pairs with pHash≤8: 187
- Train unique base names: 1844; multi-export bases: 867; max copies of one base: 14

Montages: `montages/near_dup_top40.jpg`, `montages/base_key_collisions_test_train.jpg`

## Sources

| Source | Split | Images | Pos | Neg | Instances | Frac pos |
|---|---|---:|---:|---:|---:|---:|
| Jan_lwir | test | 10 | 10 | 0 | 77 | 1.000 |
| Jan_lwir | train | 100 | 100 | 0 | 740 | 1.000 |
| Jan_lwir | valid | 45 | 45 | 0 | 308 | 1.000 |
| May_lwir | train | 776 | 776 | 0 | 7706 | 1.000 |
| ds0 | test | 161 | 158 | 3 | 158 | 0.981 |
| ds0 | train | 2589 | 2569 | 20 | 2643 | 0.992 |
| ds0 | valid | 154 | 147 | 7 | 147 | 0.955 |
| ds1 | test | 29 | 27 | 2 | 29 | 0.931 |
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

**FAIL** — rule: 0 exact cross-split dups AND ≤0.5% of val/test with near-dup in train (Hamming≤2 auto-count here: 129/400 = 32.25%).

## Unresolved / next

1. Manual review of `montages/near_dup_top40.jpg` and base-key collisions.
2. Decide May/Jan sensor type for the paper dataset table.
3. Collect negatives for test/valid or accept that image-level comparison is limited.
4. Phase 1 gate: user approval required before any re-split or training.
