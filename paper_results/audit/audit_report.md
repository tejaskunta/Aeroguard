# Phase 1 dataset audit report

Generated: 2026-10-14T14:32:07Z  ·  manual review same day  
Dataset: `C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined`  
Scripts: `scripts/phase1_audit.py`, `scripts/verify_dataset.py`  
Outputs: `paper_results/audit/`

---

## Verdict

**FAIL — do not train on this split.** Go to Phase 2 (group-aware re-split) after user approval.

Reasons (any one is sufficient; all four present):

1. **Structural leakage:** Week 2 pipeline runs `03_augment.py` then `04_split_dataset.py` (random image-level split). Augmented copies of one source image can land in train and test.
2. **Cross-split base-name collisions:** 76 Roboflow base names appear in more than one split (train∩valid 37, train∩test 31, valid∩test 10). Montage `base_key_collisions_test_train.jpg` shows the **same scene, same mine placement, different Roboflow re-export** — not different photos.
3. **Near-duplicate rate far above the 0.5% gate:** 431/676 val+test images (63.8%) have a train neighbour at pHash Hamming ≤2 under 8 dihedral transforms. Even a conservative same-base check: 50/68 cross-split same-base query–train pairs have Hamming ≤8 after 8×8 dihedral (`same_base_phash.csv`).
4. **Test has zero negatives:** image-level accuracy / balanced accuracy / specificity and a fair FP-per-image story are not supported.

Exact MD5 cross-split duplicates: **0** (Roboflow re-exports differ in bytes; near-dup is the right detector).

---

## Split inventory

| Split | Images | Instances | Empty labels (negatives) | Missing labels | Zero-box images |
|---|---:|---:|---:|---:|---:|
| train | 5723 | 11691 | 48 | 0 | 48 |
| valid | 455 | 2053 | 1 | 0 | 1 |
| test | 221 | 966 | **0** | 0 | **0** |

`verify_dataset.py`: 0 errors, 3 warnings (tiny boxes: train 66, val 22, test 5 under 4 px at 640).

Class distribution: single class `Landmine` (id 0) only.

---

## Exact duplicates (MD5)

| Scope | Groups |
|---|---:|
| Cross-split | 0 |
| Within-split | 0 |

Files: `exact_dup_cross.csv`, `exact_dup_within.csv`.

---

## Near-duplicates (64-bit DCT pHash)

Method: each valid+test image vs all train images; train side compared under **8 dihedral transforms** (4 rotations × mirror). Histogram and pairs: `near_dup_histogram.csv`, `near_dup_pairs.csv`.

| Threshold | Query images with a train neighbour | Fraction of 676 |
|---|---:|---:|
| Hamming ≤ 0 | 233 | 34.5% |
| Hamming ≤ 2 | **431** | **63.8%** |
| Hamming ≤ 4 | 597 | 88.3% |
| Hamming ≤ 6 | 666 | 98.5% |
| Hamming ≤ 8 | 674 | 99.7% |

**Manual review of `montages/near_dup_top40.jpg`:** not all Hamming-0 pairs are pixel-identical; many are the **same physical scene** (same mine disc, same terrain texture, same shadow/tracks) with different crops, colour jitter, or Roboflow multi-export. That is still leakage for a detection paper (model memorises the site).

**Same Roboflow base key across splits (filename, no hashing):**  
train∩valid **37**, train∩test **31**, valid∩test **10** → 76 cross-split base groups.

**Same-base pHash (8×8 dihedral both sides)** — `same_base_phash.csv`:

| Hamming | Pairs |
|---:|---:|
| ≤2 | 1 |
| ≤4 | 12 |
| ≤8 | 50 |
| total comparisons | 68 |

Montage `montages/base_key_collisions_test_train.jpg` is conclusive: e.g. `ds0_0058` test vs train is the same grass field and mine with different crop/augmentation. **Filename base-key alone is enough to force group-aware splitting.**

Train multiplicity: **1896** unique base names, **1239** with multiple exports, **max 17** copies of one base in train alone.

test↔valid: 218 pairs with pHash ≤8 (eval splits also share scenes).

---

## Sources and shortcut risk

| Source | train | valid | test | Pos frac | Notes |
|---|---:|---:|---:|---:|---|
| ds0 | 2787 | 265 | 134 | ~0.99 | Roboflow RGB; most negatives (30+1) |
| ds1 | 2280 | 0 | 0 | 0.99 | Roboflow RGB; **train only** in this split |
| May_lwir | 549 | 157 | 70 | **1.000** | Name says lwir; content is colour RGB (see below) |
| Jan_lwir | 107 | 33 | 17 | **1.000** | Same |

- **ds1 never appears in valid/test** — test cannot measure generalisation to that source.
- **May/Jan are 100% positive** in every split: no mine-free images from those sources → source is a perfect correlate of “has mine” for those families (shortcut risk).
- May/Jan occupy a large share of valid (190/455) and test (87/221).

---

## LWIR vs RGB (May/Jan)

- Filenames contain `lwir`; montage `montages/may_jan_vs_ds0_samples.jpg` shows **colour RGB aerial photos** with **saturated marker pins** (yellow, blue, green, pink dots) next to mine-like objects — not greyscale thermal.
- Mean saturation May/Jan sample: **0.368** (higher than ds0 **0.258**).
- Resolution: May/Jan sample **640×512** (not 640×640).
- **Paper implication:** do not describe these as thermal/LWIR without provenance. Coloured markers are a **label shortcut** (model may learn “find the bright pin”). Flag for Phase 2: either exclude markers, document them as simulated training aids, or re-label without relying on markers.

Files: `lwir_rgb_check.csv`, `ds0_rgb_reference.csv`.

---

## Labels

- Missing label files: **0** (no silent “missing = background” bug in this tree).
- Empty labels = deliberate negatives: train 48, valid 1, test 0.
- No out-of-range / non-numeric / wrong-class issues reported by `verify_dataset.py`.
- Overlays: `montages/label_overlays/` (10 random positives per split, green boxes).

---

## False-alarm / image-level support

**Not supported on current test.** Zero mine-free test images.  
Plan Phase 3 step 5 and Phase 5 FP/image comparisons need negatives in test (and ideally more in valid). Phase 2 must place negatives (and ideally more collected empty-terrain images) into val/test.

---

## Unresolved issues

1. Week 2 raw (pre-augmentation) image folder not located under `Landmine_project` beyond the pipeline scripts — need path for Phase 2 rebuild from originals if possible.
2. Exact provenance/licence of ds0, ds1, May, Jan unknown (user questions).
3. Marker pins on May/Jan: keep, remove, or disclose?
4. ds1 train-only: re-split should put ds1 scenes into val/test or drop ds1 from headline claims.
5. Whether 48+1 negatives can be augmented into a usable eval set (weak) vs collecting real empty terrain (preferred).

---

## Files produced

| Path | Content |
|---|---|
| `audit_summary.json` | Machine-readable verdict inputs |
| `manifest_v1.csv` | 6399 rows |
| `near_dup_pairs.csv`, `near_dup_histogram.csv` | pHash NN results |
| `same_base_phash.csv` | Cross-split same-base distances |
| `exact_dup_*.csv` | MD5 |
| `source_counts.csv`, `lwir_rgb_check.csv`, `ds0_rgb_reference.csv` | Sources / sensor |
| `label_sanity.csv`, `verify_dataset.txt` | Labels |
| `montages/near_dup_top40.jpg` | Visual near-dups |
| `montages/base_key_collisions_test_train.jpg` | Same base, different split |
| `montages/may_jan_vs_ds0_samples.jpg` | Colour vs RGB check |
| `montages/label_overlays/` | Box sanity |
| `train_phash_cache.npz` | Cached train hashes |

---

## Phase 1 gate — stop

Awaiting user review and explicit approval before Phase 2 re-split or any training.
