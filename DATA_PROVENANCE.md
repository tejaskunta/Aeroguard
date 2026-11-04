# Dataset provenance, licensing, and redistribution policy

This file is the authoritative record for the image corpus used in the paper.
It replaces the empty template that previously sat at the top of
`docs/TRAINING_NOTES.md`.

## What is released

**Not the images.** The repository ships:

- `paper_results/split_v2/split_manifest.csv` — one row per image: path, split,
  source id, group id, label counts.
- `paper_results/split_v2/sha256.txt` — manifest digest
  `748322d62ace769445d0bd23636990970c7a5115a99c9344e139099c608c80de`.
- `paper_results/split_v2/removed_leaks.csv` — the 30 eval images removed during
  the audit, with the reason and the similarity score.
- `paper_results/audit_v2_final/*` — audit CSVs, montages and the audit report.

Anyone holding the original source archives can reconstruct the exact splits
byte-for-byte from the manifest plus the digest. No image is redistributed here
because (a) the source terms were never recorded at download time (below), and
(b) the per-source licences are unknown to this project.

## Source inventory

Five source identifiers appear in the manifest. All were obtained as public
Roboflow Universe landmine-detection exports and pooled by an upstream merge step
that is documented in `paper_results/audit/inventory.md` but not retained in this
repository; base keys retain the `*.rf.<md5>.jpg` export suffix.

| Source id | Split membership | Images | Pos | Neg | Instances | Content notes |
|---|---|---:|---:|---:|---:|---|
| `ds0` | train / valid / test | 2589 / 138 / 151 | 2569 / 131 / 148 | 20 / 7 / 3 | 2643 / 131 / 148 | RGB, top-down, single mine disc per frame |
| `ds1` | train / valid / test | 442 / 1 / 28 | 442 / 1 / 26 | 0 / 0 / 2 | 461 / 1 / 28 | RGB; largely a re-export of the `ds0` collection (see audit) |
| `May_lwir` | train only | 776 | 776 | 0 | 7706 | Filename says LWIR; **pixels are colour RGB with marker pins** (D012/D016). Multi-mine frames, up to 17 GT per image. |
| `Jan_lwir` | train / valid / test | 100 / 42 / 10 | 100 / 42 / 10 | 0 / 0 / 0 | 740 / 287 / 77 | Same caveat as `May_lwir`. Dominates test false positives. |

### Licence status — explicit

**The licence of each downloaded archive was not recorded when it was obtained,
and the source URLs are no longer available on this machine.** This is a
provenance defect of the project and is disclosed here rather than guessed at.

Consequences, which the paper must state:

1. No image is redistributed by this repository, in any form, including crops.
   The stage-2 crop cache (`paper_results/stage2/crops/`) is regenerable from the
   manifest but is itself not released.
2. Figures in `paper_results/figures/` that contain source imagery are published
   for scholarly comment and review, not as a dataset release.
3. Before any public release of images, a licence audit against the original
   Roboflow Universe project pages is required. Until then the release is
   **manifest-and-weights only**.

## Synthetic content introduced by this project

`paper_results/gradcam/e3_*` uses **synthetic composites** built by pasting
mine crops onto background tiles (`src/gradcam/phase6_e3_paste.py`). These are
generated artifacts, not source data, and are not redistributed as images — only
the aggregate detection rates are reported.

No physical replicas were photographed for this study. No field data was
collected.

## Split construction (summary; full detail in DECISIONS D010–D016)

1. Pool all sources; derive base keys by stripping the Roboflow export suffix.
2. Build groups from base key, capture session, and same-source pHash edges —
   1048 groups.
3. Assign whole groups to train/valid/test; never split a group.
4. Two-method leak filter on every val/test image: exact MD5, then base key,
   then same-source pHash Hamming ≤ 2 **and** 32×32 RGB MSE ≤ 900.
5. Manually review every Hamming ≤ 2 candidate on montages (146 pairs). Same-scene
   pairs separated at MSE ≤ 817; terrain-texture false positives at MSE ≥ 958.6.
6. Remove 30 true-leak eval images (19 valid, 11 test). Freeze the manifest.

Final: train 3907 / valid 181 / test 189; 0 exact cross-split duplicates;
0 base-key collisions; 0 true near-duplicate leaks (0/370).

## Known content defects (disclosed, not fixed)

- `May_lwir` and `Jan_lwir` are named for a thermal sensor but are colour RGB
  with visible marker pins. Marker-shortcut risk is disclosed (D016); a
  12-crop zoom check found no consistent marker cue adjacent to test mines.
- Two objectively wrong test boxes (`ds0_0801` on a tree trunk, `ds0_0883` on a
  fence) were removed from ground truth without retraining; results are reported
  with and without the fix (`paper_results/label_fix_report.json`, D023).
- Test contains only 5 negative images, so precision and FP/image are
  high-variance. Intervals, not point estimates, are the supported claim.
