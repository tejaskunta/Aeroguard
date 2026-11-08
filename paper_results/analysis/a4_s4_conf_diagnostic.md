# A4 — Seed 4's two-stage arm loses to its own baseline. Why?

**Status:** post-hoc diagnostic, completed. Primary numbers unchanged.

## The anomaly

Seed 4 is the one seed where the two-stage arm (C) is worse than its matched-recall
YOLO-only baseline (B) on the test set:

| | P | R | FP/img |
|---|---:|---:|---:|
| B — YOLO only @ conf 0.125 | 0.7292 | 0.9684 | 0.481 |
| C — two-stage, primary config | 0.6768 | 0.9684 | 0.619 |
| C − B | **−0.052** | 0.000 | **+0.138** |

Seeds 0–3 all show C ≥ B on precision and C < B on FP/image.

## What the primary configuration actually does on seed 4

The primary protocol fixes stage-1 proposals at conf **0.05** and then sweeps the
classifier threshold on validation. On seed 4 the val sweep selected
**τ = 0.03 with protect-conf 0.8** — the lowest τ on the grid and the highest
protection band. Both choices make the classifier nearly inert:

- **τ = 0.03 is at the floor of the 0–1 sweep grid (step 0.01).** Almost no proposal
  has a classifier score that low, so almost nothing is vetoed.
- **protect-conf 0.8** means every proposal with stage-1 confidence ≥ 0.8 skips the
  classifier entirely.

Measured on the saved raw predictions (test set, 444 proposals at conf ≥ 0.05):

| config | proposal pool | eligible for classifier | vetoed | veto rate |
|---|---:|---:|---:|---:|
| primary (0.05 / τ 0.03 / P 0.8) | 444 | 289 | 82 | 18.5% |
| post-hoc (0.125 / τ 0.13 / P 0.4) | 336 | 71 | 48 | 14.3% |

The primary config vetoes 82 proposals but **adds 108** relative to arm B, because
stage-1 at conf 0.05 admits 108 more proposals than conf 0.125 and most of those
survive. The two-stage arm is being asked to clean a pool that is 32% larger and
much dirtier, using a veto rule that was tuned to remove as little as possible.

## Why validation picked that config

It was not a mistake; validation genuinely preferred it. Comparing, on **validation
only** (171 positive images + 7 negatives), the chosen two-stage point against
YOLO-only at the arm-A conf:

| seed | stage-1 conf | chosen P / τ | val FP/img (C) | val FP/img (arm A) | C better on val? |
|---:|---:|---|---:|---:|---|
| 0 | 0.05 | P=0.4 / 0.27 | 0.525 | 4.735 | yes |
| 1 | 0.05 | P=0.4 / 0.29 | 0.558 | 4.735 | yes |
| 2 | 0.05 | P=0.4 / 0.38 | 0.530 | 4.735 | yes |
| 3 | 0.05 | P=0.4 / 0.13 | 0.851 | 2.326 | yes |
| 4 | 0.05 | **P=0.8 / 0.03** | **1.448** | **0.956** | **no** |

Seed 4 is the only seed where the selected two-stage point is *already* worse than
the YOLO-only reference **on validation**, before test is touched. The honest
reading is that the val-selected configuration for seed 4 was a bad configuration,
and the test result faithfully reports that. This is a failure of the selection
rule, not evidence that the two-stage idea fails on seed 4 — but it is a real
failure and it belongs in the paper.

## The post-hoc re-run (does *not* replace the primary)

Recomputing from the same saved raw predictions (`raw_preds_{val,test}.json`, built
at conf 0.02, so every proposal at conf ≥ 0.125 is present) with stage-1 conf set
to **0.125**, matching arm B's operating point, and with τ re-chosen on validation
under the identical rule (largest τ within max-recall-drop 0.005 of stage-1 recall,
then lowest val FP/img across the protect grid):

Validation sweep (stage-1 conf 0.125, val R 0.9737, val FP/img 0.956):

| P | τ | val R | val P | val FP/img |
|---:|---:|---:|---:|---:|
| 0.4 | 0.13 | 0.9690 | 0.8185 | 0.497 |
| 0.6 | 0.03 | 0.9714 | 0.7607 | 0.707 |
| 0.8 | 0.03 | 0.9690 | 0.7632 | 0.696 |
| none | 0.03 | 0.9690 | 0.7632 | 0.696 |

Chosen: **P = 0.4, τ = 0.13**.

Test result:

| | P | R | FP/img |
|---|---:|---:|---:|
| B — YOLO only @ 0.125 | 0.7292 | 0.9684 | 0.481 |
| C post-hoc @ 0.125, τ 0.13, P 0.4 | **0.8438** | 0.9605 | **0.238** |
| C − B (paired image bootstrap) | **+0.113** [+0.083, +0.146] | −0.008 [−0.019, 0.000] | **−0.243** [−0.365, −0.153] |

The precision and FP/image intervals exclude zero. **The seed-4 loss disappears
once stage 1 proposes at the baseline's own confidence.** The cost is 0.8 points of
recall, which is the classifier doing its job.

## Per-source detail (seed 4, test set, 189 images)

| source | imgs | arm B TP/FP | B FP/img | primary C TP/FP | C FP/img | post-hoc C TP/FP | C FP/img |
|---|---:|---|---:|---|---:|---|---:|
| Jan_lwir | 10 | 71 / 44 | 4.400 | 71 / 71 | 7.100 | 69 / 22 | 2.200 |
| ds0 | 151 | 146 / 34 | 0.225 | 146 / 30 | 0.199 | 146 / 13 | 0.086 |
| ds1 | 28 | 28 / 13 | 0.464 | 28 / 16 | 0.571 | 28 / 10 | 0.357 |

The primary config's damage is concentrated almost entirely in `Jan_lwir`: 44 → 71
false positives, i.e. the two-stage arm made the worst source *worse* rather than
better. Under the post-hoc config `Jan_lwir` drops to 22, below arm B, for 2 lost
true positives on that source. Every source improves.

## What this means for the paper

1. The paper must **report the seed-4 anomaly as observed**, with the primary
   configuration, because that is what the protocol produced.
2. The paper must **stop claiming the two-stage arm is consistently better than a
   matched-recall baseline**. It is better for 4 of 5 seeds, and the fifth is a
   selection failure that a small protocol change removes. The correct claim is the
   weaker one: *the two-stage gain is conditional on the stage-1 proposal
   confidence, and a val-selected configuration can silently disable the classifier.*
3. The post-hoc re-run is reported as a **diagnosis of the mechanism**, explicitly
   labelled post-hoc, never as the headline number. Its value is that it isolates
   the confound; its cost is that its config was chosen knowing the primary result.
4. The deeper methodological point is worth keeping: the val-selection rule
   ("lowest val FP/img subject to a recall cap") picked a near-inert classifier on
   seed 4. A rule that also required the classifier to beat the stage-1-only point
   on validation would have rejected it. That is a genuine finding about
   two-stage pipelines with weak stage-2 models.

## Artifacts

- `src/analysis/a4_s4_conf_diagnostic.py` — the diagnostic (CPU-only, reads saved JSON).
- `paper_results/analysis/a4_s4_stage1_conf0125_diagnostic.json` — full sweep + CIs.
- Per-source numbers above recomputed inline from
  `paper_results/stage2/s2_ratio1_s4/raw_preds_test.json` +
  `paper_results/split_v2/split_manifest.csv`.

## Protocol compliance

- No threshold was chosen on test. τ was chosen on validation in both the primary
  run and this diagnostic.
- No test image was re-read; all numbers come from already-saved raw predictions
  whose minimum conf (0.0200) is below every conf used here.
- No data or label file was modified.