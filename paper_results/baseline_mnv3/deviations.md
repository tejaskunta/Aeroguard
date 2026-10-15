# MNv3 baseline — deviations from the base-paper recipe

Base paper = Heuschmid et al., IEEE Access 2025 (plan Appendix A). Implementation:
`phase4_mnv3_baseline.py` + shared path `mnv3_shared.py` (plan ground rule 10).

## Followed exactly

| Item | Plan / paper | This project |
|---|---|---|
| Task | image-level binary (1 = image has >=1 GT box) | same (`label = 1 if any box`) |
| Backbone | MobileNetV3-Large, ImageNet weights, no top | `torchvision` MobileNet_V3_Large, IMAGENET1K_V2 |
| Head | GAP -> Dense 128 + ReLU -> Dropout 0.3 -> Dense 1 + sigmoid | identical (`mnv3_shared.MineClassifier`) |
| Stage A | base frozen, BCE, Adam lr 1e-4, 20 epochs x 25 steps x batch 32 | identical; class weights from train [97.675, 0.5026] |
| Augmentation | rotation ±10°, zoom ≤10%, brightness 0.7–1.0, h+v flips | identical (`_augment`) |
| Input | whole image resized to 224×224 | `preprocess_uint8`: RGB, /255, ImageNet mean/std |

## Deliberate deviations (all plan-mandated or logged)

1. **Stage B** (plan §Phase 5.2, not in the base paper): unfreeze the deepest
   inverted-residual blocks — `features[13:]`, ~2.3 M params — lr 2e-5, 10 more
   epochs. The base paper has no stage B; it is the plan's ablation/primary
   baseline lever. A-rows (frozen) are reported alongside.
2. **ImageNet mean/std instead of raw 1/255** (plan §Phase 4.6): one
   normalisation shared with the stage-2 classifier.
3. **Resolution and batch ablations**: A_448, A_640, A_224_b64 rows are the
   plan ablations (not extra tuning); thresholds always chosen on val.
4. **Finding, not a deviation (D020)**: Stage A is degenerate on negatives at
   every resolution/batch tried (test balanced acc ~0.48–0.50, kappa ~0) because
   the 20-image negative class barely exists in train (3887:20). Stage B is the
   only variant that separates negatives; AB_224 is the primary row
   (seeds 0/1/2 mean±std in T3).
5. Seeds: 0/1/2 as required by plan rule 4 (per-seed `metrics.json` here).
