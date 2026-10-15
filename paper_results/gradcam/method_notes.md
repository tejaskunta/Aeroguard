# Grad-CAM method notes (plan §6.0)

Implementation: `phase6_gradcam.py` (classes `YoloCAM`, `ClsCAM`, helpers
`energy_stats`, `match_map`). Note: the plan references a file
`jetson/gradcam.py`; it does not exist in this repo (logged in RUN_LOG.md),
so the method below was implemented fresh from §6.0.

## YOLOv8 (`YoloCAM`)

- **Target layer**: the inputs of the `Detect` head, i.e. the neck feature maps
  feeding detection (strides 8/16/32), captured with a forward hook on
  `model.model[-1]`.
- **Score**: top class score of the raw (pre-NMS) output,
  `raw[:, 4:].amax()`. Single class, raw shape `(1, 5, N)`.
- **Stride deviation (documented)**: §6.0 hints at stride 16 or 32, but the
  top pre-NMS anchor usually sits on the stride-8 head, where the stride-32
  gradient is exactly zero. `cam()` therefore uses the largest-stride input
  whose gradient is non-zero (checks 32 → 16 → 8, falls back to input 0).
- **Grad flow**: Ultralytics freezes gradients on load; `YoloCAM.__init__`
  calls `requires_grad_(True)`. After any `model.predict()` call, Ultralytics
  has moved the model to CUDA inside `torch.inference_mode()`, which (a)
  strips `requires_grad` from parameters and (b) turns plain tensor
  attributes (`anchors`, `strides`) into inference tensors that break
  backward. `cam()` re-arms `requires_grad` and `_defuse()` clones any
  inference tensor before each backward pass.
- Heatmap = ReLU of the gradient-weighted activation sum, bilinearly
  upsampled to image size, normalised per image (max 1).

## MobileNetV3 (`ClsCAM`)

- **Target layer**: last conv block, `features[-1]` (hook on its output).
- **Score**: the single pre-sigmoid logit.
- Used for the stage-2 crop classifier and the image-level baseline
  (both share `mnv3_shared.build_mnv3_large`).

## Wording rule

Grad-CAM shows which regions influenced the prediction; it never proves the
prediction is correct. Every paper sentence follows this.

## Derived metrics

- `energy_share`: CAM energy inside the box (or union of boxes) / total energy.
- `enrichment`: energy_share / area_share (1.0 = uniform attention).
- `peak_inside`: whether argmax of the heatmap falls inside the box.
