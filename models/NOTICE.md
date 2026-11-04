# Third-party notices for the released weights

This directory ships trained model weights. They are **not** original works in the
copyright sense: two of the three families are derivative of third-party
architectures and inherit their licences. The repository's own `LICENSE`
(AGPL-3.0) covers the code, the split manifest, the tables and the figures — it
does not override any upstream licence.

| Artifact | Architecture | Upstream | Licence inherited |
|---|---|---|---|
| `yolo_s_seed{0,1,2}/best.pt`, `last.pt`, `best.onnx` | Ultralytics YOLOv8s, `nc=1` | `ultralytics` 8.4.117 | **AGPL-3.0** |
| `original_v1/best.pt`, `last.pt` | Ultralytics YOLOv8m, `nc=1` | `ultralytics` (v1 run) | **AGPL-3.0** |
| `stage2_s{0,1,2}/best.pt` | torchvision MobileNetV3-Large, fine-tuned | `torchvision` 0.21.0 | BSD-3-Clause (upstream code/weights); the fine-tuned checkpoint is distributed here under AGPL-3.0 for consistency |
| `SHA256SUMS.txt` | — | — | repository licence |

## What this means for you

- **YOLO checkpoints are AGPL-3.0.** If you redistribute them, or a model derived
  from them, or run them as part of a network service, AGPL-3.0 §13 requires you
  to offer the corresponding source. `ultralytics` is the relevant upstream
  source; this repository is the source for the training and evaluation code.
- **ONNX exports** (`best.onnx`) are conversions of the same weights and carry
  the same terms as the `.pt` they came from.
- **No image data is released here.** See `../DATA_PROVENANCE.md`.

## Attribution

Ultralytics YOLOv8: Jocher, Chaurasia & Qiu, *Ultralytics YOLO* (2023),
https://github.com/ultralytics/ultralytics — AGPL-3.0.

MobileNetV3: Howard et al., *Searching for MobileNetV3*, ICCV 2019;
implementation and pretrained weights from `torchvision` (BSD-3-Clause).
