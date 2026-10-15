# Models — SafeMine / AeroGuard landmine detection (v2 research)

Trained on the frozen `landmine_combined_v2` split (train 3907 / valid 181 / test 189;
0 true leaks; manifest sha256 in `paper_results/split_v2/sha256.txt`).

## Layout

| Path | What |
|---|---|
| `yolo_s_seed{0,1,2}/` | YOLOv8s 75-epoch detector, seeds 0/1/2. `best.pt` / `last.pt` / `best.onnx` + training `args.yaml` + `results.csv` |
| `stage2_s{0,1,2}/` | Stage-2 MobileNetV3-small crop classifier (two-stage pipeline arm C). Input = YOLO box + 20% context, 224×224 crop; sigmoid score |
| `original_v1/` | Original v1 yolov8m run (pre-v2-split, kept for reference only — **do not use for paper numbers**) |
| `SHA256SUMS.txt` | sha256 of every weight/config file above |

## Frozen test metrics (from `paper_results/yolo/s_seed*/test.json`)

| Seed | val mAP50 | test mAP50 | OP conf (from val) | box P | box R | FP/img |
|---|---|---|---|---|---|---|
| 0 | 0.9535 | 0.9741 | 0.02 | 0.427 | 0.976 | 1.751 |
| 1 | 0.9523 | 0.9807 | 0.01 | 0.356 | 0.980 | 2.376 |
| 2 | 0.9574 | 0.9756 | 0.21 | 0.764 | 0.972 | 0.402 |

Mean ± std test mAP50: **0.9768 ± 0.0034**.

Two-stage arm C (seed 0): precision 0.894, recall 0.964, 0.153 FP/img (vs arm B 0.407);
paired bootstrap C−B precision +0.128 CI [0.065, 0.186], FP/img −0.253 CI [−0.460, −0.095].
Cost: 16 ms/image, ~3 crops/image (PC GPU FP32).

## Load (Ultralytics)

```python
from ultralytics import YOLO
model = YOLO("models/yolo_s_seed0/best.pt")
results = model.predict(source="path/to/image.jpg", conf=0.02)  # frozen OP conf
```

Stage-2 classifier: standard torchvision MobileNetV3-Small, 1 class, sigmoid score;
threshold τ = 0.27 / 0.29 / 0.38 for seeds 0/1/2 (chosen on val only).

## Not included (by design)

- Dataset images/labels (too large for git; see original Roboflow sources + v2 split manifest).
- Epoch checkpoints every 5 epochs (only `best.pt` / `last.pt` shipped).
- Jetson/TensorRT engines (edge phases out of approved scope).

## Provenance

Every training/eval command is logged in `RUN_LOG.md` (test touches #1–#14).
Environment pin: ultralytics 8.4.117, torch 2.6.0+cu124, see `ENVIRONMENT.txt`.
Paper tables/figures regenerate via `make_tables.py`; PDF reports via `make_report_pdfs.py`.
