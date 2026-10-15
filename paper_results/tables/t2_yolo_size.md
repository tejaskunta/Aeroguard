### T2. YOLO size sweep (seed 0, PC GPU)

| model | params | GFLOPs | weights MB | latency ms (med) | val mAP50 | test mAP50 | test R@OP | test P@OP | FP/img@OP |
|---|---|---|---|---|---|---|---|---|---|
| yolov8s | 11,125,971 | 28.6 | 21.5 | 6.5 | 0.9535 | 0.9741 | 0.9763 | 0.4273 | 1.751 |

_n/m runs out of scope under the approved reduced budget (D-plan scope); only yolov8s was trained on v2. Latency: PC GPU FP32, imgsz 640 (size_latency.json). OP = recall-first conf chosen on val._
