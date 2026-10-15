### T2. YOLO size sweep (seeds 0/1/2, PC GPU)

| model | params | GFLOPs | weights MB | latency ms (med) | val mAP50 | test mAP50 | test R@OP | test P@OP | FP/img@OP | OP conf (val) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolov8s | 11,125,971 | 28.6 | 21.5 | 6.5 | 0.9544 ± 0.0027 | 0.9768 ± 0.0034 | 0.9763 ± 0.0040 | 0.5157 ± 0.2180 | 1.510 ± 1.009 | 0.02/0.01/0.21 |

_n/m runs out of scope under the approved reduced budget (D-plan scope); only yolov8s was trained on v2 (3 seeds). Metrics are mean ± std over seeds 0/1/2. Latency: PC GPU FP32, imgsz 640 (size_latency.json, seed 0). OP = recall-first conf chosen per seed on val (conf listed per seed)._
