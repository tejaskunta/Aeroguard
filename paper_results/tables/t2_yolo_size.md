### T2. YOLO size sweep on frozen split v2 (PC GPU)

| model | params | GFLOPs | weights MB | latency ms (med) | val mAP50 | test mAP50 | test R@OP | test P@OP | FP/img@OP | OP conf (val) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolov8s | 11,125,971 | 28.6 | 21.5 | 6.5 | 0.9546 ± 0.0064 | 0.9763 ± 0.0033 | 0.9771 ± 0.0076 | 0.5790 ± 0.1811 | 1.165 ± 0.863 | 0.020/0.010/0.210/0.045/0.125 |
| yolov8n | 3,005,843 | 8.2 | 6.0 | 6.9 | 0.9338 ± 0.0017 | 0.9787 ± 0.0025 | 0.9855 ± 0.0023 | 0.1847 ± 0.0947 | 7.266 ± 4.421 | 0.005/0.010/0.000 |

_yolov8s is trained on v2 with seeds 0/1/2/3/4 and yolov8n with seeds 0/1/2 (mean ± std over seeds); both rows carry intervals. yolov8m was trained on the older leaky v1 split only and is therefore not comparable here (RUN_LOG touch #18; the leaky-vs-clean comparison is reported separately)._
_Latency: PC GPU FP32, imgsz 640, 50 reps after 10 warm-up reps (paper_results/yolo/*/size_latency.json). GFLOPs are for the single-class head (nc=1) at 640; Ultralytics' published nc=80 figures for the same depths are n 8.7 / s 28.6._
_OP = recall-first operating point, confidence chosen per seed on val and then applied unchanged to test (conf listed per seed, so a single number means all seeds picked it)._
