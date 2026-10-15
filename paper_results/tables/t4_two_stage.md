### T4. Two-stage pipeline (arms A, B, C)

| arm | conf/thr | precision | recall | F2 | FP/img | img recall |
|---|---|---|---|---|---|---|
| A (YOLO deploy) | 0.02 | 0.427 ± 0.000 | 0.976 ± 0.000 | 0.777 ± 0.000 | 1.751 ± 0.000 | 1.000 ± 0.000 |
| B (YOLO matched recall) | 0.125 | 0.762 ± 0.000 | 0.972 ± 0.000 | 0.921 ± 0.000 | 0.407 ± 0.000 | 0.995 ± 0.000 |
| C (two-stage) | yolo 0.05, P=0.4, t=0.27 | 0.879 ± 0.018 | 0.964 ± 0.000 | 0.946 ± 0.004 | 0.178 ± 0.030 | 0.991 ± 0.003 |

_Paired bootstrap (seed 0): C-B precision +0.128 CI [0.065, 0.186]; FP/img -0.253 CI [-0.460, -0.095]; recall -0.0079 CI [-0.0195, 0.0000]._
_Cost (plan §5.4, seed 0): 16.0 ms/image end-to-end, 3.06 crops/image (p95 3.6), PC GPU FP32._
_Means ± std over seeds 0/1/2 for arm C; A/B are seed-independent except threshold choice (B picked per seed on val)._
