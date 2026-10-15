### T3. Image-level comparison (test)

| system (test, image level) | accuracy | balanced acc | precision | recall | kappa |
|---|---|---|---|---|---|
| MNv3 A (thr 0.5) | 0.974 | 0.500 | 0.974 | 1.000 | 0.000 |
| MNv3 A (recall-first thr) | 0.968 | 0.497 | 0.973 | 0.995 | -0.009 |
| MNv3 A+B (thr 0.5) | 0.974 ± 0.000 | 0.500 ± 0.000 | 0.974 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| MNv3 A+B (recall-first thr) | 0.968 ± 0.005 | 0.562 ± 0.056 | 0.977 ± 0.003 | 0.991 ± 0.006 | 0.156 ± 0.148 |
| YOLO deploy (conf from val) | 0.984 ± 0.009 | 0.830 ± 0.115 | 0.991 ± 0.006 | 0.993 ± 0.008 | 0.677 ± 0.199 |
| YOLO op95 (conf from val) | 0.986 ± 0.003 | 0.993 ± 0.002 | 1.000 ± 0.000 | 0.986 ± 0.003 | 0.784 ± 0.038 |
| Two-stage (arm C, 3 seeds) | 0.991 ± 0.003 | 0.995 ± 0.002 | 1.000 ± 0.000 | 0.991 ± 0.003 | 0.854 ± 0.045 |

_MNv3 A+B = mean ± std over seeds 0/1/2; MNv3 A is seed 0; YOLO deploy/op95 = mean ± std over seeds 0/1/2 (each seed's OP conf chosen on val)._
_MNv3 recall-first thresholds chosen on val only._
