### T3. Image-level comparison (test)

| system (test, image level) | accuracy | balanced acc | precision | recall | kappa |
|---|---|---|---|---|---|
| MNv3 A (thr 0.5) | 0.974 | 0.500 | 0.974 | 1.000 | 0.000 |
| MNv3 A (recall-first thr) | 0.968 | 0.497 | 0.973 | 0.995 | -0.009 |
| MNv3 A+B (thr 0.5) | 0.974 ± 0.000 | 0.500 ± 0.000 | 0.974 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| MNv3 A+B (recall-first thr) | 0.968 ± 0.005 | 0.562 ± 0.056 | 0.977 ± 0.003 | 0.991 ± 0.006 | 0.156 ± 0.148 |
| YOLO deploy (conf from val) | 0.984 ± 0.007 | 0.836 ± 0.089 | 0.991 ± 0.005 | 0.992 ± 0.006 | 0.683 ± 0.159 |
| YOLO op95 (conf from val) | 0.986 ± 0.003 | 0.973 ± 0.044 | 0.999 ± 0.002 | 0.987 ± 0.003 | 0.780 ± 0.047 |
| Two-stage (arm C, 5 seeds) | 0.987 ± 0.006 | 0.916 ± 0.084 | 0.996 ± 0.005 | 0.991 ± 0.003 | 0.767 ± 0.120 |

_MNv3 A+B = mean ± std over seeds 0/1/2; MNv3 A is seed 0; YOLO deploy/op95 = mean ± std over seeds 0/1/2/3/4 (each seed's OP conf chosen on val)._
_MNv3 recall-first thresholds chosen on val only._
