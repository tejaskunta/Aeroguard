### T4. Two-stage pipeline (arms A, B, C)

| arm | conf/thr (seed 0 / 1 / 2) | precision | recall | F2 | FP/img | img recall |
|---|---|---|---|---|---|---|
| A (YOLO deploy) | 0.02 / 0.01 / 0.21 | 0.516 ± 0.218 | 0.976 ± 0.004 | 0.808 ± 0.102 | 1.510 ± 1.009 | 0.993 ± 0.008 |
| B (YOLO matched recall) | 0.125 / 0.06 / 0.21 | 0.734 ± 0.050 | 0.974 ± 0.002 | 0.913 ± 0.014 | 0.478 ± 0.127 | 0.989 ± 0.005 |
| C (two-stage) | yolo 0.05, P=0.4, t=0.27 / 0.23 / 0.22 | 0.814 ± 0.086 | 0.968 ± 0.004 | 0.932 ± 0.020 | 0.309 ± 0.174 | 0.991 ± 0.003 |

_Paired bootstrap C-B, one row per detector seed (5000 resamples over test images): detector seed 0: C-B precision +0.128 [+0.065, +0.186], FP/img -0.2529 [-0.4603, -0.0952], recall -0.0079 [-0.0195, +0.0000]; detector seed 1: C-B precision +0.144 [+0.096, +0.181], FP/img -0.3466 [-0.5979, -0.1587], recall -0.0079 [-0.0195, +0.0000]; detector seed 2: C-B precision -0.037 [-0.077, +0.013], FP/img +0.0927 [-0.0212, +0.2275], recall +0.0001 [-0.0199, +0.0166]. The precision/FP gain is significant at detector seeds 0 and 1 only: at seed 2 the paired interval for precision spans zero and the point estimate is negative, so the two-stage advantage is not established across seeds._
_Cost (plan §5.4, detector seed 0): 16.0 ms/image end-to-end, 3.06 crops/image (p95 3.6), PC GPU FP32._
_Means ± std over the three detector seeds 0/1/2 for all arms. Stage 1 for every row is the YOLOv8s checkpoint of that seed; the stage-2 classifier weights are shared. Arms A and B are YOLO-only, so they differ between rows only through the detector checkpoint and the per-seed val-chosen threshold._
_Arm A threshold = recall-first operator point chosen on val; arm B threshold chosen on val to match arm A's val recall; arm C = YOLO dets at conf>=yolo_conf routed through the MobileNetV3-Large crop classifier with protected cutoff P and classifier threshold t._
_Caveat at detector seed 2: the val-selected matched-recall threshold coincides with the deploy threshold (both 0.21), so arms A and B are the same operating point and the C-B comparison there is against an undegraded YOLO baseline. The two-stage pipeline loses 3.6 points of precision against that baseline, i.e. the gain in T4 comes from seeds whose val-selected YOLO threshold transferred poorly to test, not from seeds where it transferred well. This is the central threat to the headline claim and is discussed in §8._
