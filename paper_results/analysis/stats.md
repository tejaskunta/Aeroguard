# Statistical analysis

## YOLOv8s size/seed CIs (image bootstrap, B=2000)

| run | conf | P | R | FP/img | image R [95% CI] |
|---|---:|---:|---:|---:|---|
| seed0 | 0.020 | 0.427 | 0.976 | 1.751 | 1.000 [0.980, 1.000] |
| seed1 | 0.010 | 0.356 | 0.980 | 2.376 | 0.995 [0.970, 0.999] |
| seed2 | 0.210 | 0.764 | 0.972 | 0.402 | 0.984 [0.953, 0.994] |

- seed0 source-cluster bootstrap (3 clusters ['Jan_lwir', 'ds0', 'ds1']): P [0.19518716577540107, 0.874251497005988], R [0.948051948051948, 1.0], FP/img [0.1390728476821192, 30.1]
- seed0 leave-one-source-out: Jan_lwir: P 0.853 R 0.989 FP 0.17; ds0: P 0.246 R 0.962 FP 8.16; ds1: P 0.405 R 0.973 FP 2.00

- seed1 source-cluster bootstrap (3 clusters ['Jan_lwir', 'ds0', 'ds1']): P [0.16898148148148148, 0.6743119266055045], R [0.948051948051948, 1.0], FP/img [0.47019867549668876, 35.9]
- seed1 leave-one-source-out: Jan_lwir: P 0.660 R 0.994 FP 0.50; ds0: P 0.211 R 0.962 FP 9.95; ds1: P 0.338 R 0.978 FP 2.67

- seed2 source-cluster bootstrap (3 clusters ['Jan_lwir', 'ds0', 'ds1']): P [0.5177304964539007, 0.9863945578231292], R [0.948051948051948, 1.0], FP/img [0.013245033112582781, 6.8]
- seed2 leave-one-source-out: Jan_lwir: P 0.956 R 0.983 FP 0.04; ds0: P 0.577 R 0.962 FP 1.95; ds1: P 0.757 R 0.969 FP 0.43

## Source / split shift

| split/source | images | boxes | boxes/img | empty rate |
|---|---:|---:|---:|---:|
| test/Jan_lwir | 10 | 77 | 7.70 | 0.00% |
| test/ds0 | 151 | 148 | 0.98 | 1.99% |
| test/ds1 | 28 | 28 | 1.00 | 7.14% |
| train/Jan_lwir | 100 | 740 | 7.40 | 0.00% |
| train/May_lwir | 776 | 7706 | 9.93 | 0.00% |
| train/ds0 | 2589 | 2643 | 1.02 | 0.77% |
| train/ds1 | 442 | 461 | 1.04 | 0.00% |
| valid/Jan_lwir | 42 | 287 | 6.83 | 0.00% |
| valid/ds0 | 138 | 131 | 0.95 | 5.07% |
| valid/ds1 | 1 | 1 | 1.00 | 0.00% |

## Threshold rules (chosen on val, applied once to test)

### seed0

Arm A conf 0.02, arm B conf 0.125, arm C proposals 0.05 with protection P=0.4.

| rule | stage | param | val P | val R | test P | test R | test FP/img |
|---|---|---:|---:|---:|---:|---:|---:|
| recall-safe (primary) | seed0 arm C (two-stage) | 0.270 | 0.809 | 0.959 | 0.894 | 0.964 | 0.153 |
| no recall protection | seed0 arm C (two-stage) | 0.020 | 0.583 | 0.959 | 0.654 | 0.964 | 0.683 |
| max F2 | seed0 arm C (two-stage) | 0.860 | 0.894 | 0.950 | 0.945 | 0.957 | 0.074 |
| conf-only above P (classifier off) | seed0 arm C (two-stage) | 1.000 | 0.902 | 0.945 | 0.953 | 0.957 | 0.063 |
| recall-safe | seed0 arm A (YOLO only) | 0.045 | 0.480 | 0.967 | 0.576 | 0.976 | 0.963 |
| matched recall to two-stage | seed0 arm A (YOLO only) | 0.195 | 0.771 | 0.955 | 0.825 | 0.968 | 0.275 |
| matched FP/img to two-stage | seed0 arm A (YOLO only) | 0.225 | 0.808 | 0.952 | 0.854 | 0.968 | 0.222 |
| max F2 | seed0 arm A (YOLO only) | 0.520 | 0.956 | 0.936 | 0.976 | 0.949 | 0.032 |

Param is the classifier threshold for arm C and the YOLO confidence for arm A; both are chosen on val and applied unchanged to test. Arm A is evaluated from the same records, which start at conf 0.02, so it cannot go below seed-0 arm A.
### seed1

Arm A conf 0.01, arm B conf 0.06, arm C proposals 0.05 with protection P=0.4.

| rule | stage | param | val P | val R | test P | test R | test FP/img |
|---|---|---:|---:|---:|---:|---:|---:|
| recall-safe (primary) | seed1 arm C (two-stage) | 0.230 | 0.785 | 0.959 | 0.825 | 0.968 | 0.275 |
| no recall protection | seed1 arm C (two-stage) | 0.030 | 0.622 | 0.959 | 0.725 | 0.957 | 0.487 |
| max F2 | seed1 arm C (two-stage) | 0.620 | 0.895 | 0.952 | 0.904 | 0.964 | 0.138 |
| conf-only above P (classifier off) | seed1 arm C (two-stage) | 1.000 | 0.927 | 0.936 | 0.921 | 0.964 | 0.111 |
| recall-safe | seed1 arm A (YOLO only) | 0.055 | 0.567 | 0.962 | 0.662 | 0.976 | 0.667 |
| matched recall to two-stage | seed1 arm A (YOLO only) | 0.075 | 0.628 | 0.955 | 0.704 | 0.976 | 0.550 |
| matched FP/img to two-stage | seed1 arm A (YOLO only) | 0.175 | 0.789 | 0.947 | 0.821 | 0.976 | 0.286 |
| max F2 | seed1 arm A (YOLO only) | 0.455 | 0.940 | 0.936 | 0.927 | 0.957 | 0.101 |

Param is the classifier threshold for arm C and the YOLO confidence for arm A; both are chosen on val and applied unchanged to test. Arm A is evaluated from the same records, which start at conf 0.02, so it cannot go below seed-0 arm A.
### seed2

Arm A conf 0.21, arm B conf 0.21, arm C proposals 0.05 with protection P=0.4.

| rule | stage | param | val P | val R | test P | test R | test FP/img |
|---|---|---:|---:|---:|---:|---:|---:|
| recall-safe (primary) | seed2 arm C (two-stage) | 0.220 | 0.537 | 0.969 | 0.724 | 0.972 | 0.497 |
| no recall protection | seed2 arm C (two-stage) | 0.030 | 0.324 | 0.969 | 0.515 | 0.968 | 1.222 |
| max F2 | seed2 arm C (two-stage) | 0.980 | 0.799 | 0.947 | 0.874 | 0.960 | 0.185 |
| conf-only above P (classifier off) | seed2 arm C (two-stage) | 1.000 | 0.799 | 0.947 | 0.877 | 0.960 | 0.180 |
| recall-safe | seed2 arm A (YOLO only) | 0.210 | 0.617 | 0.971 | 0.764 | 0.972 | 0.402 |
| matched recall to two-stage | seed2 arm A (YOLO only) | 0.280 | 0.704 | 0.964 | 0.807 | 0.972 | 0.312 |
| matched FP/img to two-stage | seed2 arm A (YOLO only) | 0.160 | 0.543 | 0.971 | 0.719 | 0.972 | 0.508 |
| max F2 | seed2 arm A (YOLO only) | 0.540 | 0.908 | 0.940 | 0.927 | 0.960 | 0.101 |

Param is the classifier threshold for arm C and the YOLO confidence for arm A; both are chosen on val and applied unchanged to test. Arm A is evaluated from the same records, which start at conf 0.02, so it cannot go below seed-0 arm A.
## Stage-2 ablations (arm C, recall-safe rule chosen on val)

| variant | pad | crop | hardneg | cls thr | P | R | FP/img | dP vs base [95% CI] |
|---|---:|---:|---|---:|---:|---:|---:|---|
| crop160 | 0.2 | 160 | True | 0.43 | 0.911 | 0.972 | 0.127 | +0.017 [+0.000, +0.036] |
| crop320 | 0.2 | 320 | True | 0.08 | 0.868 | 0.964 | 0.196 | -0.025 [-0.049, -0.006] |
| nohardneg | 0.2 | 224 | False | 0.35 | 0.837 | 0.972 | 0.254 | -0.056 [-0.084, -0.028] |
| pad0 | 0.0 | 224 | True | 0.16 | 0.866 | 0.968 | 0.201 | -0.027 [-0.043, -0.011] |
| pad04 | 0.4 | 224 | True | 0.25 | 0.868 | 0.964 | 0.196 | -0.025 [-0.042, -0.011] |
| baseline_pad02_224_hardneg | 0.2 | 224 | True | 0.27 | 0.894 | 0.964 | 0.153 | reference |

dP is a paired image bootstrap of (variant - baseline) computed on the same test images; an interval spanning zero means the ablation is indistinguishable from the baseline. Recall and FP/img intervals are in stats.json.
