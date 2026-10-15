### Per-source test results (source = dataset origin, from split_manifest.csv)

**A_yolo_only_deploy**

| source | images | TP | FP | FN | precision | recall | FP/img |
|---|---|---|---|---|---|---|---|
| Jan_lwir | 10 | 73 | 301 | 4 | 0.195 | 0.948 | 30.100 |
| ds0 | 151 | 146 | 21 | 2 | 0.874 | 0.986 | 0.139 |
| ds1 | 28 | 28 | 9 | 0 | 0.757 | 1.000 | 0.321 |
| ALL | 189 | 247 | 331 | 6 | 0.427 | 0.976 | 1.751 |

**C_two_stage**

| source | images | TP | FP | FN | precision | recall | FP/img |
|---|---|---|---|---|---|---|---|
| Jan_lwir | 10 | 70 | 21 | 7 | 0.769 | 0.909 | 2.100 |
| ds0 | 151 | 146 | 4 | 2 | 0.973 | 0.986 | 0.026 |
| ds1 | 28 | 28 | 4 | 0 | 0.875 | 1.000 | 0.143 |
| ALL | 189 | 244 | 29 | 9 | 0.894 | 0.964 | 0.153 |
