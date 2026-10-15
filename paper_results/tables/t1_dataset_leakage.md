### T1. Dataset statistics and leakage audit

| split | images | instances | negatives | sources |
|---|---|---|---|---|
| train | 3907 | 11550 | 20 | Jan_lwir:100, May_lwir:776, ds0:2589, ds1:442 |
| valid | 181 | 419 | 7 | Jan_lwir:42, ds0:138, ds1:1 |
| test | 189 | 253 | 5 | Jan_lwir:10, ds0:151, ds1:28 |

_Exact cross-split duplicates: 0._
_Near-dup candidates (pHash ham<=2 vs train): 99 — all residual MSE>=958.6 after the second-method filter (0 true leaks; 30 leaked eval images removed pre-freeze; D014/D015). Manifest sha256 in split_v2/sha256.txt._
_Base-key collisions across splits: {'train_valid': 0, 'train_test': 0, 'valid_test': 0}._
