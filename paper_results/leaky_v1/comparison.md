# Leaky-v1 vs clean-v2 (yolov8s)

Leaky checkpoint `last` + OP conf 0.065 chosen on v1 val. Clean-v2 seed 0 OP conf 0.02 chosen on v2 val.
**Contamination:** 140 of 189 v2 test images (74.1%) share a v2 group with an image the v1 model trained on; only the remaining 49 images are a clean common test.

| model | eval set | mAP50 | P | R | FP/img |
|---|---|---:|---:|---:|---:|
| leaky v1 | v1 test (own split) | 0.9697 | 0.834 | 0.975 | n/a (0 neg) |
| leaky v1 | v2 test, full (contaminated) | 0.9940 | 0.857 | 0.996 | 0.222 |
| clean v2 seed 0 | v2 test, full | 0.9741 | 0.427 | 0.976 | 1.751 |

## Contamination-free subset

| model | mAP50 | mAP50-95 | P | R | FP/img |
|---|---:|---:|---:|---:|---:|
| leaky v1 | 0.9950 | 0.7250 | 0.863 | 1.000 | 0.204 |
| clean v2 seed 0 | 0.9947 | 0.7010 | 0.354 | 1.000 | 2.347 |

## Reading

A pre-re-split model cannot be fairly scored on the full v2 test set: three quarters of its test images sit in groups it trained on. On the contamination-free subset the two models are directly comparable; the subset is small, so the estimate is low-power. The old split's real defects are structural: leaked val used for model selection, a test split with zero negatives (false positives unmeasurable), and no clean way to evaluate any pre-re-split checkpoint.
