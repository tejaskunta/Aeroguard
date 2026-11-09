# On-device measurement procedure

This document fixes the procedure for producing the one number the paper does not
report: on-device inference latency at FP16 on the Jetson Nano target.

It exists so that the measurement is a mechanical run rather than an open
question. Nothing in this document is a result. No device number is claimed
anywhere in the paper today, and the deployment project's README prose figures are
deliberately not reused, because prose is not a measurement and cannot be
re-derived from the repository that reports it.

## Why a procedure and not a number

Three separate things make a device latency number hard to trust when it is
reported loosely:

1. **The engine is not portable.** A TensorRT engine is compiled for a specific
   GPU architecture and TensorRT version. An engine built on a desktop GPU will
   not load on Maxwell, and an engine built for TensorRT 8.5 will not load on
   JetPack 4.6 (TensorRT 8.2). The engine must be built on the device, by the
   device, for the device.
2. **The artifact is not identified by its filename.** See
   `paper_results/edge/export_parity_host.json` and Table XVI in the paper: the
   deployment tree stores two files named `best` that are two different models.
   A latency number is meaningless unless it is bound to a SHA-256.
3. **The stage boundaries are invisible.** A single end-to-end FPS figure
   conflates capture, preprocessing, inference, decode and render. On a 4 GB
   Maxwell part these costs are of comparable magnitude, so a single number
   cannot be attributed or optimised.

## Prerequisites

- Jetson Nano P3450 (4 GB), JetPack 4.6 / L4T R32.7.6, CUDA 10.2,
  TensorRT 8.2, Python 3.6.
- The target artifact, copied to the device. Use a released seed export:
  `models/yolo_s_seed0/best.onnx`,
  SHA-256 `ceb768f1004e43b42c10e127ed4695905a9e1d60893374207e4873411a0f2bc0`.
- Verify the copy before building:

  ```bash
  sha256sum ~/aeroshield/weights/s0.onnx
  # must equal the digest above; if it does not, stop
  ```

## Build

Build on the device. Never cross-compile.

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=~/aeroshield/weights/s0.onnx \
    --saveEngine=~/aeroshield/weights/s0_fp16.engine \
    --fp16 \
    --workspace=2048 \
    --verbose 2>&1 | tee ~/aeroshield/weights/s0_fp16.build.log
```

Record the build log. It is the evidence that the engine corresponds to the ONNX
digest above, and it reports the layer count and precision of the final plan.

A build takes roughly 5–20 minutes on this part. `--workspace=2048` is set in MB;
reduce it if the build fails for memory.

## Measure

Run the existing per-stage harness, which reports the four stages separately
rather than collapsing them:

```bash
cd ~/aeroshield
python3 jetson/live_detect.py \
    --engine weights/s0_fp16.engine \
    --max-frames 400 \
    --quiet 2>&1 | tee weights/s0_fp16.run.log
```

## What a valid run must report

A run is only usable if the committed summary contains all of the following.
Any missing field makes the number non-comparable and it must not be quoted.

| Field | Why it is required |
| --- | --- |
| `artifact_sha256` | Binds the number to a model, not a filename |
| `engine_build_log` | Proves on-device build and the precision plan |
| `tensorrt_version` | Engines do not transfer across versions |
| `precision` | FP16 and FP32 numbers are not comparable |
| `input_shape` | The export is fixed at 640×640; resizing changes the number |
| `frames` | Sample size; a 20-frame run is not a characterisation |
| `warmup_frames` | Discarded runs; the first inferences are not steady state |
| `stage_*_ms_median` | Per-stage medians: capture, preprocess, infer, decode, render |
| `stage_*_ms_p95` | Tail behaviour, which is what an operator feels |
| `power_mode` | `nvpmodel` and clock settings change the number by a large factor |
| `thermal_state` | Whether the part was throttled during the run |

Write the result to `paper_results/edge/edge_s0_fp16.json`. That directory exists
for device measurements specifically; it currently contains only the host-side
export audit, which is labelled as such and must not be confused with a device
number.

## What may be claimed from such a run

A single device run of a single seed is a **demonstration**, not a
characterisation. It supports a claim of the form "this model of this size runs
on this hardware at this rate under these settings". It does not support:

- a distribution over seeds, because n = 1;
- a comparison against the PC GPU numbers in Table VI, because energy, thermal
  headroom and memory bandwidth all differ;
- a throughput figure for the system, because the capture path, not inference,
  may be the bottleneck;
- a claim about detection quality on the device, which is the export-parity and
  operating-point question and is separate from timing.

If a latency number is ever added to the paper it must be reported with its
`power_mode`, its precision and its sample size alongside it, and the host-side
CPU figure in `export_parity_host.json` must not be placed in the same table.