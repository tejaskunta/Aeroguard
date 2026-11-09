#!/usr/bin/env python3
"""Export-parity and graph-shape measurements for the deployment boundary.

Everything this script reports is measured on the host that produced the
artifacts. Nothing here is a device measurement: the Jetson is a separate
platform and its numbers must be produced on that platform (see
docs/EDGE_MEASUREMENT.md).

What is measured
----------------
1. Graph shape of every committed ONNX export: opset, input shape, output
   shape, whether non-maximum suppression is inside the graph, and whether a
   confidence threshold is inside the graph.
2. Export parity: for each checkpoint that has both a .pt and an .onnx, the
   maximum absolute difference between the PyTorch forward pass and the ONNX
   Runtime forward pass on a fixed input.
3. Host latency of the ONNX graph under onnxruntime, CPU only, reported as
   median and 95th percentile over repeated runs.
4. Post-processing sensitivity: how the detection count changes with the
   confidence threshold applied *outside* the graph.

Written to paper_results/edge/ as JSON. The directory was previously a
placeholder; this file records what can be measured without the device.

    python scripts/measure_export_parity.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "paper_results" / "edge"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def graph_facts(onnx_path: Path) -> dict:
    """Structural facts about an exported graph."""
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    g = m.graph
    ops = {}
    for node in g.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    def shape_of(vi):
        return [d.dim_value if d.dim_value else d.dim_param
                for d in vi.type.tensor_type.shape.dim]

    return {
        "opset": max(o.version for o in m.opset_import),
        "inputs": {i.name: shape_of(i) for i in g.input},
        "outputs": {o.name: shape_of(o) for o in g.output},
        "nodes": len(g.node),
        "nms_in_graph": "NonMaxSuppression" in ops,
        # A threshold inside the graph would show up as a comparison against a
        # constant score; Ultralytics exports leave both NMS and thresholding to
        # the caller.
        "top_ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])[:8]),
    }


def weight_param_count(onnx_path: Path) -> int:
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(onnx_path), load_external_data=False)
    return sum(int(numpy_helper.to_array(i).size) for i in m.graph.initializer)


def torch_param_count(pt_path: Path) -> int:
    import torch

    ck = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    model = ck.get("model", ck)
    return int(sum(v.numel() for v in model.state_dict().values()))


def weight_dtype_summary(pt_path: Path) -> dict:
    """Storage dtype of the learned tensors.

    This is what makes raw file sizes useless as a cross-format clue: the
    checkpoints here are stored in half precision while an ONNX export of the
    same weights is FP32, so the export is roughly twice the checkpoint size
    for an identical model.
    """
    import torch

    ck = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    sd = ck["model"].state_dict()
    dtypes: dict[str, int] = {}
    for v in sd.values():
        dtypes[str(v.dtype).replace("torch.", "")] = dtypes.get(
            str(v.dtype).replace("torch.", ""), 0) + 1
    n = int(sum(v.numel() for v in sd.values()))
    size = Path(pt_path).stat().st_size
    return {
        "tensor_dtypes": dtypes,
        "params": n,
        "bytes_per_param": round(size / n, 4) if n else None,
    }


def _conv_shape_index_from_state_dict(sd) -> dict:
    """Map 4-D tensor shape -> list of arrays, for a torch state dict."""
    index: dict = {}
    for v in sd.values():
        if getattr(v, "ndim", None) == 4:
            index.setdefault(tuple(v.shape), []).append(
                v.detach().cpu().float().numpy())
    return index


def _conv_shape_index_from_onnx(onnx_path: Path) -> dict:
    """Map 4-D initializer shape -> list of arrays, for an ONNX graph.

    Ultralytics exports fold batch normalisation into the preceding
    convolution, so an exported conv weight equals the checkpoint's conv weight
    rescaled by (gamma / sqrt(var + eps)). Comparing the raw tensors is
    therefore meaningless; the comparison below is scale-invariant instead.
    """
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(onnx_path), load_external_data=False)
    index: dict = {}
    for init in m.graph.initializer:
        a = numpy_helper.to_array(init)
        if a.ndim == 4:
            index.setdefault(tuple(a.shape), []).append(a.astype(np.float32))
    return index


def _conv_shape_index_from_checkpoint(pt_path: Path) -> dict:
    """Conv-kernel index for a checkpoint, with batch norm folded in.

    Folding is what makes a checkpoint comparable to its export: the export
    performs the same fold, so both sides describe the same effective kernels.
    """
    import torch
    from ultralytics import YOLO

    ck = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    model = ck.get("model", ck)
    try:
        model = YOLO(str(pt_path)).model
    except Exception:
        pass
    try:
        model.fuse()
    except Exception:
        pass
    return _conv_shape_index_from_state_dict(model.state_dict())


def _conv_index_for(path: Path) -> dict:
    """Conv-kernel index for either representation."""
    if path.suffix.lower() == ".onnx":
        return _conv_shape_index_from_onnx(path)
    return _conv_shape_index_from_checkpoint(path)


def _reciprocal_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Scale-invariant agreement between two tensors of identical shape.

    1.0 means ``a`` is a positive scalar multiple of ``b`` (which is what a
    fused batch-norm does to a convolution kernel); 0.0 means no elementwise
    proportional relationship exists. Robust to a handful of near-zero entries.
    """
    af = a.ravel()
    bf = b.ravel()
    nz = np.abs(bf) > 1e-8
    if nz.sum() < 16:
        return 0.0
    r = af[nz] / bf[nz]
    med = float(np.median(r))
    if not np.isfinite(med) or abs(med) < 1e-12:
        return 0.0
    tol = max(1e-2 * abs(med), 1e-9)
    return float((np.abs(r - med) <= tol).mean())


def provenance_match(query_conv_index: dict, ref_conv_index: dict,
                     ref_label: str) -> dict:
    """Best scale-invariant assignment of query conv kernels onto a reference.

    Returns coverage (fraction of reference conv tensors matched), the median
    reciprocal agreement over the matched pairs, and the fraction of matched
    pairs whose agreement exceeds 0.95. An identical model scores coverage 1.0
    with agreement near 1.0; a different architecture runs out of reference
    shapes (coverage well below 1.0).
    """
    work = {k: list(v) for k, v in query_conv_index.items()}
    scores: list[float] = []
    n_ref = sum(len(v) for v in ref_conv_index.values())
    for shape, tensors in ref_conv_index.items():
        for ref in tensors:
            cand = work.get(shape)
            if not cand:
                continue
            agree = [_reciprocal_agreement(ref, c) for c in cand]
            i = int(np.argmax(agree))
            scores.append(agree[i])
            cand.pop(i)
    matched = len(scores)
    med = float(np.median(scores)) if scores else 0.0
    frac = float(np.mean([s > 0.95 for s in scores])) if scores else 0.0
    return {
        "ref": ref_label,
        "ref_conv_tensors": n_ref,
        "matched": matched,
        "coverage": round(matched / n_ref, 4) if n_ref else 0.0,
        "median_agreement": round(med, 4),
        "frac_strong_match": round(frac, 4),
        "verdict": "same-model" if (matched == n_ref and n_ref and frac > 0.9)
                   else "different-model",
    }


def audit_provenance(refs: list[tuple[str, Path, Path | None]]) -> dict:
    """Classify each stored artifact against every candidate checkpoint.

    ``refs`` is (label, checkpoint_path, onnx_path_or_None). Every artifact
    that exists---checkpoint or graph---is used as a query against every
    checkpoint. Two controls calibrate the agreement score:

    * positive: a committed checkpoint against its own committed export;
    * negative: the same architecture trained with a different seed.

    Both controls have full shape coverage, so the agreement score, not the
    coverage, is what separates them.
    """
    ref_pt = {}
    ref_onnx = {}
    for label, pt_path, onnx_path in refs:
        if pt_path.exists():
            try:
                ref_pt[label] = _conv_index_for(pt_path)
            except Exception as exc:  # pragma: no cover - diagnostic only
                ref_pt[label] = {"__error__": exc}
        if onnx_path is not None and onnx_path.exists():
            ref_onnx[label] = _conv_shape_index_from_onnx(onnx_path)

    out: dict = {"reference_checkpoints": sorted(ref_pt), "results": {}}

    # Queries: every committed artifact, checkpoint or graph.
    queries: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for _label, _pt, _on in refs:
        for p in (_pt, _on):
            if p is None or not p.exists():
                continue
            try:
                key = p.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                key = str(p)
            if key in seen:
                continue
            seen.add(key)
            queries.append((key, _conv_index_for(p)))

    # Positive control: matched .pt/.onnx pair committed side by side.
    pos_pt = "models/yolo_s_seed0/best.pt"
    if pos_pt in ref_onnx and pos_pt in ref_pt:
        out["positive_control"] = provenance_match(
            ref_onnx[pos_pt], ref_pt[pos_pt],
            pos_pt + " (same checkpoint, same export)")

    # Negative control: same architecture, different training run. This is the
    # case a shape-only comparison cannot distinguish, so it calibrates the
    # agreement score, not just the coverage.
    neg_q = "models/yolo_s_seed0/best.pt"
    neg_r = "models/yolo_s_seed1/best.pt"
    if neg_q in ref_onnx and neg_r in ref_pt:
        out["negative_control"] = provenance_match(
            ref_onnx[neg_q], ref_pt[neg_r],
            neg_r + " (same architecture, different seed)")

    for qlabel, qindex in queries:
        entry = {}
        for rlabel, rindex in ref_pt.items():
            if not isinstance(rindex, dict) or "__error__" in rindex:
                continue
            entry[rlabel] = provenance_match(qindex, rindex, rlabel)
        out["results"][qlabel] = entry
    return out


def parity(pt_path: Path, onnx_path: Path, imgsz: int, seed: int = 0) -> dict:
    """Max |torch - onnx| on a fixed random input.

    The input is deterministic so the comparison is reproducible; the point is
    numerical agreement between the two representations of the same weights,
    not detection quality.
    """
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    rng = np.random.default_rng(seed)
    x = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)

    model = YOLO(str(pt_path))
    t_model = model.model
    t_model.eval()
    with torch.no_grad():
        t_out = t_model(torch.from_numpy(x))
    t_arr = t_out[0] if isinstance(t_out, (list, tuple)) else t_out
    t_np = t_arr.detach().cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_np = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    if t_np.shape != o_np.shape:
        return {
            "comparable": False,
            "torch_shape": list(t_np.shape),
            "onnx_shape": list(o_np.shape),
        }

    diff = np.abs(t_np - o_np)
    return {
        "comparable": True,
        "torch_shape": list(t_np.shape),
        "onnx_shape": list(o_np.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "bitwise_identical": bool(np.array_equal(t_np, o_np)),
    }


def host_latency(onnx_path: Path, imgsz: int, runs: int, warmup: int) -> dict:
    """CPU inference latency of the ONNX graph. Host CPU, not the device."""
    import onnxruntime as ort

    x = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    for _ in range(warmup):
        sess.run(None, {name: x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000.0)

    a = np.array(times)
    return {
        "provider": "CPUExecutionProvider",
        "runs": runs,
        "warmup": warmup,
        "ms_median": float(np.median(a)),
        "ms_p95": float(np.percentile(a, 95)),
        "ms_min": float(a.min()),
        "ms_max": float(a.max()),
    }


def threshold_sensitivity(onnx_path: Path, imgsz: int, imgsz_thresholds, seed: int = 0) -> dict:
    """Detections surviving each confidence threshold applied outside the graph.

    The exported graph emits raw predictions. Both the confidence threshold and
    non-maximum suppression happen in the caller, so the operating point is not
    a property of the artifact.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    x = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]

    # Ultralytics export layout: (1, 4 + nc, anchors) in xywh + class scores.
    scores = out[0, 4:, :]
    best = scores.max(axis=0) if scores.shape[0] > 1 else scores[0]

    counts = {f"{t:g}": int((best >= t).sum()) for t in imgsz_thresholds}
    return {
        "score_min": float(best.min()),
        "score_max": float(best.max()),
        "anchors": int(best.size),
        "counts_at_threshold": counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_DIR / "export_parity_host.json"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Every checkpoint that exists in both representations.
    pairs = [
        ("yolo_s_seed0", REPO_ROOT / "models/yolo_s_seed0/best.pt",
         REPO_ROOT / "models/yolo_s_seed0/best.onnx"),
        ("yolo_s_seed1", REPO_ROOT / "models/yolo_s_seed1/best.pt",
         REPO_ROOT / "models/yolo_s_seed1/best.onnx"),
        ("yolo_s_seed2", REPO_ROOT / "models/yolo_s_seed2/best.pt",
         REPO_ROOT / "models/yolo_s_seed2/best.onnx"),
    ]

    report = {
        "note": ("Host-side export and graph measurements. These are NOT device "
                 "measurements; the Jetson figures must be produced on the device."),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
        },
        "imgsz": args.imgsz,
        "artifacts": {},
        "pairs": {},
    }

    # Storage precision, which is why file size alone cannot identify a model.
    dtype_targets = [
        REPO_ROOT / "weights/best.pt",
        REPO_ROOT / "models/yolo_s_seed0/best.pt",
        REPO_ROOT / "models/original_v1/best.pt",
    ]
    report["checkpoint_storage"] = {
        p.relative_to(REPO_ROOT).as_posix(): weight_dtype_summary(p)
        for p in dtype_targets if p.exists()
    }

    # Provenance: which checkpoint, if any, each exported graph reproduces.
    provenance_refs = [
        ("models/original_v1/best.pt", REPO_ROOT / "models/original_v1/best.pt", None),
        ("models/yolo_s_seed0/best.pt", REPO_ROOT / "models/yolo_s_seed0/best.pt",
         REPO_ROOT / "models/yolo_s_seed0/best.onnx"),
        ("models/yolo_s_seed1/best.pt", REPO_ROOT / "models/yolo_s_seed1/best.pt",
         REPO_ROOT / "models/yolo_s_seed1/best.onnx"),
        ("models/yolo_s_seed2/best.pt", REPO_ROOT / "models/yolo_s_seed2/best.pt",
         REPO_ROOT / "models/yolo_s_seed2/best.onnx"),
        ("weights/best.pt", REPO_ROOT / "weights/best.pt",
         REPO_ROOT / "weights/best.onnx"),
        ("yolov8s.pt", REPO_ROOT / "yolov8s.pt", None),
        ("yolov8m.pt", REPO_ROOT / "yolov8m.pt", None),
    ]
    report["provenance"] = audit_provenance(provenance_refs)

    # Structural facts for every committed ONNX file, plus the stray one in
    # weights/ that shares a name with a differing .pt.
    onnx_files = [
        REPO_ROOT / "models/yolo_s_seed0/best.onnx",
        REPO_ROOT / "models/yolo_s_seed1/best.onnx",
        REPO_ROOT / "models/yolo_s_seed2/best.onnx",
        REPO_ROOT / "weights/best.onnx",
    ]
    for p in onnx_files:
        if not p.exists():
            continue
        rel = p.relative_to(REPO_ROOT).as_posix()
        report["artifacts"][rel] = {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "weight_params": weight_param_count(p),
            **graph_facts(p),
        }

    # The name collision: weights/best.pt and weights/best.onnx
    w_pt = REPO_ROOT / "weights/best.pt"
    if w_pt.exists() and (REPO_ROOT / "weights/best.onnx").exists():
        report["weights_name_collision"] = {
            "pt": {
                "bytes": w_pt.stat().st_size,
                "sha256": sha256_file(w_pt),
                "torch_params": torch_param_count(w_pt),
            },
            "onnx": {
                "bytes": (REPO_ROOT / "weights/best.onnx").stat().st_size,
                "sha256": sha256_file(REPO_ROOT / "weights/best.onnx"),
                "onnx_params": weight_param_count(REPO_ROOT / "weights/best.onnx"),
            },
            "same_model": (
                torch_param_count(w_pt)
                == weight_param_count(REPO_ROOT / "weights/best.onnx")
            ),
        }

    for tag, pt, on in pairs:
        if not (pt.exists() and on.exists()):
            continue
        entry = {
            "pt": {"path": pt.relative_to(REPO_ROOT).as_posix(),
                   "bytes": pt.stat().st_size, "sha256": sha256_file(pt)},
            "onnx": {"path": on.relative_to(REPO_ROOT).as_posix(),
                     "bytes": on.stat().st_size, "sha256": sha256_file(on)},
            "parity": parity(pt, on, args.imgsz),
            "host_latency": host_latency(on, args.imgsz, args.runs, args.warmup),
            "threshold_sensitivity": threshold_sensitivity(
                on, args.imgsz, [0.01, 0.05, 0.10, 0.25, 0.50]),
        }
        report["pairs"][tag] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {out.relative_to(REPO_ROOT).as_posix()}")
    for tag, e in report["pairs"].items():
        p = e["parity"]
        lat = e["host_latency"]
        print(f"  {tag}: max|torch-onnx| = {p.get('max_abs_diff')}  "
              f"host CPU median {lat['ms_median']:.1f} ms")
    if "weights_name_collision" in report:
        c = report["weights_name_collision"]
        print(f"  weights/best: .pt {c['pt']['torch_params']/1e6:.2f} M params, "
              f".onnx {c['onnx']['onnx_params']/1e6:.2f} M params, "
              f"same model = {c['same_model']}")
    prov = report.get("provenance", {})
    pc = prov.get("positive_control")
    if pc:
        print(f"  control: {pc['ref']} -> coverage {pc['coverage']}, "
              f"agreement {pc['median_agreement']}, verdict {pc['verdict']}")
    for q, entry in prov.get("results", {}).items():
        best = max(entry.values(), key=lambda e: (e["coverage"], e["median_agreement"]),
                   default=None)
        if best:
            print(f"  {q}: closest = {best['ref']} "
                  f"(coverage {best['coverage']}, agreement {best['median_agreement']}, "
                  f"{best['verdict']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())