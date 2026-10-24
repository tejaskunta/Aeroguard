#!/usr/bin/env python3
"""Leaky-v1 vs clean-v2: what the old split can and cannot measure.

A model trained before the re-split saw duplicate groups of many v2 test
images, so it cannot be evaluated cleanly on the full v2 test set. We:
  1. report the leaky model on its own v1 test split (the number the old
     pipeline reported; note v1 test has zero empty labels),
  2. report it on full v2 test and quantify group-level contamination from
     v1 train (images sharing a v2 group with a v1-train image),
  3. evaluate both the leaky model and the clean-v2 seed-0 model on the
     contamination-free v2 test subset (post-hoc, hash/group-based only).

Writes paper_results/leaky_v1/comparison.{json,md}
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import phase3_eval as p3
import two_stage_eval as ts
from ultralytics import YOLO

PR = ROOT / "paper_results"
V1 = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined")
V2 = Path(r"C:\Users\Delta\Downloads\Landmine_project\Datasets\landmine_combined_v2")
LEAKY_RUN = ROOT / "runs" / "detect" / "runs" / "detect" / "landmine_yolov8s_75ep_leakyv1"
CLEAN_RUN = ROOT / "runs" / "detect" / "runs" / "detect" / "landmine_yolov8s_75ep_s0_v2"
OUT = PR / "leaky_v1"
SUBSET = OUT / "clean_subset"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    v1_train = {p.name for p in (V1 / "train" / "images").iterdir()}
    manifest = list(csv.DictReader(open(PR / "split_v2" / "split_manifest.csv",
                                       encoding="utf-8")))
    groups: dict[str, list[str]] = {}
    for r in manifest:
        groups.setdefault(r["group_id"], []).append(Path(r["image"]).name)
    test_rows = [r for r in manifest if r["split"] == "test"]
    contaminated = {Path(r["image"]).name for r in test_rows
                    if any(m in v1_train for m in groups[r["group_id"]])}
    clean = {Path(r["image"]).name for r in test_rows} - contaminated

    v1_val = json.load(open(OUT / "val.json", encoding="utf-8"))
    v1_test = json.load(open(OUT / "test.json", encoding="utf-8"))
    v2_ref = json.load(open(PR / "yolo" / "s_seed0" / "test.json", encoding="utf-8"))
    ckpt = v1_val["checkpoint_choice"]["chosen"]
    op_conf = float(v1_test["chosen_on_val"]["primary_op_conf"])
    ref_conf = float(v2_ref["chosen_on_val"]["primary_op_conf"])
    leaky_w = LEAKY_RUN / "weights" / f"{ckpt}.pt"
    clean_w = CLEAN_RUN / "weights" / "last.pt"
    print(f"leaky {leaky_w} @{op_conf}; clean {clean_w} @{ref_conf}")

    leaky = YOLO(str(leaky_w))
    clean_model = YOLO(str(clean_w))
    paths, labels, names = p3.split_paths(V2, "test")
    leaky_recs = p3.predict_records(leaky, paths, labels, names, conf=0.001)
    clean_recs = p3.predict_records(clean_model, paths, labels, names, conf=0.001)
    p3.save_records(OUT / "leaky_preds_on_v2_test.json", leaky_recs)
    p3.save_records(OUT / "clean_v2_preds_on_v2_test.json", clean_recs)

    # contamination-free subset
    if SUBSET.exists():
        shutil.rmtree(SUBSET)
    for sub in ("images", "labels"):
        (SUBSET / sub).mkdir(parents=True)
    for n in clean:
        shutil.copy2(V2 / "test" / "images" / n, SUBSET / "images" / n)
        lbl = V2 / "test" / "labels" / (n.rsplit(".", 1)[0] + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, SUBSET / "labels" / lbl.name)
    (SUBSET / "data.yaml").write_text(
        f"path: {SUBSET.as_posix()}\ntrain: images\nval: images\nnc: 1\nnames:\n- Landmine\n",
        encoding="utf-8")

    def subset_map(model):
        m = model.val(data=str(SUBSET / "data.yaml"), split="val", conf=0.001,
                      imgsz=640, iou=0.7, max_det=300, plots=False, verbose=False)
        return {"map50": float(m.box.map50), "map50_95": float(m.box.map)}

    leaky_sub_map = subset_map(leaky)
    clean_sub_map = subset_map(clean_model)
    leaky_sub = p3.metrics_at([r for r in leaky_recs if r["name"] in clean], op_conf)
    clean_sub = p3.metrics_at([r for r in clean_recs if r["name"] in clean], ref_conf)
    leaky_full_map = p3.ultralytics_map(leaky, V2 / "data.yaml", "test")

    v1_op = v1_test["ops"]["op97_primary"]
    ref_op = v2_ref["ops"]["op97_primary"]
    leaky_full = p3.metrics_at(leaky_recs, op_conf)
    clean_full = p3.metrics_at(clean_recs, ref_conf)

    out = {
        "checkpoints": {"leaky": str(leaky_w), "clean_v2": str(clean_w)},
        "conf": {"leaky": op_conf, "clean_v2": ref_conf},
        "contamination": {
            "v2_test_images": len(test_rows),
            "groups_touching_v1_train": len(contaminated),
            "fraction": round(len(contaminated) / len(test_rows), 4),
            "clean_subset_images": len(clean),
            "clean_subset_instances": sum(len(r["gt"]) for r in leaky_recs
                                          if r["name"] in clean),
        },
        "v1_test_own_split": {
            "map50": v1_test["ultralytics_map"]["test"]["map50"],
            "map50_95": v1_test["ultralytics_map"]["test"]["map50_95"],
            "precision": v1_op["precision"], "recall": v1_op["recall"],
            "fp_per_image": None,
            "negatives": 0,
        },
        "full_v2_test": {
            "leaky": {"map50": leaky_full_map["map50"],
                      "map50_95": leaky_full_map["map50_95"], **leaky_full},
            "clean_v2": {"map50": v2_ref["ultralytics_map"]["test"]["map50"], **clean_full},
        },
        "clean_subset": {
            "leaky": {**leaky_sub_map, **leaky_sub},
            "clean_v2": {**clean_sub_map, **clean_sub},
        },
    }
    (OUT / "comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = [
        "# Leaky-v1 vs clean-v2 (yolov8s)",
        "",
        f"Leaky checkpoint `{ckpt}` + OP conf {op_conf} chosen on v1 val. Clean-v2 seed 0 OP "
        f"conf {ref_conf} chosen on v2 val.",
        f"**Contamination:** {out['contamination']['groups_touching_v1_train']} of "
        f"{len(test_rows)} v2 test images ({100*out['contamination']['fraction']:.1f}%) share a "
        "v2 group with an image the v1 model trained on; only the remaining "
        f"{len(clean)} images are a clean common test.",
        "",
        "| model | eval set | mAP50 | P | R | FP/img |",
        "|---|---|---:|---:|---:|---:|",
        f"| leaky v1 | v1 test (own split) | {out['v1_test_own_split']['map50']:.4f} | "
        f"{out['v1_test_own_split']['precision']:.3f} | "
        f"{out['v1_test_own_split']['recall']:.3f} | n/a (0 neg) |",
        f"| leaky v1 | v2 test, full (contaminated) | "
        f"{leaky_full_map['map50']:.4f} | {leaky_full['precision']:.3f} | "
        f"{leaky_full['recall']:.3f} | {leaky_full['fp_per_image']:.3f} |",
        f"| clean v2 seed 0 | v2 test, full | {v2_ref['ultralytics_map']['test']['map50']:.4f} | "
        f"{clean_full['precision']:.3f} | {clean_full['recall']:.3f} | "
        f"{clean_full['fp_per_image']:.3f} |",
        "",
        "## Contamination-free subset",
        "",
        "| model | mAP50 | mAP50-95 | P | R | FP/img |",
        "|---|---:|---:|---:|---:|---:|",
        f"| leaky v1 | {leaky_sub_map['map50']:.4f} | {leaky_sub_map['map50_95']:.4f} | "
        f"{leaky_sub['precision']:.3f} | {leaky_sub['recall']:.3f} | "
        f"{leaky_sub['fp_per_image']:.3f} |",
        f"| clean v2 seed 0 | {clean_sub_map['map50']:.4f} | {clean_sub_map['map50_95']:.4f} | "
        f"{clean_sub['precision']:.3f} | {clean_sub['recall']:.3f} | "
        f"{clean_sub['fp_per_image']:.3f} |",
        "",
        "## Reading",
        "",
        "A pre-re-split model cannot be fairly scored on the full v2 test set: three quarters of "
        "its test images sit in groups it trained on. On the contamination-free subset the two "
        "models are directly comparable; the subset is small, so the estimate is low-power. The "
        "old split's real defects are structural: leaked val used for model selection, a test "
        "split with zero negatives (false positives unmeasurable), and no clean way to evaluate "
        "any pre-re-split checkpoint.",
    ]
    (OUT / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {OUT / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
