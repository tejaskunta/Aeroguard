#!/usr/bin/env python3
"""Phase 3 item 6: per-source test results (recall and FP/image per data
source) to expose source shortcuts. Arms A (deploy 0.02) and C (two-stage).

Reads stage-2 raw preds (dump_raw_stage2.py) + split_manifest.csv for the
image->source map. Outputs paper_results/yolo/s_seed0/per_source.{json,md}.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
import two_stage_eval as ts  # noqa: E402

MAN = ROOT / "paper_results/split_v2/split_manifest.csv"
RAW = ROOT / "paper_results/stage2/s2_ratio1_s0/raw_preds_test.json"
REP = ROOT / "paper_results/twostage/s0/test_report.json"
OUT = ROOT / "paper_results/yolo/s_seed0"

src_of = {}
for r in csv.DictReader(open(MAN, encoding="utf-8")):
    if r["split"] == "test":
        src_of[Path(r["image"]).name] = r["source"]

recs = json.loads(RAW.read_text(encoding="utf-8"))
rep = json.loads(REP.read_text(encoding="utf-8"))
c = rep["arms"]["C_two_stage"]
conf_a = rep["arm_a_conf"]

by_src = defaultdict(list)
for r in recs:
    by_src[src_of.get(r["name"], "unknown")].append(r)

out = {"n_sources": len(by_src), "arms": {}}
for arm, kw in (("A_yolo_only_deploy", dict(yolo_conf=conf_a)),
                ("C_two_stage", dict(yolo_conf=c["yolo_conf"], cls_thr=c["cls_thr"],
                                     protect_conf=c["protect_conf"]))):
    out["arms"][arm] = {}
    for src, subset in sorted(by_src.items()):
        m = ts.evaluate(subset, **kw)
        out["arms"][arm][src] = {
            "images": len(subset),
            "precision": m["precision"], "recall": m["recall"],
            "fp_per_image": m["fp_per_image"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]}
    m_all = ts.evaluate(recs, **kw)
    out["arms"][arm]["ALL"] = {"images": len(recs), **{k: m_all[k] for k in
                               ("precision", "recall", "fp_per_image", "tp", "fp", "fn")}}

(OUT / "per_source.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

lines = ["### Per-source test results (source = dataset origin, from split_manifest.csv)", ""]
for arm, rows in out["arms"].items():
    lines += [f"**{arm}**", "",
              "| source | images | TP | FP | FN | precision | recall | FP/img |",
              "|---|---|---|---|---|---|---|---|"]
    for src, d in rows.items():
        lines.append(f"| {src} | {d['images']} | {d['tp']} | {d['fp']} | {d['fn']} | "
                     f"{d['precision']:.3f} | {d['recall']:.3f} | {d['fp_per_image']:.3f} |")
    lines.append("")
(OUT / "per_source.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
