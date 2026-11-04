#!/usr/bin/env python3
"""Statistical analysis (Block B): CIs, source shift, and val-chosen threshold rules.

Everything here is computed from saved per-image records (rule 11), so it needs
no GPU and can be re-run after each experiment block:

  paper_results/yolo/s_seed{0,1,2}/raw_preds_test.json
  paper_results/stage2/s2_ratio1_s{0,1,2}/raw_preds_{val,test}.json   (s1/s2 optional)
  paper_results/ablation/<variant>/raw_preds_{val,test}.json
  paper_results/split_v2/split_manifest.csv

Outputs paper_results/analysis/{stats.json,stats.md}.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "src" / "common"))
sys.stdout.reconfigure(encoding="utf-8")
import two_stage_eval as ts  # noqa: E402

PR = ROOT / "paper_results"
N_BOOT = 2000
SEED = 42


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float(max(0.0, (c - h) / d)), float(min(1.0, (c + h) / d))]


def boot_from_counts(c: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(c)
    out = {"precision": [], "recall": [], "fp_per_image": [], "f2": []}
    for _ in range(n_boot):
        m = ts.metrics_from_counts(c[rng.integers(0, n, n)])
        for k in out:
            out[k].append(m[k])
    return {f"{k}_ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in out.items()}


def counts(records, conf, cls_thr=None, protect=1.01):
    return ts.per_image_counts(records, conf, cls_thr, protect)


def summarize(records, conf, cls_thr=None, protect=1.01, boot=True):
    c = counts(records, conf, cls_thr, protect)
    m = ts.metrics_from_counts(c)
    keep = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()}
    if boot:
        keep.update(boot_from_counts(c))
    n = len(c)
    keep["n_images"] = n
    keep["n_gt"] = int(c[:, 0].sum() + c[:, 2].sum())
    keep["img_recall_ci"] = wilson(m["img_tp"], m["img_tp"] + m["img_fn"])
    keep["img_precision_ci"] = wilson(m["img_tp"], m["img_tp"] + m["img_fp"])
    return keep


# ---------------------------------------------------------------- threshold rules (val -> test)
def pick_fp_budget(val_records, conf, protect, budget):
    best = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        m = ts.evaluate(val_records, conf, float(t), protect)
        if m["fp_per_image"] <= budget:
            best = float(t)
    return best


def pick_max_f2(val_records, conf, protect):
    best_t, best_f2 = 0.0, -1.0
    for t in np.linspace(0.0, 1.0, 101):
        m = ts.evaluate(val_records, conf, float(t), protect)
        if m["f2"] > best_f2:
            best_t, best_f2 = float(t), m["f2"]
    return best_t


def pick_recall_safe(val_records, conf, protect, drop=0.005):
    base = ts.evaluate(val_records, conf, None, protect)["recall"]
    best = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        if ts.evaluate(val_records, conf, float(t), protect)["recall"] >= base - drop:
            best = float(t)
    return best


def pick_yolo_matched(val_records, target_recall):
    best = None
    for t in np.linspace(0.0, 1.0, 201):
        m = ts.evaluate(val_records, float(t), None, 1.01)
        if m["recall"] >= target_recall:
            best = {"yolo_conf": float(t), **m}
    return best


def paired_diff_counts(ca: np.ndarray, cb: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """Paired image bootstrap of (A - B) using per-image counts from two configs.

    ca/cb must be aligned by image (same order), e.g. the same test set scored by
    two different classifiers.
    """
    rng = np.random.default_rng(seed)
    n = len(ca)
    out = {k: [] for k in ("precision", "recall", "fp_per_image")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ma, mb = ts.metrics_from_counts(ca[idx]), ts.metrics_from_counts(cb[idx])
        for k in out:
            out[k].append(ma[k] - mb[k])
    return {k: {"mean_diff": float(np.mean(v)),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for k, v in out.items()}


def pick_yolo_conf(val_records, pred, target=None):
    """YOLO-only confidence meeting a val constraint.

    recall: largest conf (max precision) whose recall stays above target.
    fp:     smallest conf (max recall) whose FP/img stays at or below target.
    """
    base = ts.evaluate(val_records, 0.0, None, 1.01)
    grid = np.linspace(0.0, 1.0, 201)
    best = 0.0
    for t in grid:
        m = ts.evaluate(val_records, float(t), None, 1.01)
        ok = (m["recall"] >= (target if target is not None else base["recall"]) - 0.005
              if pred == "recall" else
              m["fp_per_image"] <= (target if target is not None else base["fp_per_image"]))
        if ok:
            best = float(t)
            if pred == "fp":
                break
    return best


def pick_max_f2_yolo(val_records):
    best_t, best_f2 = 0.0, -1.0
    for t in np.linspace(0.0, 1.0, 201):
        m = ts.evaluate(val_records, float(t), None, 1.01)
        if m["f2"] > best_f2:
            best_t, best_f2 = float(t), m["f2"]
    return best_t


def _row(rule, param, v, test, conf, cls_thr, protect, tag, note=""):
    return {"rule": rule, "chosen_on": "val", "param": float(param), "note": note,
            "val_p": v["precision"], "val_r": v["recall"], "val_fp": v["fp_per_image"],
            "test": summarize(test, conf, cls_thr, protect), "stage": tag}


def rules_two_stage(val, test, prov, P, tag):
    rows = []
    t = pick_recall_safe(val, prov, P)
    rows.append(_row("recall-safe (primary)", t, ts.evaluate(val, prov, t, P),
                     test, prov, t, P, tag))
    t2 = pick_recall_safe(val, prov, 1.01)
    rows.append(_row("no recall protection", t2, ts.evaluate(val, prov, t2, 1.01),
                     test, prov, t2, 1.01, tag,
                     note="classifier can veto any proposal"))
    t3 = pick_max_f2(val, prov, P)
    rows.append(_row("max F2", t3, ts.evaluate(val, prov, t3, P), test, prov, t3, P, tag))
    v4 = ts.evaluate(val, prov, 1.0, P)
    rows.append(_row("conf-only above P (classifier off)", 1.0, v4, test, prov, 1.0, P, tag,
                     note="reject every proposal below P"))
    return rows, {"val_recall": ts.evaluate(val, prov, t, P)["recall"],
                  "val_fp": ts.evaluate(val, prov, t, P)["fp_per_image"]}


def rules_yolo(val, test, tag, target_recall, target_fp):
    rows = []
    t = pick_yolo_conf(val, "recall")
    rows.append(_row("recall-safe", t, ts.evaluate(val, t, None, 1.01),
                     test, t, None, 1.01, tag))
    t2 = pick_yolo_conf(val, "recall", target_recall)
    rows.append(_row("matched recall to two-stage", t2, ts.evaluate(val, t2, None, 1.01),
                     test, t2, None, 1.01, tag,
                     note=f"target val recall {target_recall:.3f}"))
    t3 = pick_yolo_conf(val, "fp", target_fp)
    rows.append(_row("matched FP/img to two-stage", t3, ts.evaluate(val, t3, None, 1.01),
                     test, t3, None, 1.01, tag,
                     note=f"target val FP/img {target_fp:.3f}"))
    t4 = pick_max_f2_yolo(val)
    rows.append(_row("max F2", t4, ts.evaluate(val, t4, None, 1.01),
                     test, t4, None, 1.01, tag))
    return rows


# ---------------------------------------------------------------- source shift / LOO
def source_map() -> dict[str, str]:
    out = {}
    for r in csv.DictReader(open(PR / "split_v2/split_manifest.csv", encoding="utf-8")):
        out[r["image"].split("/")[-1]] = r["source"]
    return out


def shift_table():
    rows = list(csv.DictReader(open(PR / "split_v2/split_manifest.csv", encoding="utf-8")))
    tab = {}
    for r in rows:
        k = (r["split"], r["source"])
        d = tab.setdefault(k, {"images": 0, "boxes": 0, "empty": 0})
        d["images"] += 1
        d["boxes"] += int(r["n_boxes"])
        d["empty"] += int(int(r["n_boxes"]) == 0)
    for d in tab.values():
        d["boxes_per_image"] = d["boxes"] / d["images"] if d["images"] else 0.0
        d["empty_rate"] = d["empty"] / d["images"] if d["images"] else 0.0
    return {f"{a}/{b}": d for (a, b), d in sorted(tab.items())}


def loo_sources(records, test_conf, cls_thr, protect, smap):
    out = {}
    srcs = sorted({smap.get(r["name"], "?") for r in records})
    for s in srcs:
        keep = [r for r in records if smap.get(r["name"], "?") != s]
        m = summarize(keep, test_conf, cls_thr, protect, boot=False)
        out[f"without_{s}"] = {"n_images": m["n_images"], "precision": m["precision"],
                               "recall": m["recall"], "fp_per_image": m["fp_per_image"]}
    # cluster bootstrap over sources
    rng = np.random.default_rng(SEED)
    by = {}
    for r in records:
        by.setdefault(smap.get(r["name"], "?"), []).append(r)
    keys = sorted(by)
    pr, rc, fp = [], [], []
    for _ in range(N_BOOT):
        pick = rng.choice(keys, size=len(keys), replace=True)
        recs = [r for k in pick for r in by[k]]
        m = ts.evaluate(recs, test_conf, cls_thr, protect)
        pr.append(m["precision"]); rc.append(m["recall"]); fp.append(m["fp_per_image"])
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out["cluster_bootstrap"] = {"sources": keys, "precision_ci": q(pr),
                                "recall_ci": q(rc), "fp_per_image_ci": q(fp)}
    return out


# ---------------------------------------------------------------- main
def main() -> int:
    out = {"yolo": {}, "stage2": {}, "ablations": {}, "shift": shift_table()}
    smap = source_map()

    for s in (0, 1, 2):
        d = jload(PR / "yolo" / f"s_seed{s}" / "test.json")
        rec = jload(PR / "yolo" / f"s_seed{s}" / "raw_preds_test.json")
        if not d or not rec:
            continue
        conf = d["chosen_on_val"]["primary_op_conf"]
        m = summarize(rec, conf, None, 1.01)
        out["yolo"][f"seed{s}"] = {"conf": conf, "checkpoint": d["chosen_on_val"]["checkpoint"],
                                   "metrics": m, **loo_sources(rec, conf, None, 1.01, smap)}

    # stage-2 arms at val-chosen rules
    for s in (0, 1, 2):
        d = {"seed0": "s0", "seed1": "s1_aligned", "seed2": "s2_aligned"}[f"seed{s}"]
        rep = jload(PR / "twostage" / d / "test_report.json")
        val = jload(PR / "stage2" / f"s2_ratio1_s{s}" / "raw_preds_val.json")
        test = jload(PR / "stage2" / f"s2_ratio1_s{s}" / "raw_preds_test.json")
        if not rep or not val or not test:
            continue
        a_conf = rep["arm_a_conf"]
        b_conf = rep["arm_b_conf"]
        P = 0.4
        prov = 0.05
        rows, tgt = rules_two_stage(val, test, prov, P, f"seed{s} arm C (two-stage)")
        rows += rules_yolo(val, test, f"seed{s} arm A (YOLO only)",
                           tgt["val_recall"], tgt["val_fp"])
        out["stage2"][f"seed{s}"] = {
            "arm_a_conf": a_conf, "arm_b_conf": b_conf, "proposal_conf": prov,
            "protect_conf": P,
            "arm_a": summarize(test, a_conf, None, 1.01),
            "arm_b": summarize(test, b_conf, None, 1.01),
            "arm_c_primary": summarize(test, prov, rep["chosen_effective"]["cls_thr"], P),
            "paired_C_minus_B": rep.get("paired_diff_C_minus_B"),
            "threshold_rules": rows,
        }

    # ablations
    for d in sorted((PR / "ablation").glob("*")):
        val = jload(d / "raw_preds_val.json")
        test = jload(d / "raw_preds_test.json")
        meta = jload(d / "meta.json")
        if not val or not test:
            continue
        t = pick_recall_safe(val, 0.05, 0.4)
        m = summarize(test, 0.05, t, 0.4)
        out["ablations"][d.name] = {"meta": meta, "cls_thr": t, "arm_c": m}
    # baseline reference (main run, seed 0) under the same rule
    base_val = jload(PR / "stage2/s2_ratio1_s0/raw_preds_val.json")
    base_test = jload(PR / "stage2/s2_ratio1_s0/raw_preds_test.json")
    if base_val and base_test:
        t = pick_recall_safe(base_val, 0.05, 0.4)
        out["ablations"]["baseline_pad02_224_hardneg"] = {
            "meta": {"variant": "baseline", "pad": 0.2, "crop_size": 224, "hardneg": True},
            "cls_thr": t, "arm_c": summarize(base_test, 0.05, t, 0.4)}
        base_counts = counts(base_test, 0.05, t, 0.4)
        for k, v in out["ablations"].items():
            if k == "baseline_pad02_224_hardneg":
                continue
            rec = jload(PR / "ablation" / k / "raw_preds_test.json")
            if not rec or [r["name"] for r in rec] != [r["name"] for r in base_test]:
                continue
            v["paired_vs_baseline"] = paired_diff_counts(
                counts(rec, 0.05, v["cls_thr"], 0.4), base_counts)

    (PR / "analysis").mkdir(parents=True, exist_ok=True)
    (PR / "analysis/stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_md(out)
    print("wrote", PR / "analysis/stats.json")
    return 0


def write_md(out):
    L = ["# Statistical analysis", ""]
    L += ["## YOLOv8s size/seed CIs (image bootstrap, B=2000)", "",
          "| run | conf | P | R | FP/img | image R [95% CI] |", "|---|---:|---:|---:|---:|---|"]
    for k, v in out["yolo"].items():
        m = v["metrics"]
        L.append(f"| {k} | {v['conf']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                 f"{m['fp_per_image']:.3f} | {m['img_recall']:.3f} "
                 f"[{m['img_recall_ci'][0]:.3f}, {m['img_recall_ci'][1]:.3f}] |")
    if out["yolo"]:
        for k, v in out["yolo"].items():
            cb = v.get("cluster_bootstrap")
            if cb:
                L.append(f"\n- {k} source-cluster bootstrap ({len(cb['sources'])} clusters "
                         f"{cb['sources']}): P {cb['precision_ci']}, R {cb['recall_ci']}, "
                         f"FP/img {cb['fp_per_image_ci']}")
                L.append(f"- {k} leave-one-source-out: " + "; ".join(
                    f"{a.replace('without_','')}: P {b['precision']:.3f} R {b['recall']:.3f} "
                    f"FP {b['fp_per_image']:.2f}"
                    for a, b in v.items() if a.startswith("without_")))
    L += ["", "## Source / split shift", "",
          "| split/source | images | boxes | boxes/img | empty rate |", "|---|---:|---:|---:|---:|"]
    for k, d in out["shift"].items():
        L.append(f"| {k} | {d['images']} | {d['boxes']} | {d['boxes_per_image']:.2f} | "
                 f"{d['empty_rate']:.2%} |")
    L += ["", "## Threshold rules (chosen on val, applied once to test)", ""]
    for s, v in out["stage2"].items():
        L.append(f"### {s}")
        L.append("")
        L.append(f"Arm A conf {v['arm_a_conf']}, arm B conf {v['arm_b_conf']}, "
                 f"arm C proposals {v['proposal_conf']} with protection P={v['protect_conf']}.")
        L.append("")
        L.append("| rule | stage | param | val P | val R | test P | test R | test FP/img |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for r in v["threshold_rules"]:
            t = r["test"]
            L.append(f"| {r['rule']} | {r['stage']} | {r['param']:.3f} | {r['val_p']:.3f} | "
                     f"{r['val_r']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | "
                     f"{t['fp_per_image']:.3f} |")
        L.append("")
        L.append("Param is the classifier threshold for arm C and the YOLO confidence for arm A; "
                 "both are chosen on val and applied unchanged to test. Arm A is evaluated from "
                 "the same records, which start at conf 0.02, so it cannot go below seed-0 arm A.")
    if out["ablations"]:
        L += ["## Stage-2 ablations (arm C, recall-safe rule chosen on val)", "",
              "| variant | pad | crop | hardneg | cls thr | P | R | FP/img | dP vs base [95% CI] |",
              "|---|---:|---:|---|---:|---:|---:|---:|---|"]
        base = out["ablations"].get("baseline_pad02_224_hardneg", {}).get("arm_c")
        for k, v in out["ablations"].items():
            m, meta = v["arm_c"], v["meta"]
            if k == "baseline_pad02_224_hardneg":
                dp = "reference"
            else:
                d = v.get("paired_vs_baseline", {}).get("precision", {})
                dp = (f"{d.get('mean_diff', 0):+.3f} "
                      f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]") if d else "-"
            L.append(f"| {k} | {meta['pad']} | {meta['crop_size']} | {meta['hardneg']} | "
                     f"{v['cls_thr']:.2f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['fp_per_image']:.3f} | {dp} |")
        L.append("")
        L.append("dP is a paired image bootstrap of (variant - baseline) computed on the same "
                 "test images; an interval spanning zero means the ablation is indistinguishable "
                 "from the baseline. Recall and FP/img intervals are in stats.json.")
    (PR / "analysis/stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
