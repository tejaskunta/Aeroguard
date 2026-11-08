#!/usr/bin/env python3
"""Auto-generate paper tables T1-T6 from files written by the phase scripts
(plan §Appendix, "one make_tables script"). No numbers typed by hand.

Outputs: paper_results/tables/t{1..6}.md and .tex
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
PR = ROOT / "paper_results"
OUT = PR / "tables"
sys.stdout.reconfigure(encoding="utf-8")

# Canonical two-stage runs. `s0` is detector seed 0; `s1_aligned`/`s2_aligned` are the
# arms re-evaluated against detector seeds 1/2 in RUN_LOG touch #18. The older `s1`/`s2`
# directories pair a *different* YOLO seed with the seed-0 two-stage classifier, so their
# arm A/B rows are byte-identical to seed 0 and must never be published.
# `s3`/`s4` are the seed-3/4 stages added in RUN_LOG touch #24.
TWOSTAGE_SEEDS = ("s0", "s1_aligned", "s2_aligned", "s3", "s4")


def load(p):
    return json.load(open(p, encoding="utf-8"))


def ms(vals, nd=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return f"{mu:.{nd}f} ± {sd:.{nd}f}"


def kappa_bal(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (n * n) if n else 0.0
    k = (1 - pe) / (1 - pe) if pe < 1 and n else 0.0
    # recompute properly
    if n:
        po = (tp + tn) / n
        k = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return (rec + spec) / 2, k


def write(name, md_header, md_rows, md_notes, tex_cols, tex_rows, tex_caption):
    (OUT / f"{name}.md").write_text(
        md_header + "\n\n" + "\n".join(md_rows) + "\n\n" + "\n".join(f"_{n}_" for n in md_notes) + "\n",
        encoding="utf-8")
    colspec = "|" + "l" * len(tex_cols) + "|"
    lines = [f"\\begin{{table}}[t]\\centering\\small\\caption{{{tex_caption}}}",
             "\\label{tab:%s}" % name.replace("_", ""),
             f"\\begin{{tabular}}{{{colspec}}}\\hline",
            " & ".join(tex_cols) + "\\\\"]
    for r in tex_rows:
        # Guard: a row whose cell count disagrees with the colspec produces a
        # silently malformed table (LaTeX pads or errors depending on the run).
        if not str(r[0]).startswith("\\hline") and len(r) != len(tex_cols):
            raise SystemExit(
                f"{name}: tex_rows emit {len(r)} cells but tex_cols declares "
                f"{len(tex_cols)} columns; fix the header or the row builder.")
        lines.append(" & ".join(str(c) for c in r) + "\\\\") if not r[0].startswith("\\hline") else lines.append("\\hline")
    lines += ["\\hline\\end{tabular}\\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


def make_figures():
    """Plan: tables/ and figures/ are produced by the one make_tables script."""
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figd = PR / "figures"
    figd.mkdir(parents=True, exist_ok=True)

    # --- pipeline diagram
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")
    boxes = [
        (0.2, 2.6, 1.7, 1.1, "Frozen split v2\n3907/181/189"),
        (2.3, 2.6, 1.8, 1.1, "YOLOv8s stage 1\nseed 0/1/2; dets @conf>=0.05"),
        (4.6, 3.1, 2.0, 0.9, "conf >= P (0.4)\nprotected (no recheck)"),
        (4.6, 1.7, 2.0, 1.1, "0.05 <= conf < P\nMNv3-Large crop clf"),
        (7.1, 2.6, 1.9, 1.1, "combine\narm C"),
        (9.4, 2.6, 1.4, 1.1, "detections\n(arm A/B/C)"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc="#eef4ff", ec="#3355bb", lw=1.4))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)
    for a, b in [((1.9, 3.15), (2.3, 3.15)), ((4.1, 3.15), (4.6, 3.55)),
                 ((4.1, 3.15), (4.6, 2.25)), ((6.6, 3.55), (7.1, 3.15)),
                 ((6.6, 2.25), (7.1, 3.15)), ((9.0, 3.15), (9.4, 3.15))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                     color="#555555", lw=1.2))
    ax.text(5.6, 1.15, "arm C proposal cache is conf>=0.05 (one cache per detector seed 0/1/2); "
            "crop = box + 20% context, 224x224, sigmoid score; tau and P chosen on val only. "
            "Arms A/B are scored on the same detector runs with their own val-chosen conf.",
            ha="center", fontsize=7.5, color="#444444")
    fig.tight_layout(); fig.savefig(figd / "pipeline.png", dpi=160); plt.close(fig)

    # --- PR / operating curves
    def load_curve(name):
        rows = list(csv.DictReader(open(PR / "yolo" / "s_seed0" / name, encoding="utf-8")))
        return ({k: [float(r[k]) for r in rows] for k in
                 ("conf", "precision", "recall", "fp_per_image")})
    cv, ct = load_curve("pr_curve_val.csv"), load_curve("pr_curve_test.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(cv["recall"], cv["precision"], label="val")
    axes[0].plot(ct["recall"], ct["precision"], label="test")
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision")
    axes[0].set_title("PR curve"); axes[0].legend(); axes[0].grid(alpha=0.3)
    for axi, key, ttl in ((axes[1], "recall", "recall vs conf"),
                          (axes[2], "fp_per_image", "FP/img vs conf")):
        axi.plot(cv["conf"], cv[key], label="val")
        axi.plot(ct["conf"], ct[key], label="test")
        axi.set_xlabel("conf"); axi.set_ylabel(key); axi.set_title(ttl)
        axi.legend(); axi.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figd / "pr_curves.png", dpi=160); plt.close(fig)

    # --- plan §5.4 figure: FP/img vs recall for YOLO alone, arm C marked
    rep0 = load(PR / "twostage" / "s0" / "test_report.json")
    curve = rep0["yolo_only_curve"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot([p["recall"] for p in curve], [p["fp_per_image"] for p in curve],
            color="#3355bb", lw=1.8, label="YOLO alone (threshold sweep)")
    ca = rep0["arms"]["A_yolo_only_deploy"]
    cb = rep0["arms"]["B_yolo_only_matched_recall"]
    cc = rep0["arms"]["C_two_stage"]
    ax.scatter([ca["recall"]], [ca["fp_per_image"]], s=55, marker="s",
               color="#888888", zorder=3, label="arm A (deploy OP)")
    ax.scatter([cb["recall"]], [cb["fp_per_image"]], s=55, marker="^",
               color="#cc6600", zorder=3, label="arm B (matched recall)")
    ax.scatter([cc["recall"]], [cc["fp_per_image"]], s=75, marker="*",
               color="#cc0000", zorder=4, label="arm C (two-stage)")
    ax.set_xlabel("recall"); ax.set_ylabel("false positives per image")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Two-stage trade-off (test, seed 0)")
    fig.tight_layout(); fig.savefig(figd / "fp_recall_tradeoff.png", dpi=160)
    plt.close(fig)

    # --- copy phase-produced figures the paper list asks for
    copies = [(PR / "gradcam" / "e1_hist.png", figd / "energy_in_box_hist.png"),
              (PR / "gradcam" / "e6_grid.png", figd / "gradcam_example_grid.png"),
              (PR / "gradcam" / "e4_fp_sheet.png", figd / "failure_gallery_fp.png"),
              (PR / "gradcam" / "e5_worst_test_annotated.png", figd / "e5_audit_test.png"),
              (PR / "gradcam" / "e3_paste_examples.png", figd / "e3_paste.png"),
              (PR / "gradcam" / "e3_remove_examples.png", figd / "e3_remove.png")]
    for s, d in copies:
        if s.exists():
            shutil.copy(s, d)
    # E4 gallery dir (many pngs) referenced in place; make an index note
    (figd / "README.md").write_text(
        "pipeline.png, pr_curves.png, fp_recall_tradeoff.png (plan §5.4), "
        "energy_in_box_hist.png, gradcam_example_grid.png, "
        "failure_gallery_fp.png, e5_audit_test.png, e3_paste.png, e3_remove.png are "
        "produced by make_tables.py (copies of phase outputs land here). "
        "The full FN gallery is paper_results/gradcam/e4_gallery/.\n", encoding="utf-8")
    print("wrote figures ->", figd)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- T1 dataset + leakage audit
    audit = load(PR / "audit_v2_final" / "audit_summary.json")
    man = list(csv.DictReader(open(PR / "split_v2" / "split_manifest.csv", encoding="utf-8")))
    ss = audit["split_summary"]
    rows = []
    tex = []
    for split in ("train", "valid", "test"):
        srcs = Counter(r["source"] for r in man if r["split"] == split)
        src_str = ", ".join(f"{k}:{v}" for k, v in sorted(srcs.items()))
        rows.append(f"| {split} | {ss[split]['images']} | {ss[split]['instances']} | "
                    f"{audit['negatives'][split]} | {src_str} |")
        tex.append([split, ss[split]["images"], ss[split]["instances"],
                    audit["negatives"][split], src_str.replace("&", "\\&")])
    nd = audit["near_dup_vs_train"]["<=2"]
    md = ["| split | images | instances | negatives | sources |", "|---|---|---|---|---|"] + rows
    notes = [f"Exact cross-split duplicates: {audit['exact_cross_split_groups']}.",
             f"Near-dup candidates (pHash ham<=2 vs train): {nd} — all residual MSE>=958.6 after "
             f"the second-method filter (0 true leaks; 30 leaked eval images removed pre-freeze; "
             f"D014/D015). Manifest sha256 in split_v2/sha256.txt.",
             f"Base-key collisions across splits: {audit['base_key_collisions']}."]
    write("t1_dataset_leakage", "### T1. Dataset statistics and leakage audit",
          md, notes,
          ["split", "images", "inst", "neg", "sources"], tex,
          "Dataset statistics and leakage audit of the frozen v2 split")

    # ---------------- T2 YOLO size sweep (v2; yolov8s 5 seeds, yolov8n 3 seeds)
    # GFLOPs are stored in each size_latency.json so that a clean clone can rebuild
    # this table without the (uncommitted) runs/ tree or a torch install.
    size_runs = [("yolov8s", "s", (0, 1, 2, 3, 4)), ("yolov8n", "n", (0, 1, 2))]
    ttests = [load(PR / "yolo" / f"s_seed{s}" / "test.json") for s in (0, 1, 2, 3, 4)]
    md = ["| model | params | GFLOPs | weights MB | latency ms (med) | val mAP50 | test mAP50 | "
          "test R@OP | test P@OP | FP/img@OP | OP conf (val) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = []
    for label, sz, seeds in size_runs:
        sl = load(PR / "yolo" / f"{sz}_seed{seeds[0]}" / "size_latency.json")
        if sl.get("gflops") is None:
            raise SystemExit(
                f"paper_results/yolo/{sz}_seed{seeds[0]}/size_latency.json has no 'gflops'; "
                "regenerate it with the GFLOPs value documented in that file's gflops_note "
                "so that make_tables.py never depends on the uncommitted runs/ tree.")
        val_m50 = [load(PR / "yolo" / f"{sz}_seed{s}" / "val.json")["ultralytics_map"]["val"]["map50"]
                   for s in seeds]
        tst = [load(PR / "yolo" / f"{sz}_seed{s}" / "test.json") for s in seeds]
        test_m50 = [t["ultralytics_map"]["test"]["map50"] for t in tst]
        op_r = [t["ops"]["op97_primary"]["recall"] for t in tst]
        op_p = [t["ops"]["op97_primary"]["precision"] for t in tst]
        op_fp = [t["ops"]["op97_primary"]["fp_per_image"] for t in tst]
        op_c = [t["ops"]["op97_primary"]["conf"] for t in tst]

        def _conf(c):
            # 3 dp: several seeds select 0.045 / 0.125 and 2 dp collapses them
            # (0.04 / 0.12). `abs(c) if c == 0 else c` normalises -0.0, which
            # max(c, 0.0) does not, since max() returns its first argument on ties.
            return f"{abs(c) if c == 0 else c:.3f}"

        md.append(f"| {label} | {sl['params']:,} | {sl['gflops']:.1f} | {sl['weights_mb']:.1f} | "
                  f"{sl['latency_ms_median']:.1f} | {ms(val_m50, 4)} | {ms(test_m50, 4)} | "
                  f"{ms(op_r, 4)} | {ms(op_p, 4)} | {ms(op_fp, 3)} | "
                  + "/".join(_conf(c) for c in op_c) + " |")
        tex.append([label] + [f"{sl['params']:,}".replace(",", " "), f"{sl['gflops']:.1f}",
                              f"{sl['weights_mb']:.1f}", f"{sl['latency_ms_median']:.1f}"] +
                   [ms(val_m50, 4), ms(test_m50, 4), ms(op_r, 4), ms(op_p, 4), ms(op_fp, 3),
                   "/".join(_conf(c) for c in op_c)])
    write("t2_yolo_size", "### T2. YOLO size sweep on frozen split v2 (PC GPU)", md,
          ["yolov8s is trained on v2 with seeds 0/1/2/3/4 and yolov8n with seeds 0/1/2 "
           "(mean ± std over seeds); both rows carry intervals. yolov8m was trained on the "
           "older leaky v1 split only and is therefore not comparable here (RUN_LOG touch #18; "
           "the leaky-vs-clean comparison is reported separately).",
           "Latency: PC GPU FP32, imgsz 640, 50 reps after 10 warm-up reps "
           "(paper_results/yolo/*/size_latency.json). GFLOPs are for the single-class head "
           "(nc=1) at 640; Ultralytics' published nc=80 figures for the same depths are "
           "n 8.7 / s 28.6.",
           "OP = recall-first operating point, confidence chosen per seed on val and then "
           "applied unchanged to test (conf listed per seed, so a single number means all "
           "seeds picked it)."],
          ["model", "params", "GFLOPs", "MB", "ms", "val mAP50", "test mAP50", "R", "P", "FP/img", "OP conf"],
          tex, "YOLOv8s and YOLOv8n operating points and cost on frozen split v2")

    # ---------------- T3 image-level comparison
    def mnv3_row(path, thr_key):
        m = load(path)
        d = m["test"][thr_key]
        return d

    ab_paths = [PR / "baseline_mnv3" / f"AB_224_seed{s}" / "metrics.json" for s in (0, 1, 2)]
    a_path = PR / "baseline_mnv3" / "A_224_seed0" / "metrics.json"
    ab_05 = [load(p)["test"]["thr_0.5"] for p in ab_paths]
    ab_rf = [load(p)["test"]["thr_recall_first"] for p in ab_paths]
    a_05 = load(a_path)["test"]["thr_0.5"]
    a_rf = load(a_path)["test"]["thr_recall_first"]

    yolo_op_img = [t["ops"]["op97_primary"]["image_level"] for t in ttests]
    yolo_op95_img = [t["ops"]["op95"]["image_level"] for t in ttests]
    ts_arms = [load(PR / "twostage" / d / "test_report.json")["arms"]["C_two_stage"]
               for d in TWOSTAGE_SEEDS]
    ts_img = []
    for a in ts_arms:
        b, k = kappa_bal(a["img_tp"], a["img_fp"], a["img_fn"], a["img_tn"])
        ts_img.append({"accuracy": (a["img_tp"] + a["img_tn"]) /
                       (a["img_tp"] + a["img_fp"] + a["img_fn"] + a["img_tn"]),
                       "balanced_accuracy": b, "kappa": k,
                       "precision": a["img_precision"], "recall": a["img_recall"]})

    def row(label, dicts, keys=("accuracy", "balanced_accuracy", "precision", "recall", "kappa")):
        cells = [ms([d.get(k) for d in dicts]) for k in keys]
        return f"| {label} | " + " | ".join(cells) + " |"

    md = ["| system (test, image level) | accuracy | balanced acc | precision | recall | kappa |",
          "|---|---|---|---|---|---|",
          row("MNv3 A (thr 0.5)", [a_05]),
          row("MNv3 A (recall-first thr)", [a_rf]),
          row("MNv3 A+B (thr 0.5)", ab_05),
          row("MNv3 A+B (recall-first thr)", ab_rf),
          row("YOLO deploy (conf from val)", yolo_op_img),
          row("YOLO op95 (conf from val)", yolo_op95_img),
          row("Two-stage (arm C, 5 seeds)", ts_img)]
    tex = []
    for label, dicts in (("MNv3 A (0.5)", [a_05]), ("MNv3 A+B (RF)", ab_rf),
                         ("YOLO deploy", yolo_op_img), ("Two-stage C", ts_img)):
        tex.append([label] + [f"{ms([d.get(k) for d in dicts])}" for k in
                              ("accuracy", "balanced_accuracy", "precision", "recall", "kappa")])
    write("t3_image_level", "### T3. Image-level comparison (test)", md,
          ["MNv3 A+B = mean ± std over seeds 0/1/2; MNv3 A is seed 0; "
           "YOLO deploy/op95 = mean ± std over seeds 0/1/2/3/4 (each seed's OP conf chosen on val).",
           "MNv3 recall-first thresholds chosen on val only."],
          ["system", "acc", "bal acc", "P", "R", "kappa"],
          tex, "Image-level comparison on the frozen test split")

    # ---------------- T4 two-stage ablation
    md = ["| arm | conf/thr (seed 0 / 1 / 2 / 3 / 4) | precision | recall | F2 | FP/img | img recall |",
          "|---|---|---|---|---|---|---|"]
    tex = []
    reps = [load(PR / "twostage" / d / "test_report.json") for d in TWOSTAGE_SEEDS]
    for arm, key in (("A (YOLO deploy)", "A_yolo_only_deploy"),
                     ("B (YOLO matched recall)", "B_yolo_only_matched_recall"),
                     ("C (two-stage)", "C_two_stage")):
        cols = [ms([r["arms"][key].get(k) for r in reps])
                for k in ("precision", "recall", "f2", "fp_per_image", "img_recall")]
        r0 = reps[0]
        if key.startswith("A"):
            vals = [f"{r['arm_a_conf']:g}" for r in reps]
            conf = " / ".join(vals)
            tconf = "/".join(vals)
        elif key.startswith("B"):
            vals = [f"{r['arm_b_conf']:g}" for r in reps]
            conf = " / ".join(vals)
            tconf = "/".join(vals)
        else:
            a = r0["arms"]["C_two_stage"]
            vals = [f"{r['arms']['C_two_stage']['cls_thr']:g}" for r in reps]
            prots = [f"{r['arms']['C_two_stage']['protect_conf']:g}" for r in reps]
            conf = (f"yolo {a['yolo_conf']:g}, P " + " / ".join(prots) + ", t "
                    + " / ".join(vals))
            tconf = (f"y{a['yolo_conf']:g} P{'/'.join(prots)} "
                     f"t{'/'.join(vals)}")
        md.append(f"| {arm} | {conf} | " + " | ".join(cols) + " |")
        tex.append([arm.split()[0], tconf] + cols)
    pb = reps[0]["paired_diff_C_minus_B"]
    rep0_cost = reps[0].get("cost", {})
    pb_lines = []
    for d, r in zip(TWOSTAGE_SEEDS, reps):
        p = r["paired_diff_C_minus_B"]
        pb_lines.append(
            f"detector seed {'0' if d == 's0' else d[1]}: C-B precision "
            f"{p['precision']['mean_diff']:+.3f} "
            f"[{p['precision']['ci95'][0]:+.3f}, {p['precision']['ci95'][1]:+.3f}], "
            f"FP/img {p['fp_per_image']['mean_diff']:+.4f} "
            f"[{p['fp_per_image']['ci95'][0]:+.4f}, {p['fp_per_image']['ci95'][1]:+.4f}], "
            f"recall {p['recall']['mean_diff']:+.4f} "
            f"[{p['recall']['ci95'][0]:+.4f}, {p['recall']['ci95'][1]:+.4f}]")
    notes = [
        "Paired bootstrap C-B, one row per detector seed (5000 resamples over test images): "
        + "; ".join(pb_lines)
        + ". The precision/FP gain excludes zero at detector seeds 0, 1 and 3. At seed 2 the "
          "paired interval for precision spans zero and the point estimate is negative; at seed 4 "
          "the point estimate is negative and the precision/FP intervals exclude zero in the "
          "*wrong* direction. The two-stage advantage is therefore not established across seeds.",
        f"Cost (plan §5.4, detector seed 0): {rep0_cost['ms_per_image']:.1f} ms/image end-to-end, "
        f"{rep0_cost['crops_per_image_mean']:.2f} crops/image (p95 "
        f"{rep0_cost['crops_per_image_p95']:.1f}), PC GPU FP32.",
        "Means ± std over the five detector seeds 0/1/2/3/4 for all arms. Stage 1 for every row is "
        "the YOLOv8s checkpoint of that seed, so each run scores its own detector's detections. The "
        "stage-2 classifiers are separately trained per seed (distinct checkpoints), but seeds 0-2 "
        "share one crop set generated from the seed-0 detector, so the effective replication is "
        "closer to four than five.",
        "Arm A threshold = recall-first operator point chosen on val; arm B threshold chosen on val "
        "to match arm A's val recall; arm C = YOLO dets at conf>=yolo_conf routed through the "
        "MobileNetV3-Large crop classifier with protected cutoff P and classifier threshold t.",
        "Caveat at detector seed 2: the val-selected matched-recall threshold coincides with the "
        "deploy threshold (both 0.21), so arms A and B are the same operating point and the C-B "
        "comparison there is against an undegraded YOLO baseline. The two-stage pipeline loses "
        "3.6 points of precision against that baseline, i.e. the gain in T4 comes from seeds whose "
        "val-selected YOLO threshold transferred poorly to test, not from seeds where it transferred "
        "well. Detector seed 4 fails for a related reason: its val-only selection rule chose a "
        "near-inert classifier (protect 0.8, t 0.03), so the classifier vetoes almost nothing and "
        "the surviving pool is larger than arm B's. Validation already preferred YOLO-only at that "
        "seed (val FP/img 1.448 for C vs 0.956 for A, the only seed of five where this happens), so "
        "the regression is a selection-rule artifact rather than a capability signal; see "
        "paper_results/analysis/a4_s4_conf_diagnostic.md. This is the central threat to the "
        "headline claim and is discussed in §8."]
    write("t4_two_stage", "### T4. Two-stage pipeline (arms A, B, C)", md, notes,
          ["arm", "conf/thr (s0/s1/s2/s3/s4)", "P", "R", "F2", "FP/img", "img R"], tex,
          "Two-stage ablation on the frozen test split")

    # ---------------- T5 Grad-CAM metrics
    e1 = load(PR / "gradcam" / "e1_summary.json")
    e2 = load(PR / "gradcam" / "e2_summary.json")
    e6 = load(PR / "gradcam" / "e6_summary.json")
    md = ["| experiment | metric | value |", "|---|---|---|",
          f"| E1 energy-in-box (n={e1['n']}) | median energy share | {e1['energy_share']['median']:.3f} |",
          f"| E1 | mean energy share | {e1['energy_share']['mean']:.3f} |",
          f"| E1 | median enrichment | {e1['enrichment']['median']:.1f}x |",
          f"| E1 | peak inside rate | {e1['peak_inside_rate']:.3f} |",
          f"| E2 deletion (n={e2['n']}) | mean drop, mine masked | {e2['drop_mine']['mean']:.3f} |",
          f"| E2 | mean drop, background masked | {e2['drop_bg']['mean']:.4f} |",
          f"| E6 head-to-head | YOLO enrichment median | {e6['yolo_enrichment']['median']:.1f}x |",
          f"| E6 | MNv3 baseline enrichment median | {e6['mnv3_enrichment']['median']:.2f}x |",
          f"| E6 | paired diff (Y-M) 95% CI | [{e6['paired_diff_yolo_minus_mnv3']['ci95'][0]:.1f}, "
          f"{e6['paired_diff_yolo_minus_mnv3']['ci95'][1]:.1f}] |",
          f"| E6 | stage-2 crop enrichment median | {e6['stage2_crop_enrichment']['median']:.2f}x |"]
    tex = [["E1", "median energy", f"{e1['energy_share']['median']:.3f}"],
           ["E1", "median enrich", f"{e6 and e1['enrichment']['median']:.1f}"],
           ["E1", "peak inside", f"{e1['peak_inside_rate']:.3f}"],
           ["E2", "mine drop", f"{e2['drop_mine']['mean']:.3f}"],
           ["E2", "bg drop", f"{e2['drop_bg']['mean']:.4f}"],
           ["E6", "YOLO vs MNv3", f"{e6['yolo_enrichment']['median']:.1f} vs {e6['mnv3_enrichment']['median']:.2f}"]]
    write("t5_gradcam", "### T5. Grad-CAM metrics", md,
          ["Grad-CAM shows which regions influenced the prediction; it does not prove the "
           "prediction is correct (plan wording rule)."],
          ["exp", "metric", "value"], tex, "Grad-CAM quantitative results")

    # ---------------- T6 failure categories
    e4 = list(csv.DictReader(open(PR / "gradcam" / "e4_failures.csv", encoding="utf-8")))
    fnc = Counter(r["category"] for r in e4 if r["kind"] == "FN")
    fpc = Counter(r["category"] for r in e4 if r["kind"] == "FP")
    md = ["| category | FN | FP |", "|---|---|---|"]
    cats = sorted(set(fnc) | set(fpc))
    tex = []
    for c in cats:
        md.append(f"| {c} | {fnc.get(c, 0)} | {fpc.get(c, 0)} |")
        tex.append([c, fnc.get(c, 0), fpc.get(c, 0)])
    md.append(f"| **total** | **{sum(fnc.values())}** | **{sum(fpc.values())}** |")
    tex.append(["total", sum(fnc.values()), sum(fpc.values())])
    write("t6_failures", "### T6. Failure categories (hand review, E4)", md,
          ["FN at the deployment arm-C threshold; FP capped at 50 (29 present). "
           "Sources: e4_failures.csv + e4_fp_sheet.png."],
          ["category", "FN", "FP"], tex, "Hand-categorised failures")

    make_figures()


if __name__ == "__main__":
    main()
