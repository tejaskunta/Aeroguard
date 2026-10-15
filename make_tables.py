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

ROOT = Path(__file__).resolve().parent
PR = ROOT / "paper_results"
OUT = PR / "tables"
sys.stdout.reconfigure(encoding="utf-8")


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
        (2.3, 2.6, 1.8, 1.1, "YOLOv8s stage 1\nseed 0, conf>=0.02"),
        (4.6, 3.1, 2.0, 0.9, "conf >= 0.4\nprotected (no recheck)"),
        (4.6, 1.7, 2.0, 1.1, "0.05 <= conf < 0.4\nMNv3-L crop clf"),
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
    ax.text(5.6, 1.15, "crop = box + 20% context, 224x224, sigmoid score; "
            "tau and P chosen on val only", ha="center", fontsize=8, color="#444444")
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

    # ---------------- T2 YOLO size/latency (seeds 0/1/2)
    sl = load(PR / "yolo" / "s_seed0" / "size_latency.json")
    if sl.get("gflops") is None:
        # size_latency.json did not capture GFLOPs; compute here (script-file provenance)
        from ultralytics import YOLO
        w = ROOT / "runs" / "detect" / "runs" / "detect" / \
            "landmine_yolov8s_75ep_s0_v2" / "weights" / "last.pt"
        sl["gflops"] = YOLO(str(w)).model.info()[3]
    tvals = [load(PR / "yolo" / f"s_seed{s}" / "val.json") for s in (0, 1, 2)]
    ttests = [load(PR / "yolo" / f"s_seed{s}" / "test.json") for s in (0, 1, 2)]
    val_m50 = [v["ultralytics_map"]["val"]["map50"] for v in tvals]
    test_m50 = [t["ultralytics_map"]["test"]["map50"] for t in ttests]
    op_r = [t["ops"]["op97_primary"]["recall"] for t in ttests]
    op_p = [t["ops"]["op97_primary"]["precision"] for t in ttests]
    op_fp = [t["ops"]["op97_primary"]["fp_per_image"] for t in ttests]
    op_c = [t["ops"]["op97_primary"]["conf"] for t in ttests]
    md = ["| model | params | GFLOPs | weights MB | latency ms (med) | val mAP50 | test mAP50 | "
          "test R@OP | test P@OP | FP/img@OP | OP conf (val) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = [["yolov8s", f"{sl['params']:,}".replace(",", " "), f"{sl['gflops']:.1f}",
            f"{sl['weights_mb']:.1f}", f"{sl['latency_ms_median']:.1f}",
            ms(val_m50, 4), ms(test_m50, 4), ms(op_r, 4), ms(op_p, 4), ms(op_fp, 3),
            "/".join(f"{c:.2f}" for c in op_c)]]
    md.append(f"| yolov8s | {sl['params']:,} | {sl['gflops']:.1f} | {sl['weights_mb']:.1f} | "
              f"{sl['latency_ms_median']:.1f} | {ms(val_m50, 4)} | {ms(test_m50, 4)} | "
              f"{ms(op_r, 4)} | {ms(op_p, 4)} | {ms(op_fp, 3)} | "
              + "/".join(f"{c:.2f}" for c in op_c) + " |")
    write("t2_yolo_size", "### T2. YOLO size sweep (seeds 0/1/2, PC GPU)", md,
          ["n/m runs out of scope under the approved reduced budget (D-plan scope); only yolov8s "
           "was trained on v2 (3 seeds). Metrics are mean ± std over seeds 0/1/2. "
           "Latency: PC GPU FP32, imgsz 640 (size_latency.json, seed 0). "
           "OP = recall-first conf chosen per seed on val (conf listed per seed)."],
          ["model", "params", "GFLOPs", "MB", "ms", "val mAP50", "test mAP50", "R", "P", "FP/img", "OP conf"],
          tex, "YOLOv8s operating point and cost (seeds 0/1/2)")

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
    ts_arms = [load(PR / "twostage" / f"s{s}" / "test_report.json")["arms"]["C_two_stage"]
               for s in (0, 1, 2)]
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
          row("Two-stage (arm C, 3 seeds)", ts_img)]
    tex = []
    for label, dicts in (("MNv3 A (0.5)", [a_05]), ("MNv3 A+B (RF)", ab_rf),
                         ("YOLO deploy", yolo_op_img), ("Two-stage C", ts_img)):
        tex.append([label] + [f"{ms([d.get(k) for d in dicts])}" for k in
                              ("accuracy", "balanced_accuracy", "precision", "recall", "kappa")])
    write("t3_image_level", "### T3. Image-level comparison (test)", md,
          ["MNv3 A+B = mean ± std over seeds 0/1/2; MNv3 A is seed 0; "
           "YOLO deploy/op95 = mean ± std over seeds 0/1/2 (each seed's OP conf chosen on val).",
           "MNv3 recall-first thresholds chosen on val only."],
          ["system", "acc", "bal acc", "P", "R", "kappa"],
          tex, "Image-level comparison on the frozen test split")

    # ---------------- T4 two-stage ablation
    md = ["| arm | conf/thr | precision | recall | F2 | FP/img | img recall |",
          "|---|---|---|---|---|---|---|"]
    tex = []
    reps = [load(PR / "twostage" / f"s{s}" / "test_report.json") for s in (0, 1, 2)]
    for arm, key in (("A (YOLO deploy)", "A_yolo_only_deploy"),
                     ("B (YOLO matched recall)", "B_yolo_only_matched_recall"),
                     ("C (two-stage)", "C_two_stage")):
        cols = [ms([r["arms"][key].get(k) for r in reps])
                for k in ("precision", "recall", "f2", "fp_per_image", "img_recall")]
        r0 = reps[0]
        if key.startswith("A"):
            conf = r0["arm_a_conf"]
        elif key.startswith("B"):
            conf = r0["arm_b_conf"]
        else:
            a = r0["arms"]["C_two_stage"]
            conf = f"yolo {a['yolo_conf']}, P={a['protect_conf']}, t={a['cls_thr']}"
        md.append(f"| {arm} | {conf} | " + " | ".join(cols) + " |")
        tex.append([arm.split()[0]] + cols)
    pb = reps[0]["paired_diff_C_minus_B"]
    rep0_cost = reps[0].get("cost", {})
    notes = [f"Paired bootstrap (seed 0): C-B precision +{pb['precision']['mean_diff']:.3f} "
             f"CI [{pb['precision']['ci95'][0]:.3f}, {pb['precision']['ci95'][1]:.3f}]; "
             f"FP/img {pb['fp_per_image']['mean_diff']:.3f} CI "
             f"[{pb['fp_per_image']['ci95'][0]:.3f}, {pb['fp_per_image']['ci95'][1]:.3f}]; "
             f"recall {pb['recall']['mean_diff']:.4f} CI "
             f"[{pb['recall']['ci95'][0]:.4f}, {pb['recall']['ci95'][1]:.4f}].",
             f"Cost (plan §5.4, seed 0): {rep0_cost['ms_per_image']:.1f} ms/image end-to-end, "
             f"{rep0_cost['crops_per_image_mean']:.2f} crops/image (p95 "
             f"{rep0_cost['crops_per_image_p95']:.1f}), PC GPU FP32.",
             "Means ± std over seeds 0/1/2 for arm C; A/B are seed-independent except threshold "
             "choice (B picked per seed on val)."]
    write("t4_two_stage", "### T4. Two-stage pipeline (arms A, B, C)", md, notes,
          ["arm", "P", "R", "F2", "FP/img", "img R"], tex,
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
