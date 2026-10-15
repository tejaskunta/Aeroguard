"""Generate PDF reports (mirrors original safemine_report.pdf layout + research summary).

Outputs:
  reports/yolov8s_seed{0,1,2}_report.pdf  — per-seed training/eval report
  reports/research_summary.pdf            — multi-phase research summary

All metrics are READ from committed JSON/CSV (no recompute). Images come from
training run dirs + paper_results/vis (visualization-only pass).
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(r"C:\Users\Delta\Downloads\SafeMineAI-main")
PR = ROOT / "paper_results"
RUNS = ROOT / "runs" / "detect" / "runs" / "detect"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1x", parent=styles["Heading1"], fontSize=16, spaceAfter=8)
H2 = ParagraphStyle("H2x", parent=styles["Heading2"], fontSize=12, spaceAfter=6)
BODY = ParagraphStyle("BODYx", parent=styles["Normal"], fontSize=9, leading=12)
SMALL = ParagraphStyle("SMALLx", parent=styles["Normal"], fontSize=7.5, leading=9, textColor=colors.grey)


def img(path: Path, max_w: float = 6.8 * inch, max_h: float = 7.2 * inch) -> Image | None:
    if not path or not path.exists():
        return None
    from reportlab.lib.utils import ImageReader

    ir = ImageReader(str(path))
    iw, ih = ir.getSize()
    scale = min(max_w / iw, max_h / ih)
    return Image(str(path), width=iw * scale, height=ih * scale)


def fig_row(*paths: Path, max_w: float = 3.3 * inch, max_h: float = 2.6 * inch) -> list:
    cells = []
    for p in paths:
        im = img(p, max_w, max_h)
        cells.append(im if im else Paragraph("-", SMALL))
    t = Table([cells], colWidths=[max_w] * len(cells))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return [t]


def metric_table(rows: list[list[str]], col_widths=None) -> Table:
    t = Table(rows, colWidths=col_widths or [1.6 * inch] + [1.1 * inch] * (len(rows[0]) - 1))
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def vis_dir(seed: int, split: str) -> Path:
    return PR / "vis" / f"s_seed{seed}" / split / "vis"


def train_run(seed: int) -> Path:
    return RUNS / f"landmine_yolov8s_75ep_s{seed}_v2"


def seed_report(seed: int) -> Path:
    out = REPORTS / f"yolov8s_seed{seed}_report.pdf"
    test = json.load(open(PR / "yolo" / f"s_seed{seed}" / "test.json", encoding="utf-8"))
    val = json.load(open(PR / "yolo" / f"s_seed{seed}" / "val.json", encoding="utf-8"))
    run = train_run(seed)
    op = test["ops"]["op97_primary"]
    op95 = test["ops"]["op95"]
    vm = val["ultralytics_map"]["val"]
    tm = test["ultralytics_map"]["test"]
    opl = op["image_level"]

    doc = SimpleDocTemplate(str(out), pagesize=A4, title=f"SafeMine YOLOv8s seed {seed} report")
    story: list = []
    story.append(Paragraph(f"SafeMine — Landmine Detection Model<br/>YOLOv8s v2 Report (seed {seed})", H1))
    story.append(Paragraph(f"Generated {date.today().isoformat()} · frozen v2 split · ultralytics 8.4.117", SMALL))
    story.append(Spacer(1, 8))

    rows = [
        ["Metric", "Validation", "Test"],
        ["Precision (mean)", f"{vm['precision_mean']:.4f}", f"{tm['precision_mean']:.4f}"],
        ["Recall (mean)", f"{vm['recall_mean']:.4f}", f"{tm['recall_mean']:.4f}"],
        ["mAP@50", f"{vm['map50']:.4f}", f"{tm['map50']:.4f}"],
        ["mAP@50-95", f"{vm['map50_95']:.4f}", f"{tm['map50_95']:.4f}"],
        ["OP conf (val-chosen)", "—", f"{op['conf']}"],
        ["Box P/R @ OP", "—", f"{op['precision']:.4f} / {op['recall']:.4f}"],
        ["FP/img @ OP", "—", f"{op['fp_per_image']:.3f}"],
        ["Image-level P/R @ OP", "—", f"{opl['precision']:.4f} / {opl['recall']:.4f}"],
    ]
    story.append(metric_table(rows))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Training Configuration", H2))
    args_txt = (run / "args.yaml").read_text(encoding="utf-8")
    keep = {}
    for line in args_txt.splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k, _, v = line.partition(":")
            k = k.strip()
            if k in ("model", "epochs", "imgsz", "batch", "seed", "save_period", "lr0", "data"):
                keep[k] = v.strip()
    cfg_rows = [["Param", "Value"]] + [[k, str(keep.get(k, "—"))] for k in
                                       ("model", "epochs", "imgsz", "batch", "seed", "save_period", "lr0", "data")]
    story.append(metric_table(cfg_rows, col_widths=[1.8 * inch, 4.5 * inch]))
    story.append(Paragraph("Hardware: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB), FP32, imgsz 640.", SMALL))
    story.append(PageBreak())

    # Training curves
    story.append(Paragraph("Training Curves", H2))
    im = img(run / "results.png", max_w=6.8 * inch, max_h=6.5 * inch)
    if im:
        story.append(im)
    else:
        story.append(Paragraph("results.png missing", SMALL))
    story.append(PageBreak())

    # Confusion + PR
    vdir, tdir = vis_dir(seed, "val"), vis_dir(seed, "test")
    story.append(Paragraph("Confusion Matrix (Normalized) — Val", H2))
    story.extend(fig_row(vdir / "confusion_matrix_normalized.png", vdir / "confusion_matrix.png"))
    story.append(PageBreak())

    story.append(Paragraph("Precision-Recall / P / R / F1 Curves — Val", H2))
    story.extend(fig_row(vdir / "BoxPR_curve.png", vdir / "BoxP_curve.png"))
    story.append(Spacer(1, 4))
    story.extend(fig_row(vdir / "BoxR_curve.png", vdir / "BoxF1_curve.png"))
    story.append(PageBreak())

    story.append(Paragraph("Confusion + Curves — Test (visualization pass @ frozen OP conf)", H2))
    story.extend(fig_row(tdir / "confusion_matrix_normalized.png", tdir / "BoxPR_curve.png"))
    story.append(PageBreak())

    # Val GT / pred batches
    story.append(Paragraph("Validation Ground Truth (batch 0)", H2))
    story.extend(fig_row(vdir / "val_batch0_labels.jpg", max_w=6.8 * inch, max_h=6.8 * inch))
    story.append(PageBreak())
    story.append(Paragraph("Validation Predictions (batch 0)", H2))
    story.extend(fig_row(vdir / "val_batch0_pred.jpg", max_w=6.8 * inch, max_h=6.8 * inch))
    story.append(PageBreak())

    # Test GT / pred
    story.append(Paragraph(f"Test Ground Truth (batch 0) — seed {seed}", H2))
    story.extend(fig_row(tdir / "val_batch0_labels.jpg", max_w=6.8 * inch, max_h=6.8 * inch))
    story.append(PageBreak())
    story.append(Paragraph(f"Test Predictions (batch 0) @ conf {op['conf']} — seed {seed}", H2))
    story.extend(fig_row(tdir / "val_batch0_pred.jpg", max_w=6.8 * inch, max_h=6.8 * inch))
    story.append(PageBreak())
    story.append(Paragraph(f"Test Predictions (batch 1) @ conf {op['conf']} — seed {seed}", H2))
    story.extend(fig_row(tdir / "val_batch1_pred.jpg", max_w=6.8 * inch, max_h=6.8 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Training Augmentation Sample", H2))
    story.extend(fig_row(run / "train_batch0.jpg", run / "train_batch1.jpg"))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Official metrics come from paper_results/yolo/s_seed"
        f"{seed}/{{val,test}}.json (frozen protocol; OP conf chosen on val only). "
        "Test batch images above are from a visualization-only pass (make_test_vis.py); "
        "they do not change any reported number.", SMALL))

    doc.build(story)
    print("wrote", out)
    return out


def research_summary() -> Path:
    out = REPORTS / "research_summary.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, title="SafeMine v2 research summary")
    story: list = []
    story.append(Paragraph("SafeMine v2 — Landmine Detection Research Summary", H1))
    story.append(Paragraph(
        f"Generated {date.today().isoformat()} · AeroGuard · frozen v2 split · "
        "full provenance in RUN_LOG.md / DECISIONS.md", SMALL))
    story.append(Spacer(1, 8))

    # Headline numbers
    tests = [json.load(open(PR / "yolo" / f"s_seed{s}" / "test.json", encoding="utf-8")) for s in (0, 1, 2)]
    m50 = [t["ultralytics_map"]["test"]["map50"] for t in tests]
    import statistics as stats

    story.append(Paragraph("Headline results", H2))
    hl = [
        ["System", "Key metric", "Value"],
        ["YOLOv8s (3 seeds)", "test mAP50 (mean ± std)",
         f"{stats.mean(m50):.4f} ± {stats.stdev(m50):.4f}"],
        ["Two-stage arm C (seed 0)", "FP/img @ R≈0.96", "0.153"],
        ["Two-stage arm C (seed 0)", "precision", "0.894"],
        ["Two-stage C−B bootstrap", "Δprecision CI95", "[0.065, 0.186] (excl. 0)"],
        ["Grad-CAM saliency", "energy enrichment (YOLO)", "86× (vs MNv3 1.8×)"],
        ["Label audit", "objective test-box errors fixed", "2 (ds0_0801, ds0_0883)"],
    ]
    story.append(metric_table(hl, col_widths=[2.0 * inch, 2.2 * inch, 2.2 * inch]))
    story.append(PageBreak())

    # Tables as text pages
    for tname, title in (
        ("t1_dataset_leakage", "T1. Dataset & leakage audit"),
        ("t2_yolo_size", "T2. YOLO operating point (3 seeds)"),
        ("t3_image_level", "T3. Image-level comparison"),
        ("t4_two_stage", "T4. Two-stage ablation"),
        ("t5_gradcam", "T5. Grad-CAM experiments"),
        ("t6_failures", "T6. Failure analysis"),
    ):
        p = PR / "tables" / f"{tname}.md"
        if not p.exists():
            continue
        story.append(Paragraph(title, H2))
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("###"):
                continue
            story.append(Paragraph(line.replace("|", " · "), SMALL))
        story.append(Spacer(1, 6))

    story.append(PageBreak())
    # Figures
    figs = [
        (PR / "figures" / "pipeline.png", "Two-stage pipeline"),
        (PR / "figures" / "pr_curves.png", "PR / recall / FP curves (seed 0)"),
        (PR / "figures" / "fp_recall_tradeoff.png", "FP/img vs recall (plan §5.4)"),
        (PR / "figures" / "energy_in_box_hist.png", "Grad-CAM energy in GT boxes (E1)"),
        (PR / "figures" / "gradcam_example_grid.png", "Grad-CAM head-to-head (E6)"),
        (PR / "figures" / "failure_gallery_fp.png", "FP failure gallery (E4)"),
        (PR / "figures" / "e5_audit_test.png", "E5 zero-energy audit"),
        (PR / "figures" / "e3_paste.png", "E3 paste-hold counterfactual"),
        (PR / "figures" / "e3_remove.png", "E3 remove-disappear counterfactual"),
    ]
    for fp, title in figs:
        story.append(Paragraph(title, H2))
        im = img(fp, max_w=6.8 * inch, max_h=6.2 * inch)
        if im:
            story.append(im)
        else:
            story.append(Paragraph(f"missing {fp}", SMALL))
        story.append(PageBreak())

    # Per-source
    ps = PR / "yolo" / "s_seed0" / "per_source.md"
    if ps.exists():
        story.append(Paragraph("Per-source false positives (seed 0)", H2))
        for line in ps.read_text(encoding="utf-8").splitlines()[:40]:
            if line.strip():
                story.append(Paragraph(line.replace("|", " · "), SMALL))
        story.append(PageBreak())

    # Label fix
    lf = PR / "label_fix_report.json"
    if lf.exists():
        d = json.load(open(lf, encoding="utf-8"))
        c = d["arms"]["C_two_stage"]
        story.append(Paragraph("Label fixes (D023)", H2))
        story.append(Paragraph(
            f"Two objectively wrong test boxes corrected ({', '.join(x.split('_rf')[0] for x in d['fix_images'])}). "
            f"Arm C recall {c['recall']['before']:.4f} → {c['recall']['after']:.4f}; "
            f"image recall {c['img_recall']['before']:.4f} → {c['img_recall']['after']:.4f}; "
            f"FP/img unchanged at {c['fp_per_image']['after']:.3f}. "
            "See paper_results/label_fix_report.json for full numbers.", BODY))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Scope notes", H2))
    story.append(Paragraph(
        "Approved reduced budget: yolov8s ×3 seeds only (no n/m size rows); "
        "PC GPU only (Jetson/T7 edge phases dropped); "
        "no retrain on fixed train labels (none found). "
        "Dataset not in git — v2 split manifest sha256 in paper_results/split_v2/sha256.txt. "
        "Weights in models/ (see models/README.md).", BODY))

    doc.build(story)
    print("wrote", out)
    return out


def main() -> None:
    for s in (0, 1, 2):
        seed_report(s)
    research_summary()


if __name__ == "__main__":
    main()
