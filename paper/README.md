# Manuscript

Two-column IEEE conference format (`IEEEtran`, `\documentclass[conference]`),
built with `pdflatex` + BibTeX. Current build: 16 pages, zero errors, zero
undefined references, zero overfull boxes.

```bash
cd paper
make            # -> main.pdf
make clean
```

Requires a TeX distribution providing `IEEEtran.cls` and `IEEEtranN.bst`, plus
`booktabs`, `natbib`, `float`, `microtype` (MacTeX, TeX Live, and MiKTeX all
work). No Python, GPU, or dataset needed.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | preamble, `\input`s, headline-number macros |
| `sections/` | one file per section, in reading order |
| `refs.bib` | bibliography (BibTeX, `IEEEtranN` style + `natbib`) |
| `figs/` | copies of `paper_results/figures/*.png` |
| `Makefile` | `pdflatex → bibtex → pdflatex ×2` |

Figures are **copies**, not symlinks, so the directory builds standalone. To
refresh them after regenerating `paper_results/figures/`:

```bash
cp ../paper_results/figures/{pipeline,pr_curves,fp_recall_tradeoff,energy_in_box_hist,gradcam_example_grid,failure_gallery_fp,e3_paste,e3_remove}.png figs/
```

## Format notes

IEEEtran's column is 3.4 in wide, narrower than the old single-column layout.
Wide tables therefore use the full-width `table*` float, and the reproduction
command block uses `figure*`. `hyperref` is loaded with `hypertexnames=false` to
avoid duplicate pdfTeX destination anchors from the many `table*` floats. If you
add a table with more than ~4 columns, check the log for `Overfull \hbox`.

## Before submitting

1. Replace `\author{...}` in `main.tex` with the real author block (currently
   `Anonymous Authors` / "Preprint. Under review.").
2. `sections/availability.tex` says no DOI has been assigned yet. Add the Zenodo
   DOI and release tag once one exists.
3. Update `CITATION.cff` at the repository root to match (the `preferred-citation`
   entry and the `software` entry both still carry `"TODO: author list"`).
4. Re-run `make` and confirm zero warnings:

   ```bash
   make 2>&1 | grep -E "Overfull|^!|undefined|same identifier"
   ```

## Number traceability

Every quantitative claim traces to a file under `../paper_results/`. The mapping:

| Claim | Source |
|---|---|
| Split sizes, leakage audit | `audit_v2_final/audit_summary.json`, `split_v2/sha256.txt` |
| Detector size sweep, per-seed OP | `yolo/{s,n}_seed*/{val,test,size_latency}.json` |
| Two-stage arms, bootstrap CIs | `twostage/s0`, `s1_aligned`, `s2_aligned`/`test_report.json` |
| Threshold policy, max-F2 comparison, leave-one-source-out | `analysis/stats.{json,md}` |
| Stage-2 ablations | `ablation/`, `analysis/stats.md` |
| Image-level baseline | `baseline_mnv3/*/metrics.json` |
| Leaky-vs-clean | `leaky_v1/comparison.md`, `leaky_v1/*.json` |
| Grad-CAM E1–E6 | `gradcam/e{1,2,3,5,6}_summary.json` |
| Failure categories | `gradcam/e4_failures.csv`, `tables/t6_failures.md` |
| Label fix delta | `label_fix_report.json` |

The generated table fragments in `../paper_results/tables/*.tex` are **not**
included by the manuscript: the paper's tables are hand-set with `booktabs` for
readability, using the same numbers. If a number changes, change it in both
places or regenerate and diff.
