# Manuscript

Single-column preprint, built with `pdflatex` + BibTeX.

```bash
cd paper
make            # -> main.pdf
make clean
```

Requires a TeX distribution with `booktabs`, `natbib`, `float`, `microtype`
(MacTeX or TeX Live both work). No Python, GPU, or dataset needed.

## Layout

| Path | Contents |
|---|---|
| `main.tex` | preamble, `\input`s, headline-number macros |
| `sections/` | one file per section, in reading order |
| `refs.bib` | bibliography (BibTeX, `plainnat`) |
| `figs/` | copies of `paper_results/figures/*.png` |
| `Makefile` | `pdflatex → bibtex → pdflatex ×2` |

Figures are **copies**, not symlinks, so the directory builds standalone. To
refresh them after regenerating `paper_results/figures/`:

```bash
cp ../paper_results/figures/{pipeline,pr_curves,fp_recall_tradeoff,energy_in_box_hist,gradcam_example_grid,failure_gallery_fp,e3_paste,e3_remove}.png figs/
```

## Before submitting

1. Replace `\author{...}` in `main.tex` with the real author block.
2. Resolve the `\todo{...}` in `sections/availability.tex` with the Zenodo DOI
   and release tag.
3. Update `CITATION.cff` at the repository root to match.
4. Re-run `make` and confirm zero warnings:

   ```bash
   make 2>&1 | grep -E "Overfull|^!|undefined"
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
