# DECISIONS.md — SafeMine research retrain (v2 plan)

Protocol: choose on val only; never tune on test. Log every decision and reason here.

---

## 2026-10-14 — Phase 0

| ID | Decision | Reason |
|---|---|---|
| D001 | Work under `SafeMineAI-main/paper_results/` only; never overwrite original dataset, runs, or Week 2 scripts. | Plan §0.3. Protects v1 evidence. |
| D002 | Original run location is `runs/detect/runs/detect/landmine_yolov8m_75ep/` (not `Weights/runs/...` as in the PDF text). | Verified on disk 2026-10-14. |
| D003 | Dataset root is `C:/Users/Delta/Downloads/Landmine_project/Datasets/landmine_combined`; validation folder name is `valid` (not `val`). `data.yaml` already maps `val: valid/images`. | Plan assumed `val`; folder is `valid`. |
| D004 | Backup verified by SHA-256: best.pt, last.pt, best.onnx, args.yaml, results.csv, safemine_summary.json → `paper_results/original_run/` + `backup_verify.json`. | Plan Phase 0. |
| D005 | `jetson/gradcam.py` does not exist. Phase 6 will implement Grad-CAM from scratch after architecture check. | User instruction 2026-10-14. |
| D006 | `two_stage_eval.py` / `test_two_stage.py` not present on this machine; implement in Phase 5 with tests. | User instruction 2026-10-14. |
| D007 | Fair-comparison confidence thresholds chosen on **val only**; applied frozen to test. | Non-negotiable rule. |
| D008 | May/Jan `*_lwir_*` files left in place, not renamed. Audit will classify thermal vs RGB vs derived and flag for paper narrative. | User instruction 2026-10-14. |
| D009 | Pin Ultralytics 8.4.117 / torch 2.6.0+cu124 as in original run. | Plan §3.9. |

---

## Open (awaiting Phase 1 gate)

- Whether current 5723/455/221 split leaks (near-dup / augment-before-split).
- Whether test has enough negatives for image-level / FP-per-image metrics (prelim: 0 empty labels in test).
- RGB-only paper narrative vs presence of LWIR-named sources.
