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

## 2026-10-14 — Phase 1 verdict

| ID | Decision | Reason |
|---|---|---|
| D010 | **FAIL the current split.** Do not train Phase 3 models on `landmine_combined` as-is. | 76 cross-split base-name groups; montage shows same scene/re-export; 63.8% of val+test have pHash≤2 train neighbour; augment-before-split confirmed in Week 2 scripts. |
| D011 | Image-level / fair FP-per-image metrics **blocked** until negatives exist in test (currently 0). | Phase 1 gate ASK rule. |
| D012 | May/Jan files are **colour RGB with marker pins**, not greyscale thermal, despite `lwir` in the name. Keep files; disclose marker-shortcut risk; decide in Phase 2 whether to exclude/relabel. | Manual montage + saturation stats. |
| D013 | Phase 2 must be **group-aware** (connected components on base-key + near-dup + scene), not a random re-shuffle. | Plan §Phase 2; evidence above. |

## 2026-10-14 — Phase 2 freeze (v2 split)

| ID | Decision | Reason |
|---|---|---|
| D014 | **Leakage definition (operational):** a val/test image leaks if any train image shares (a) exact MD5, (b) Roboflow base key, (c) same source AND pHash Hamming≤2 under 8 dihedral AND 32×32 RGB MSE≤900. All candidates were manually reviewed on montages (`audit_v2_final/montages/manual_review_*.jpg`); same-scene pairs occurred at MSE≤817, terrain-similarity FPs at MSE≥958, gap used as the cut. | Raw pHash alone over-fires on repetitive grass/dirt (99 residual pairs, all MSE>900, are texture FPs). Manual review of every ham≤2 candidate is what the plan's ≤0.5% rule requires. |
| D015 | **Freeze `landmine_combined_v2`** as the Phase 3–6 dataset: train 3907 / valid 181 / test 189; 11550 / 419 / 253 instances; 20 / 7 / 5 negatives; 0 exact cross-split dups; 0 base-key collisions; 0 true near-dup leaks (0/370 = 0% ≤ 0.5% rule). Manifest sha256 `748322d62ace769445d0bd23636990970c7a5115a99c9344e139099c608c80de`. | 30 eval images removed as true leaks (13 at MSE≤100, 17 more at MSE≤900 incl. `ds0_0043`↔`ds1_0043` cross-folder duplicate). Final audit: `paper_results/audit_v2_final/`. |
| D016 | May/Jan sources **kept** in all splits (D012 resolved as keep+disclose): colour RGB with marker pins, marker-shortcut risk disclosed in paper; not used for thermal claims. | User instruction 2026-10-14; no evidence of a better option within budget. |
