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

## 2026-10-14 — Phase 3 result + protocol deviation

| ID | Decision | Reason |
|---|---|---|
| D017 | **Protocol deviation logged:** `week3_train_yolov8.py` step 3/5 auto-validates **both val and test** at conf 0.25 (v1 pipeline behavior, no CLI flag to skip). Test metrics were produced at train end without any decision being made from them. Mitigation: no threshold, model, or checkpoint choice will reference these numbers; the frozen-protocol test evaluation (thresholds chosen on val only) is done once in Phase 5 and reported as the official test result. The conf-0.25 auto-numbers are disclosed as supplementary. | RUN_LOG honesty rule; plan's "test evaluated once" applies to protocol-governing evaluations. |
| D018 | Phase 3 run `landmine_yolov8s_75ep_v2`: EarlyStopping at **67/75** (best epoch 47, patience 20), 50 min on RTX 4070; val mAP50 0.9367 / mAP50-95 0.6012. Accept early stop as final (no retrain with patience=0 within budget — reserve decision if Phase 5 gaps appear). | Ultralytics default patience; budget §25 h; results.csv + log archived in run dir. |
| D019 | **Retrain yolov8s seed 0 at batch 8** as `landmine_yolov8s_75ep_s0_v2` (plan §Phase 3.1: reuse original args.yaml settings except model/seed/name/data — original batch was 8, script default 16). The batch-16 run `landmine_yolov8s_75ep_v2` is kept as a settings-pilot, not a paper result. | Plan ground rule for old-vs-new comparability; cost ~1.5 h inside budget. |
## 2026-10-15 — Phase 4 baseline findings + Phase 5 protocol

| ID | Decision | Reason |
|---|---|---|
| D020 | **Frozen-feature MNv3 baseline (Stage A) is degenerate on negatives across all resolutions (224/448/640) and batch sizes (32/64)**: recall-first thresholds give test balanced accuracy ~0.48-0.50, kappa ~0 (neg scores 0.74-0.99 overlap positives; loss converges, so not a harness bug). Stage B fine-tuning (unfreeze features[13:], 2.3M params) is the only variant that separates negatives (AB_224 seed0: test bal 0.597, kappa 0.275). Kept all A-rows as the plan-mandated stage-B ablation; AB_224 is the primary MNv3 baseline row for seeds 0/1/2. Extreme train imbalance (3887:20) with per-sample weights [97.7, 0.50] retained as per plan (no rebalancing beyond class weights — fairness lever is stage B + resolution, not sampling). | Plan §Phase 5.2 requires Stage A then Stage B; resolution/batch/batch-size rows are the plan ablations; no recipe handicap. |
| D021 | **Phase 5.3 threshold rule operationalised** (plan §Phase 5.3): sweep P in {0.4,0.6,0.8,none}; for each P pick the largest classifier tau whose val box-recall is within 0.5 pts of stage-1 recall at the proposal conf; among those pick the pair with lowest val FP/image (tie: higher P). Arm A = YOLO alone at the Phase-3 deployment OP conf (primary_op_conf from phase3 val.json); Arm B = YOLO alone at the conf matching the two-stage val recall (ts.yolo_only_at_recall); Arm C = two-stage. All three applied to test ONCE after the val choice is frozen; paired bootstrap C-B. | Plan text; tau/P both chosen on val only. |
| D022 | **E5 flags images by Grad-CAM energy in the UNION of all GT boxes (one row per image), not per detection.** First implementation scored every det-GT pair; on multi-mine images (May source: up to 17 gt/img) only the salient instance receives CAM energy, so 90% of May rows read exactly 0 and the lowest-10% flag became an instance-count artefact. Union metric answers the plan question directly: does this image's CAM look at ANY mine? Test threshold 0.800 (19/184 flagged), train 0.721 (389/3887). | Plan E5 says flag *images* in the lowest 10% of energy-in-box; per-instance rows made the flag unusable. Logged in RUN_LOG with the fix history. |
| D023 | **Two objectively-wrong test boxes fixed and reported with/without (no retrain).** ds0_0801 (box on tree trunk) and ds0_0883 (box on fence) removed from GT per LABEL_FIXES.csv; label_fix_report.json gives before/after per arm (arm C recall 0.9644->0.9721, img recall 0.989->1.000, fp/img unchanged 0.153). Baseline arm numbers asserted to reproduce the frozen test_report before applying the fix. The optional retrain on fixed TRAIN labels is not run (no train-label fixes found in E5 review; budget scope: yolov8s-only, phases 7/8 dropped). | Plan: test labels only if objectively wrong, numbers with and without; retrain optional. |
