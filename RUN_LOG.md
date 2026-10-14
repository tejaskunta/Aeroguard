# RUN_LOG.md — SafeMine research retrain (v2)

Every run and every test-set evaluation is logged here. Test evaluations before the frozen protocol are bugs.

| Timestamp (local) | What | Command / notes | Test set touched? |
|---|---|---|---|
| 2026-10-14 | Phase 0 inventory + backup | PowerShell copy of original run → `paper_results/original_run/`; SHA-256 verified | No |
| 2026-10-14 | Phase 1 audit start | `scripts/phase1_audit.py` (CPU only; no training, no weight load for inference) | Read-only on test images for leakage audit (standard Phase 1) |
| 2026-10-14 | Phase 2 re-split v2 | `scripts/phase2_resplit_full.py` (base_key + session + same-source pHash ham≤0 graph → 1048 groups; 200/200 eval) | Labels read for counts only |
| 2026-10-14 | Phase 2 second-method filter | `scripts/phase2_second_filter.py --apply --mse 900` after manual montage review → 30 true-leak eval images removed | Removed 13+17 leak images from valid/test; no metrics computed |
| 2026-10-14 | Phase 2 final audit | `scripts/phase1_audit.py --dataset landmine_combined_v2 --out paper_results/audit_v2_final` → PASS (0 true leaks, 0 exact/base-key dups) | Read-only leakage audit |
| 2026-10-14 | **v2 frozen** | manifest sha256 `748322d6…c80de`; DECISIONS D014–D016 | No |
| 2026-10-14 21:27 | **Phase 3 train start** | `week3_train_yolov8.py --data …/landmine_combined_v2/data.yaml --model yolov8s --epochs 75 --seed 0 --save-period 5 --batch 16 --name landmine_yolov8s_75ep_v2` → `logs/phase3_train.log` (PID 3632) | No (train/val only) |
| 2026-10-14 22:17 | Phase 3 train end | EarlyStopping at **67/75** (best epoch 47, patience 20); 50 min; checkpoints every 5 ep; val (best.pt, conf 0.25): mAP50 0.9367 / mAP50-95 0.6012, P 0.9322, R 0.9189 | **Deviation:** script step 3/5 auto-ran test at conf 0.25 → test mAP50 0.9599 / mAP50-95 0.6651 written to `safemine_summary.json`. Pre-existing v1 pipeline behavior; no threshold/model decision made from it. Logged as D017; official frozen-protocol test eval happens once in Phase 5. |
| 2026-10-14 ~22:30 | Phase 3 compliant rerun start | `week3_train_yolov8.py --batch 8 --name landmine_yolov8s_75ep_s0_v2 --save-period 5 --seed 0` → `logs/phase3_s0_batch8.log` (PID 33148). Batch 8 matches original args.yaml (D019); pilot run kept. | No (train/val only) |

## Notes

- Original v1 training (pre-this-session): `landmine_yolov8m_75ep`, finished 2026-08-15, 15h 16m 42s, conf 0.25 re-validation (see `paper_results/original_run/safemine_summary.json`).
