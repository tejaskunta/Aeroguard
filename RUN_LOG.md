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

## Notes

- Original v1 training (pre-this-session): `landmine_yolov8m_75ep`, finished 2026-08-15, 15h 16m 42s, conf 0.25 re-validation (see `paper_results/original_run/safemine_summary.json`).
