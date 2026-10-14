# RUN_LOG.md — SafeMine research retrain (v2)

Every run and every test-set evaluation is logged here. Test evaluations before the frozen protocol are bugs.

| Timestamp (local) | What | Command / notes | Test set touched? |
|---|---|---|---|
| 2026-10-14 | Phase 0 inventory + backup | PowerShell copy of original run → `paper_results/original_run/`; SHA-256 verified | No |
| 2026-10-14 | Phase 1 audit start | `scripts/phase1_audit.py` (CPU only; no training, no weight load for inference) | Read-only on test images for leakage audit (standard Phase 1) |

## Notes

- Original v1 training (pre-this-session): `landmine_yolov8m_75ep`, finished 2026-08-15, 15h 16m 42s, conf 0.25 re-validation (see `paper_results/original_run/safemine_summary.json`).
