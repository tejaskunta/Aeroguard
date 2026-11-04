"""Make the flat ``src/`` modules importable when running pytest from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for sub in ("src/common", "src/evaluation", "src/gradcam", "src/pipeline",
            "src/training", "src/baseline", "src/analysis", "src/reporting"):
    p = ROOT / sub
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))