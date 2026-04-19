"""Make src/ and the repo root importable for all tests without an editable install."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

for p in (SRC, ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
