#!/usr/bin/env python3
"""Repository script entry point for collaborative reports."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.collaborative_report.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
