#!/usr/bin/env python
"""Stage 1.5 entry point — thin shim around build/stage_15.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.stage_15 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
