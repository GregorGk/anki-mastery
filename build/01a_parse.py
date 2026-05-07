#!/usr/bin/env python
"""Stage 1a entry point — thin shim around build/stage_1a.py.

Filename starts with a digit so it sorts naturally next to 01b/01c, but Python
import doesn't allow that. The real implementation lives in build.stage_1a.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build.stage_1a import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
