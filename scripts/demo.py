#!/usr/bin/env python
"""Voice computer-use demo. See docs/DEMO.md, or run with --help.

    .venv/bin/python scripts/demo.py                      # microphone, dry-run
    .venv/bin/python scripts/demo.py --live               # microphone, really acts
    .venv/bin/python scripts/demo.py --text "open notes"  # text replay
    .venv/bin/python scripts/demo.py --say "open notes"   # say -o audio through whisper.cpp
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_local.demo import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
