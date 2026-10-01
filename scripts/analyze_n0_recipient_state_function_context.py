from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.n0_recipient_state_function_context import analyze_campaign  # noqa: E402


if __name__ == "__main__":
    analyze_campaign()
