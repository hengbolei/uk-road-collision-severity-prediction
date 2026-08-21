"""Configure test imports so the local ``road_severity`` package is exercised."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
