import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "model", ROOT / "webapp"):
    sys.path.insert(0, str(p))
