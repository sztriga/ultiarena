from __future__ import annotations

import sys
from pathlib import Path


# Allow `import trickster` when running tests without installing the package.
# Snapszer keeps the `trickster` package name and lives isolated under snapszer/.
SRC = Path(__file__).resolve().parents[2] / "snapszer"
sys.path.insert(0, str(SRC))

