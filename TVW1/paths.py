# paths.py — import this FIRST in any TVW1 script to wire sys.path.
#
# It locates the repository from this file's own location (no hardcoded,
# machine-specific paths) and prepends the project's importable directories.
# Replaces every old `sys.path.insert(0, r"C:\Users\...\My Drive\PYHTON\...")`.
#
# Usage at the top of a script that lives anywhere under the repo:
#
#     import sys
#     from pathlib import Path
#     sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents]
#                                 if (p / "TVW1").is_dir()) / "TVW1"))
#     import paths  # noqa: F401  — wires Libraries + core + models + data
#
import sys
from pathlib import Path

_TVW1 = Path(__file__).resolve().parent      # .../PYTHON/TVW1
_ROOT = _TVW1.parent                         # .../PYTHON

_DIRS = [
    _ROOT / "Libraries",   # general-purpose, cross-project toolbox (untouched)
    _TVW1 / "core",        # TVW1-specific shared operators / forward / metrics / graphs
    _TVW1 / "models",      # the variational models
    _TVW1 / "data",        # test-data generators
]

for _d in _DIRS:
    _s = str(_d)
    if _d.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)
