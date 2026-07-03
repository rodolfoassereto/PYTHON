# paths.py — import this FIRST in any TVW1 script to wire sys.path.
#
# It locates the repository from this file's own location (no machine-specific paths) and prepends the project's importable directories.
# Replaces every old `sys.path.insert(0, r"C:\Users\...\My Drive\PYHTON\...")`.
#
# Usage at the top of a script that lives anywhere under the repo:
#
#     import sys
#     from pathlib import Path
# sys.path.insert(0, str(next(p for p in [Path.cwd(), *Path.cwd().parents]
#                             if (p / "paths_rodolfoassereto.py").is_file())))
#     import paths  # noqa: F401  — wires Libraries + core + models + data
#
import sys
from pathlib import Path

_CURRENT = Path(__file__).resolve().parent      # the folder where paths_rodolfoassereto.py is located
_ROOT = _CURRENT.parent                         # its parent folder

_DIRS = [
    _ROOT / "Libraries",
    _CURRENT
    ]

# _DIRS = [
#     _ROOT / "Libraries",   # Note: _ROOT is now a Path object, and the operator / is overloaded for this class to mean "join paths"
#     _CURRENT / "core",        # TVW1-specific shared operators / forward / metrics / graph_DR_auxiliary_functions
#     _CURRENT / "data",      # the variational models
#     _CURRENT / "TVW1_naif",        # test-data generators
# ]

# Add all folders in _DIRS
for _d in _DIRS:
    _s = str(_d)
    if _d.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)

# # Also add all subfolders
# for newpath in _CURRENT.rglob("*"): # rglob is a Path method that returns all subobject's paths (as a Path object, not as a string)
#     if newpath.is_dir() and str(newpath) not in sys.path:
#         sys.path.append(str(newpath))
