"""Path anchors for the cirrhosis competing-risks pipeline.

Resolves data / results / figure directories from this file's location, so the
pipeline runs from anywhere (run drivers as `python -m experiments.<name>` from the
repository root, the directory holding `experiments/` and `core/`). Inputs are read
from `data/`; everything the pipeline generates is written under the repository root
(`results/` for the CSVs, `figures/` for the PNGs). These are regenerable, so the public code
deposit excludes them; the research hub keeps the committed copies as the record.
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]      # core/ -> repository root
DATA_DIR = _ROOT / "data"                         # committed public survival CSVs
RESULTS_DIR = _ROOT / "results"                   # generated CSVs (regenerable; not shipped in the public deposit)
FIG_DIR = _ROOT / "figures"                       # generated figures (regenerable; not shipped in the public deposit)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
