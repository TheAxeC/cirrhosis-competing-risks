"""Generate the graphical abstract submitted with the paper.

Produces figures/graphical_abstract.png (8 x 3.2 in at 200 dpi, cropped tight, about 1517x619 px,
~5:2, readable at 13x5 cm). Schematic; the transplant F1 shown is the range (minimum to
maximum) of the transplant-class F1 in results/classification.csv, read at run time. Edit freely or replace with a
designed version before submission.
"""
from __future__ import annotations
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import os
import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Default style, so the output is the same whether run alone or after the figure drivers in paper.py.
matplotlib.rcdefaults()
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(FIG_DIR, "graphical_abstract.png")   # FIG_DIR from core.paths (-> figures/)

# Range of the transplant-class (CL) F1 over the three-class classifiers of run_classification.
_F1_CL = pd.read_csv(RESULTS_DIR / "classification.csv")["f1_CL"]
MIN_CL_F1, BEST_CL_F1 = _F1_CL.min(), _F1_CL.max()

BLUE, RED, GREY, GREEN = "#3b6ea5", "#b2462f", "#5a5a5a", "#2e7d4f"

fig, ax = plt.subplots(figsize=(8.0, 3.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

ax.text(50, 38, "From classification to competing risks on the Mayo PBC benchmark (n=418)",
        ha="center", va="center", fontsize=11, weight="bold")

# Left: conventional framing
ax.add_patch(FancyBboxPatch((3, 8), 38, 25, boxstyle="round,pad=0.6,rounding_size=1.5",
                            fc="#f2f2f2", ec=GREY, lw=1.3))
ax.text(22, 30, "Conventional framing", ha="center", fontsize=10.5, weight="bold", color=GREY)
ax.text(22, 26, "status label; tree ensembles tested", ha="center", fontsize=9.5)
ax.text(22, 22, "alive  /  transplant  /  death", ha="center", fontsize=9)
ax.text(22, 16.5, "no time-resolved risk of either event;\nsimulated RF ranks true risk at 0.56",
        ha="center", fontsize=8.5, color=GREY)
ax.text(22, 11, f"transplant F1 {MIN_CL_F1:.2f} to {BEST_CL_F1:.2f} (our CV)", ha="center", fontsize=9.5,
        weight="bold", color=RED)

# Arrow
ax.add_patch(FancyArrowPatch((42, 20.5), (57, 20.5), arrowstyle="-|>", mutation_scale=22,
                             lw=2.2, color="#222222"))
ax.text(49.5, 23, "reframe", ha="center", fontsize=9, style="italic")

# Right: competing-risks framing
ax.add_patch(FancyBboxPatch((58, 8), 39, 25, boxstyle="round,pad=0.6,rounding_size=1.5",
                            fc="#eaf1f8", ec=BLUE, lw=1.5))
ax.text(77.5, 30, "Competing-risks survival", ha="center", fontsize=10.5, weight="bold", color=BLUE)
ax.text(77.5, 25.8, "death and transplant as rival events", ha="center", fontsize=9)
ax.text(77.5, 21.5, "transplant C = 0.82   [0.80, 0.84]", ha="center", fontsize=9.5,
        weight="bold", color=GREEN)
ax.text(77.5, 17.5, "ceiling set by Mayo score (C = 0.833)", ha="center", fontsize=9)
ax.text(77.5, 13.0, "opposite age effect:\ndeath HR 1.36  /  transplant HR 0.74",
        ha="center", fontsize=8.7, color=BLUE)

ax.text(50, 4.0, "Competing-risks models recover cause-specific risk; "
                 "status-label F1 does not measure it.", ha="center", fontsize=8.8, style="italic", color="#333333")

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
