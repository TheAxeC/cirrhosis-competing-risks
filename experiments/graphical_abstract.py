"""Generate the Journal of Biomedical Informatics graphical abstract (mandatory).

Produces figures/graphical_abstract.png at Elsevier-compliant size (1600x640 px,
~5:2, readable at 13x5 cm). Schematic only; no data dependency. Edit freely or
replace with a designed version before submission.
"""
from __future__ import annotations
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(FIG_DIR, "graphical_abstract.png")   # FIG_DIR from core.paths (-> figures/)

BLUE, RED, GREY, GREEN = "#3b6ea5", "#b2462f", "#5a5a5a", "#2e7d4f"

fig, ax = plt.subplots(figsize=(8.0, 3.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

ax.text(50, 38, "From classification to competing risks on the Mayo PBC benchmark (n=418)",
        ha="center", va="center", fontsize=11, weight="bold")

# Left: conventional framing
ax.add_patch(FancyBboxPatch((3, 8), 38, 25, boxstyle="round,pad=0.6,rounding_size=1.5",
                            fc="#f2f2f2", ec=GREY, lw=1.3))
ax.text(22, 30, "Conventional framing", ha="center", fontsize=10.5, weight="bold", color=GREY)
ax.text(22, 26, "3-class classification", ha="center", fontsize=9.5)
ax.text(22, 22, "alive  /  transplant  /  death", ha="center", fontsize=9)
ax.text(22, 16.5, "discards follow-up time;\ntransplant treated as static label",
        ha="center", fontsize=8.5, color=GREY)
ax.text(22, 11, "transplant F1 $\\approx$ 0.21  (fails)", ha="center", fontsize=9.5,
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
ax.text(77.5, 17.5, "no ML model beats Mayo (C = 0.833)", ha="center", fontsize=9)
ax.text(77.5, 13.0, "opposite age effect:\ndeath HR 1.36  /  transplant HR 0.74",
        ha="center", fontsize=8.7, color=BLUE)

ax.text(50, 4.0, "Correct problem formulation, not model complexity, yields calibrated risk "
                 "for both outcomes.", ha="center", fontsize=8.8, style="italic", color="#333333")

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
