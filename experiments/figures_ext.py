"""Figures for the dynamic (Lever 1) and multi-dataset (Lever 2) extensions."""
import os, sys
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mpl.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
FIG = str(FIG_DIR)


def fig_landmark():
    d = pd.read_csv(RESULTS_DIR / "dynamic_landmark.csv")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = d["landmark_y"].values
    ax.plot(x, d["C_baseline"], "o--", color="#7f8c8d", lw=2, ms=8, label="Baseline measurement")
    ax.plot(x, d["C_dynamic"], "o-", color="#27ae60", lw=2, ms=8, label="Most recent measurement")
    for xi, b, dy in zip(x, d["C_baseline"], d["C_dynamic"]):
        ax.annotate(f"+{dy-b:.2f}", (xi, dy), textcoords="offset points",
                    xytext=(0, 8), ha="center", color="#27ae60", fontsize=10)
    ax.set_xticks(x); ax.set_xlabel("Landmark time (years)")
    ax.set_ylabel("3-year death-risk concordance (C)")
    ax.set_title("Updating biomarkers restores discrimination as baseline goes stale")
    ax.set_ylim(0.6, 0.92); ax.legend(frameon=False)
    fig.savefig(f"{FIG}/fig_landmark.png"); plt.close(fig)
    print("fig_landmark")


def fig_multidataset():
    d = pd.read_csv(RESULTS_DIR / "multidataset.csv")
    datasets = list(d["dataset"].unique())
    models = ["Cox PH", "RSF", "Gradient Boost", "DeepSurv"]
    colors = {"Cox PH": "#c0392b", "RSF": "#2980b9",
              "Gradient Boost": "#8e44ad", "DeepSurv": "#16a085"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(datasets)); w = 0.2
    for i, m in enumerate(models):
        vals = [d[(d.dataset == ds) & (d.model == m)]["c_index"].values[0]
                if len(d[(d.dataset == ds) & (d.model == m)]) else np.nan
                for ds in datasets]
        ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=colors[m])
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("Concordance index"); ax.set_ylim(0.55, 0.75)
    ax.set_title("Across clinical datasets, complex models barely exceed Cox")
    ax.legend(frameon=False, ncol=4, loc="upper center", fontsize=9)
    fig.savefig(f"{FIG}/fig_multidataset.png"); plt.close(fig)
    print("fig_multidataset")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "landmark"):
        fig_landmark()
    if which in ("all", "multi") and os.path.exists(RESULTS_DIR / "multidataset.csv"):
        fig_multidataset()
    print("done ->", FIG)
