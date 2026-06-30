"""Publication figures for the cirrhosis competing-risks survival re-analysis."""
import os, sys, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from lifelines import AalenJohansenFitter, KaplanMeierFitter
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis

from core.data_prep import load_clean, make_preprocessor, NUM_COLS, CAT_COLS
from core.eval_utils import survival_at_times

warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
YEAR = 365.25
FIGDIR = str(FIG_DIR)
C_DEATH, C_TX, C_MODEL = "#c0392b", "#2980b9", "#27ae60"


def fig_cif(df):
    """Cumulative incidence: death vs transplant (the competing-risks picture)."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for cause, code, col, lbl in [("Death", 1, C_DEATH, "Death"),
                                  ("Transplant", 2, C_TX, "Liver transplant")]:
        ajf = AalenJohansenFitter(calculate_variance=True)
        ajf.fit(df["time"], df["event_cr"], event_of_interest=code)
        cif = ajf.cumulative_density_.iloc[:, 0]
        ci = ajf.confidence_interval_
        t = cif.index / YEAR
        ax.step(t, cif.values, where="post", color=col, lw=2, label=lbl)
        ax.fill_between(t, ci.iloc[:, 0], ci.iloc[:, 1], step="post",
                        color=col, alpha=0.15)
    ax.set_xlabel("Years from registration"); ax.set_ylabel("Cumulative incidence")
    ax.set_title("Cumulative incidence of competing outcomes")
    ax.set_xlim(0, 12); ax.set_ylim(0, 0.5); ax.legend(frameon=False)
    fig.savefig(f"{FIGDIR}/fig_cif.png"); plt.close(fig)


def fig_forest():
    """Forest plot of cause-specific hazard ratios (death vs transplant)."""
    cs = pd.read_csv(RESULTS_DIR / "cause_specific_cox.csv")
    pretty = {"Age_years": "Age", "log_Bilirubin": "Bilirubin (log)",
              "Albumin": "Albumin", "log_Prothrombin": "Prothrombin (log)",
              "log_Copper": "Copper (log)", "log_SGOT": "SGOT (log)",
              "Platelets": "Platelets", "Ascites": "Ascites", "Edema": "Edema",
              "Hepatomegaly": "Hepatomegaly", "Spiders": "Spiders", "Sex": "Sex (F)",
              "Stage4": "Stage 4"}
    death = cs[cs.cause == "death"].copy()
    order = death.reindex(death["HR"].sort_values().index)["variable"].tolist()
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for i, var in enumerate(order):
        for cause, col, off in [("death", C_DEATH, -0.16), ("transplant", C_TX, 0.16)]:
            r = cs[(cs.cause == cause) & (cs.variable == var)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            ax.plot([r.HR_lo, r.HR_hi], [i + off]*2, color=col, lw=1.5)
            ax.plot(r.HR, i + off, "o", color=col, ms=6,
                    label=cause.capitalize() if i == 0 else None)
    ax.axvline(1, color="k", ls="--", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([pretty.get(v, v) for v in order])
    ax.set_xscale("log"); ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_title("Cause-specific hazard ratios: death vs. transplant")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(f"{FIGDIR}/fig_forest.png"); plt.close(fig)


def oof_risk(df):
    """Out-of-fold Cox risk + 5-year survival (for stratification & calibration)."""
    X = df[NUM_COLS + CAT_COLS].copy()
    y = Surv.from_arrays(df["event_death"].values, df["time"].values)
    strat = df["event_cr"].values
    risk = np.zeros(len(df)); s5 = np.zeros(len(df))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, strat):
        pre = make_preprocessor()
        Ztr = np.asarray(pre.fit_transform(X.iloc[tr]), float)
        Zte = np.asarray(pre.transform(X.iloc[te]), float)
        m = CoxPHSurvivalAnalysis(alpha=1.0).fit(Ztr, y[tr])
        risk[te] = m.predict(Zte)
        s5[te] = survival_at_times(m, Zte, [5 * YEAR]).ravel()
    return risk, s5


def fig_stratification(df, risk):
    """KM by predicted-risk tertile: does the model separate real risk groups?"""
    grp = pd.qcut(risk, 3, labels=["Low", "Intermediate", "High"])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    cols = {"Low": C_MODEL, "Intermediate": "#e67e22", "High": C_DEATH}
    for g in ["Low", "Intermediate", "High"]:
        m = grp == g
        kmf = KaplanMeierFitter().fit(df["time"][m] / YEAR, df["event_death"][m],
                                      label=f"{g} risk (n={m.sum()})")
        kmf.plot_survival_function(ax=ax, color=cols[g], ci_show=True, lw=2)
    ax.set_xlabel("Years"); ax.set_ylabel("Survival probability")
    ax.set_title("Survival by predicted-risk tertile (out-of-fold)")
    ax.set_xlim(0, 12); ax.set_ylim(0, 1); ax.legend(frameon=False)
    fig.savefig(f"{FIGDIR}/fig_stratification.png"); plt.close(fig)


def fig_calibration(df, s5):
    """5-year calibration: predicted survival vs KM-observed by decile bins."""
    pred = s5
    bins = pd.qcut(pred, 5, labels=False, duplicates="drop")
    xs, ys, lo, hi = [], [], [], []
    for b in np.unique(bins):
        m = bins == b
        kmf = KaplanMeierFitter().fit(df["time"][m], df["event_death"][m])
        obs = float(kmf.predict(5 * YEAR))
        ci = kmf.confidence_interval_survival_function_
        idx = np.searchsorted(ci.index.values, 5 * YEAR) - 1
        idx = max(0, min(idx, len(ci) - 1))
        xs.append(pred[m].mean()); ys.append(obs)
        lo.append(ci.iloc[idx, 0]); hi.append(ci.iloc[idx, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.errorbar(xs, ys, yerr=[np.array(ys)-np.array(lo), np.array(hi)-np.array(ys)],
                fmt="o-", color=C_MODEL, capsize=3, label="Cox model")
    ax.set_xlabel("Predicted 5-year survival"); ax.set_ylabel("Observed (Kaplan-Meier)")
    ax.set_title("5-year calibration"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.savefig(f"{FIGDIR}/fig_calibration.png"); plt.close(fig)


def fig_metrics():
    """Model discrimination comparison with bootstrap CIs."""
    s = pd.read_csv(RESULTS_DIR / "summary_death.csv")
    order = ["Mayo PBC score", "Cox PH", "RSF", "Gradient Boost"]
    s = s.set_index("model").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(order)); w = 0.35
    for i, (m, lbl, c) in enumerate([("c_harrell", "Harrell's C", "#34495e"),
                                     ("c_uno", "Uno's C", "#95a5a6")]):
        err = [s[m] - s[m+"_lo"], s[m+"_hi"] - s[m]]
        ax.bar(x + (i-0.5)*w, s[m], w, yerr=err, capsize=3, label=lbl, color=c)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=15)
    ax.set_ylim(0.7, 0.88); ax.set_ylabel("Concordance index")
    ax.set_title("Discrimination: ML survival models vs. the Mayo clinical score")
    ax.legend(frameon=False)
    fig.savefig(f"{FIGDIR}/fig_metrics.png"); plt.close(fig)


if __name__ == "__main__":
    df = load_clean()
    fig_cif(df);            print("fig_cif")
    fig_forest();           print("fig_forest")
    fig_metrics();          print("fig_metrics")
    risk, s5 = oof_risk(df)
    fig_stratification(df, risk); print("fig_stratification")
    fig_calibration(df, s5);      print("fig_calibration")
    print("All figures ->", FIGDIR)
