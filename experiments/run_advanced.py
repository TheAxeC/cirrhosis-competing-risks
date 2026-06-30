"""Full model comparison including advanced (penalized + deep) survival models.

Adds Coxnet (elastic-net Cox), DeepSurv and DeepHit to the Mayo / Cox / RSF /
Gradient-Boosting line-up, all under the same leakage-free repeated stratified
CV, and writes a combined summary + an updated discrimination figure.
"""
import os, sys, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

from core.data_prep import load_clean, make_preprocessor, NUM_COLS, CAT_COLS
from core.eval_utils import evaluate_fold, bootstrap_ci
from experiments.run_models import fit_predict_standard, fit_predict_mayo
from core.deep_models import fit_deepsurv, fit_deephit

warnings.filterwarnings("ignore")
RNG = 42

SKSURV_MODELS = {
    "Cox PH": lambda: CoxPHSurvivalAnalysis(alpha=1.0),
    "Coxnet (elastic net)": lambda: CoxnetSurvivalAnalysis(
        l1_ratio=0.5, alpha_min_ratio=0.01, fit_baseline_model=True, max_iter=10**5),
    "RSF": lambda: RandomSurvivalForest(n_estimators=300, min_samples_leaf=15,
                                        max_features="sqrt", n_jobs=-1, random_state=RNG),
    "Gradient Boost": lambda: GradientBoostingSurvivalAnalysis(
        n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=RNG),
}
DEEP_MODELS = {"DeepSurv": fit_deepsurv, "DeepHit": fit_deephit}
ORDER = ["Mayo PBC score", "Cox PH", "Coxnet (elastic net)", "RSF",
         "Gradient Boost", "DeepSurv", "DeepHit"]


def run(n_splits=5, n_repeats=4):
    df = load_clean()
    X = df[NUM_COLS + CAT_COLS + ["mayo_score"]].copy()
    y = Surv.from_arrays(df["event_death"].values, df["time"].values)
    strat = df["event_cr"].values
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RNG)

    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, strat)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        # clinical baseline
        risk, sfn = fit_predict_mayo(Xtr, ytr, Xte)
        rows.append({"model": "Mayo PBC score", "fold": fold, **evaluate_fold(ytr, yte, risk, sfn)})
        # statistical / ensemble
        for name, ctor in SKSURV_MODELS.items():
            risk, sfn = fit_predict_standard(ctor(), make_preprocessor(), Xtr, ytr, Xte)
            rows.append({"model": name, "fold": fold, **evaluate_fold(ytr, yte, risk, sfn)})
        # deep models share the standard preprocessing
        pre = make_preprocessor()
        Ztr = np.asarray(pre.fit_transform(Xtr), float)
        Zte = np.asarray(pre.transform(Xte), float)
        for name, fitfn in DEEP_MODELS.items():
            risk, sfn = fitfn(Ztr, ytr, Zte, seed=fold)
            rec = {"model": name, "fold": fold}
            rec.update(evaluate_fold(ytr, yte, risk, sfn) if risk is not None
                       else {k: np.nan for k in ["c_harrell", "c_uno", "auc_1y",
                             "auc_3y", "auc_5y", "ibs"]})
            rows.append(rec)
        print(f"  fold {fold+1}/{n_splits*n_repeats} done")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "cv_folds_advanced.csv", index=False)

    metrics = ["c_harrell", "c_uno", "auc_1y", "auc_3y", "auc_5y", "ibs"]
    summ = []
    for name in ORDER:
        sub = res[res.model == name]
        row = {"model": name}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(sub[m].values)
            row[m], row[m + "_lo"], row[m + "_hi"] = mean, lo, hi
        summ.append(row)
    summary = pd.DataFrame(summ)
    summary.to_csv(RESULTS_DIR / "summary_advanced.csv", index=False)
    return summary


def fig_metrics_all(summary):
    s = summary.set_index("model").loc[ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(ORDER)); w = 0.38
    for i, (m, lbl, c) in enumerate([("c_harrell", "Harrell's C", "#34495e"),
                                     ("c_uno", "Uno's C", "#95a5a6")]):
        err = [s[m] - s[m + "_lo"], s[m + "_hi"] - s[m]]
        ax.bar(x + (i - 0.5) * w, s[m], w, yerr=err, capsize=3, label=lbl, color=c)
    ax.axhline(s[s.model == "Mayo PBC score"]["c_harrell"].iloc[0], color="#c0392b",
               ls="--", lw=1, label="Mayo (Harrell's C)")
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=20, ha="right")
    ax.set_ylim(0.7, 0.88); ax.set_ylabel("Concordance index")
    ax.set_title("Discrimination across model families (death endpoint)")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_metrics_all.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    summary = run()
    fig_metrics_all(summary)
    print(f"\n{'Model':<22}{'C-Harrell':>11}{'C-Uno':>11}{'AUC@5y':>11}{'IBS':>11}")
    for _, r in summary.iterrows():
        print(f"{r['model']:<22}{r['c_harrell']:>11.3f}{r['c_uno']:>11.3f}"
              f"{r['auc_5y']:>11.3f}{r['ibs']:>11.3f}")
    print("\nSaved: summary_advanced.csv + fig_metrics_all.png")
