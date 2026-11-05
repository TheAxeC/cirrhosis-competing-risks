"""Main model comparison: overall survival (death) endpoint.

Repeated stratified 5-fold CV comparing the Mayo PBC clinical score against
Cox PH and two ML survival ensembles. All preprocessing is fit inside each
fold (no leakage), so no test-fold information reaches imputation or scaling.
"""
import os, sys, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

from core.data_prep import load_clean, make_preprocessor, NUM_COLS, CAT_COLS
from core.eval_utils import evaluate_fold, bootstrap_ci, survival_at_times, serial_predict, HORIZONS

warnings.filterwarnings("ignore")
RNG = 42


def make_models():
    return {
        "Cox PH": lambda: CoxPHSurvivalAnalysis(alpha=1.0),
        "RSF": lambda: RandomSurvivalForest(
            n_estimators=300, min_samples_leaf=15, max_features="sqrt",
            n_jobs=-1, random_state=RNG),
        "Gradient Boost": lambda: GradientBoostingSurvivalAnalysis(
            n_estimators=300, learning_rate=0.05, max_depth=2,
            subsample=0.8, random_state=RNG),
    }


def fit_predict_standard(model, pre, Xtr, ytr, Xte):
    """Fit preprocessing + model on a fold; return (risk, surv_fn)."""
    Ztr = pre.fit_transform(Xtr)
    Zte = pre.transform(Xte)
    Ztr = np.asarray(Ztr, dtype=float)
    Zte = np.asarray(Zte, dtype=float)
    serial_predict(model.fit(Ztr, ytr))
    risk = model.predict(Zte)
    surv_fn = lambda times: survival_at_times(model, Zte, times)
    return risk, surv_fn


def fit_predict_mayo(Xtr, ytr, Xte):
    """Clinical baseline: 1-covariate Cox on the Mayo score (imputed in-fold)."""
    imp = SimpleImputer(strategy="median")
    ztr = imp.fit_transform(Xtr[["mayo_score"]])
    zte = imp.transform(Xte[["mayo_score"]])
    m = CoxPHSurvivalAnalysis(alpha=1e-4)
    m.fit(ztr, ytr)
    risk = m.predict(zte)
    surv_fn = lambda times: survival_at_times(m, zte, times)
    return risk, surv_fn


def run(n_splits=5, n_repeats=4):
    df = load_clean()
    X = df[NUM_COLS + CAT_COLS + ["mayo_score"]].copy()
    y_all = Surv.from_arrays(df["event_death"].values, df["time"].values)
    strat = df["event_cr"].values  # keep transplants balanced across folds

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=RNG)
    models = make_models()
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, strat)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y_all[tr], y_all[te]
        # clinical baseline
        risk, sfn = fit_predict_mayo(Xtr, ytr, Xte)
        rows.append({"model": "Mayo PBC score", "fold": fold,
                     **evaluate_fold(ytr, yte, risk, sfn)})
        # ML / statistical models
        for name, ctor in models.items():
            pre = make_preprocessor()
            risk, sfn = fit_predict_standard(ctor(), pre, Xtr, ytr, Xte)
            rows.append({"model": name, "fold": fold,
                         **evaluate_fold(ytr, yte, risk, sfn)})

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "cv_folds_death.csv", index=False)

    # Aggregate with bootstrap CIs
    metrics = ["c_harrell", "c_uno", "auc_1y", "auc_3y", "auc_5y", "ibs"]
    summ = []
    for name in ["Mayo PBC score", "Cox PH", "RSF", "Gradient Boost"]:
        sub = res[res["model"] == name]
        row = {"model": name}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(sub[m].values)
            row[m] = mean
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        summ.append(row)
    summary = pd.DataFrame(summ)
    summary.to_csv(RESULTS_DIR / "summary_death.csv", index=False)
    return res, summary


def fmt(summary):
    cols = [("c_harrell", "C-Harrell"), ("c_uno", "C-Uno"),
            ("auc_3y", "AUC@3y"), ("auc_5y", "AUC@5y"), ("ibs", "IBS")]
    print(f"\n{'Model':<18}" + "".join(f"{lbl:>22}" for _, lbl in cols))
    for _, r in summary.iterrows():
        line = f"{r['model']:<18}"
        for key, _ in cols:
            line += f"{r[key]:>10.3f} [{r[key+'_lo']:.3f},{r[key+'_hi']:.3f}]"[:22].rjust(22)
        print(line)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if smoke:
        # single-split sanity check before the full CV
        df = load_clean()
        X = df[NUM_COLS + CAT_COLS + ["mayo_score"]].copy()
        y = Surv.from_arrays(df["event_death"].values, df["time"].values)
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(np.arange(len(X)), test_size=0.25,
                                  stratify=df["event_cr"].values, random_state=RNG)
        ytr, yte = y[tr], y[te]
        risk, sfn = fit_predict_mayo(X.iloc[tr], ytr, X.iloc[te])
        print("Mayo  :", {k: round(v, 3) for k, v in
                          evaluate_fold(ytr, yte, risk, sfn).items()})
        for name, ctor in make_models().items():
            pre = make_preprocessor()
            risk, sfn = fit_predict_standard(ctor(), pre, X.iloc[tr], ytr, X.iloc[te])
            print(f"{name:<14}:", {k: round(v, 3) for k, v in
                                   evaluate_fold(ytr, yte, risk, sfn).items()})
    else:
        res, summary = run()
        fmt(summary)
        print("\nSaved: analysis/results/summary_death.csv")
