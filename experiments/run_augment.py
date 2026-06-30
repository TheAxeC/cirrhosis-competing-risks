"""Positive contribution: can we IMPROVE the historical Mayo PBC score?

The Mayo score uses only bilirubin, albumin, age, prothrombin and edema. The
cohort, however, records additional routinely-collected markers it ignores
(copper, SGOT, platelets, cholesterol, alkaline phosphatase, fibrosis stage,
ascites, hepatomegaly...). We test whether adding these on top of the Mayo
score -- linearly (Cox) or with learned interactions (gradient boosting) --
yields a better-discriminating, better-calibrated risk model.

Same leakage-free repeated stratified CV as the main comparison.
"""
import os, sys, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from core.data_prep import load_clean
from core.eval_utils import evaluate_fold, bootstrap_ci, survival_at_times

warnings.filterwarnings("ignore")
RNG = 42

EXTRA_NUM = ["Copper", "SGOT", "Platelets", "Cholesterol", "Tryglicerides", "Alk_Phos"]
EXTRA_LOG = ["Copper", "SGOT", "Cholesterol", "Tryglicerides", "Alk_Phos"]
EXTRA_CAT = ["Ascites", "Hepatomegaly", "Spiders", "Stage", "Sex", "Drug"]


def aug_preprocessor():
    log_idx = [EXTRA_NUM.index(c) for c in EXTRA_LOG]

    def logt(X):
        X = np.asarray(X, float).copy()
        X[:, log_idx] = np.log1p(np.clip(X[:, log_idx], 0, None))
        return X

    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("log", FunctionTransformer(logt)),
                    ("sc", StandardScaler())])
    mayo = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([
        ("mayo", mayo, ["mayo_score"]),
        ("num", num, EXTRA_NUM),
        ("cat", cat, EXTRA_CAT),
    ])


def mayo_only(Xtr, ytr, Xte):
    imp = SimpleImputer(strategy="median")
    ztr = imp.fit_transform(Xtr[["mayo_score"]]); zte = imp.transform(Xte[["mayo_score"]])
    m = CoxPHSurvivalAnalysis(alpha=1e-4).fit(ztr, ytr)
    return m.predict(zte), (lambda t: survival_at_times(m, zte, t))


def aug_model(ctor, Xtr, ytr, Xte):
    pre = aug_preprocessor()
    Ztr = np.asarray(pre.fit_transform(Xtr), float)
    Zte = np.asarray(pre.transform(Xte), float)
    m = ctor().fit(Ztr, ytr)
    return m.predict(Zte), (lambda t: survival_at_times(m, Zte, t))


CONFIGS = {
    "Mayo (5 vars)": ("mayo", None),
    "Mayo + extras (Cox)": ("aug", lambda: CoxPHSurvivalAnalysis(alpha=1.0)),
    "Mayo + extras (Boosting)": ("aug", lambda: GradientBoostingSurvivalAnalysis(
        n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8,
        random_state=RNG)),
}


def run(n_splits=5, n_repeats=4):
    df = load_clean()
    cols = ["mayo_score"] + EXTRA_NUM + EXTRA_CAT
    X = df[cols].copy()
    y = Surv.from_arrays(df["event_death"].values, df["time"].values)
    strat = df["event_cr"].values
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RNG)

    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, strat)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        for name, (kind, ctor) in CONFIGS.items():
            if kind == "mayo":
                risk, sfn = mayo_only(Xtr, ytr, Xte)
            else:
                risk, sfn = aug_model(ctor, Xtr, ytr, Xte)
            rows.append({"model": name, "fold": fold, **evaluate_fold(ytr, yte, risk, sfn)})

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "cv_folds_augment.csv", index=False)

    metrics = ["c_harrell", "c_uno", "auc_3y", "auc_5y", "ibs"]
    summ = []
    base = {m: res[res.model == "Mayo (5 vars)"][m].values for m in metrics}
    for name in CONFIGS:
        sub = res[res.model == name]
        row = {"model": name}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(sub[m].values)
            row[m], row[m + "_lo"], row[m + "_hi"] = mean, lo, hi
            # paired delta vs Mayo (per-fold), bootstrap CI of the mean delta
            if name != "Mayo (5 vars)":
                d = sub[m].values - base[m]
                dm, dlo, dhi = bootstrap_ci(d)
                row[m + "_delta"], row[m + "_dlo"], row[m + "_dhi"] = dm, dlo, dhi
        summ.append(row)
    summary = pd.DataFrame(summ)
    summary.to_csv(RESULTS_DIR / "summary_augment.csv", index=False)
    return summary


if __name__ == "__main__":
    s = run()
    print(f"\n{'Model':<26}{'C-Harrell':>12}{'C-Uno':>12}{'AUC@5y':>12}{'IBS':>12}")
    for _, r in s.iterrows():
        print(f"{r['model']:<26}{r['c_harrell']:>12.3f}{r['c_uno']:>12.3f}"
              f"{r['auc_5y']:>12.3f}{r['ibs']:>12.3f}")
    print("\nPaired improvement vs Mayo (mean delta [95% CI], + = better):")
    for _, r in s.iterrows():
        if r["model"] == "Mayo (5 vars)":
            continue
        print(f"  {r['model']}")
        for m, lbl in [("c_harrell", "C-Harrell"), ("c_uno", "C-Uno"),
                       ("auc_5y", "AUC@5y"), ("ibs", "IBS (lower=better)")]:
            print(f"     {lbl:<20} {r[m+'_delta']:+.4f} "
                  f"[{r[m+'_dlo']:+.4f}, {r[m+'_dhi']:+.4f}]")
