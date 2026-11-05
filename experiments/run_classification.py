"""Three-class classification baseline: status at end of follow-up (C / CL / D).

Treats the outcome as the UCI benchmark does, a label C (alive), CL (transplant) or D
(death) with follow-up time discarded, and scores how well standard classifiers recover
each class. Features are the baseline covariates of the main seven-model comparison
(NUM_COLS + CAT_COLS through core.data_prep.make_preprocessor); follow-up time and the
status / event columns are excluded. The preprocessor, and SMOTE where used, are fit on
the training fold only, inside an imblearn pipeline, so oversampling never touches the
test fold.

Protocol: RepeatedStratifiedKFold, 5 folds x 10 repeats, stratified on the label, seed 42.
Reports per-class F1, macro-F1 and accuracy as the mean over the 50 folds with a 95%
percentile bootstrap CI of that mean (core.eval_utils.bootstrap_ci).
Output: results/classification.csv.
"""
import os, sys, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from core.data_prep import load_clean, make_preprocessor, NUM_COLS, CAT_COLS
from core.eval_utils import bootstrap_ci, serial_predict

warnings.filterwarnings("ignore")
RNG = 42
CLASSES = ["C", "CL", "D"]

# Fixed hyperparameters. The forest uses the scikit-learn classification defaults for leaf
# size and feature sampling; the boosting settings are those of the survival gradient
# boosting model in run_models.py / run_advanced.py.
RF_PARAMS = dict(n_estimators=300, max_features="sqrt", min_samples_leaf=1, random_state=RNG)
GB_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8,
                 random_state=RNG)
SMOTE_PARAMS = dict(k_neighbors=5, random_state=RNG)

MODELS = {
    "Random forest": (lambda: RandomForestClassifier(n_jobs=-1, **RF_PARAMS), False),
    "Random forest + SMOTE": (lambda: RandomForestClassifier(n_jobs=-1, **RF_PARAMS), True),
    "Gradient boosting": (lambda: GradientBoostingClassifier(**GB_PARAMS), False),
    "Gradient boosting + SMOTE": (lambda: GradientBoostingClassifier(**GB_PARAMS), True),
}
METRICS = ["f1_C", "f1_CL", "f1_D", "macro_f1", "accuracy"]


def make_pipeline(ctor, smote):
    steps = [("pre", make_preprocessor())]
    if smote:
        steps.append(("smote", SMOTE(**SMOTE_PARAMS)))
    steps.append(("clf", ctor()))
    return Pipeline(steps)


def run(n_splits=5, n_repeats=10):
    df = load_clean()
    X = df[NUM_COLS + CAT_COLS].copy()
    y = df["Status"].values
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RNG)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        for name, (ctor, smote) in MODELS.items():
            pipe = serial_predict(make_pipeline(ctor, smote).fit(X.iloc[tr], y[tr]))
            pred = pipe.predict(X.iloc[te])
            f1 = f1_score(y[te], pred, labels=CLASSES, average=None, zero_division=0)
            rows.append({"model": name, "fold": fold,
                         **{f"f1_{c}": v for c, v in zip(CLASSES, f1)},
                         "macro_f1": f1.mean(), "accuracy": accuracy_score(y[te], pred)})
    res = pd.DataFrame(rows)
    summ = []
    for name in MODELS:
        sub = res[res.model == name]
        row = {"model": name, "n_folds": len(sub)}
        for m in METRICS:
            mean, lo, hi = bootstrap_ci(sub[m].values)
            row[m], row[m + "_lo"], row[m + "_hi"] = round(mean, 3), round(lo, 3), round(hi, 3)
        summ.append(row)
    summary = pd.DataFrame(summ)
    summary.to_csv(RESULTS_DIR / "classification.csv", index=False)
    return summary


if __name__ == "__main__":
    s = run()
    print(f"{'Model':<27}" + "".join(f"{m:>22}" for m in METRICS))
    for _, r in s.iterrows():
        print(f"{r['model']:<27}" + "".join(
            f"{r[m]:>8.3f} [{r[m+'_lo']:.3f},{r[m+'_hi']:.3f}]".rjust(22) for m in METRICS))
    print("\nSaved: results/classification.csv")
