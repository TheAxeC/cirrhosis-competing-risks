"""Cross-validated discrimination for BOTH competing endpoints.

Backs the headline claim that treating transplant as a competing event (rather
than a classification label) yields strong, cross-validated risk discrimination
for the very outcome that defeated the thesis's classifier (transplant F1=0.21).

Death endpoint : event = death,      transplant censored.
Transplant     : event = transplant, death censored.
The transplant endpoint has only 25 events, so we use more CV repeats and a
parsimonious model, and report bootstrap CIs.
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
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

from core.data_prep import load_clean
from core.eval_utils import bootstrap_ci

warnings.filterwarnings("ignore")
RNG = 42
# Parsimonious, stable predictor set (works for the rare transplant endpoint).
NUM = ["Age_years", "Bilirubin", "Albumin", "Prothrombin"]
LOG = ["Bilirubin", "Prothrombin"]


def prep():
    log_idx = [NUM.index(c) for c in LOG]
    def logt(X):
        X = np.asarray(X, float).copy()
        X[:, log_idx] = np.log1p(np.clip(X[:, log_idx], 0, None))
        return X
    from sklearn.preprocessing import FunctionTransformer
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("log", FunctionTransformer(logt)),
                     ("sc", StandardScaler())])


def eval_endpoint(df, event_code, n_repeats=10):
    X = df[NUM].copy()
    event = (df["event_cr"] == event_code).values
    y = Surv.from_arrays(event, df["time"].values)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RNG)
    ch, cu = [], []
    for tr, te in cv.split(X, df["event_cr"].values):
        pre = prep()
        Ztr = pre.fit_transform(X.iloc[tr]); Zte = pre.transform(X.iloc[te])
        m = CoxPHSurvivalAnalysis(alpha=1.0).fit(Ztr, y[tr])
        risk = m.predict(Zte)
        ch.append(concordance_index_censored(y[te]["event"], y[te]["time"], risk)[0])
        tau = min(y[tr]["time"].max(), y[te]["time"].max())
        try:
            cu.append(concordance_index_ipcw(y[tr], y[te], risk, tau=tau)[0])
        except Exception:
            cu.append(np.nan)
    return ch, cu


if __name__ == "__main__":
    df = load_clean()
    rows = []
    for cause, code in [("Death", 1), ("Transplant", 2)]:
        ch, cu = eval_endpoint(df, code)
        hm, hlo, hhi = bootstrap_ci(ch)
        um, ulo, uhi = bootstrap_ci(cu)
        n_ev = int((df["event_cr"] == code).sum())
        rows.append({"endpoint": cause, "n_events": n_ev,
                     "c_harrell": round(hm, 3), "c_harrell_lo": round(hlo, 3),
                     "c_harrell_hi": round(hhi, 3),
                     "c_uno": round(um, 3), "c_uno_lo": round(ulo, 3),
                     "c_uno_hi": round(uhi, 3)})
        print(f"{cause:<12} (events={n_ev:3d}):  "
              f"C-Harrell {hm:.3f} [{hlo:.3f}, {hhi:.3f}]   "
              f"C-Uno {um:.3f} [{ulo:.3f}, {uhi:.3f}]")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "endpoints_cv.csv", index=False)
    print("\nSaved: analysis/results/endpoints_cv.csv")
