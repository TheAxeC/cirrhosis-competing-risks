"""Lever 2: does 'complex does not beat simple' generalise beyond cirrhosis?

We replicate the simple-vs-complex survival benchmark on three widely-used public
clinical survival datasets (METABRIC, GBSG, SUPPORT; via pycox), comparing a Cox
baseline against random survival forests, gradient boosting and DeepSurv under the
same repeated stratified CV + Harrell C protocol. This tests whether the cirrhosis
finding -- that model complexity buys little over a simple baseline -- is a quirk
of one small dataset or a general pattern on tabular clinical survival data.
"""
import os, sys, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from core.deep_models import fit_deepsurv

warnings.filterwarnings("ignore")
RNG = 42


def load_datasets():
    from pycox import datasets
    out = {}
    for name, loader in [("METABRIC", datasets.metabric),
                         ("GBSG", datasets.gbsg),
                         ("SUPPORT", datasets.support)]:
        df = loader.read_df()
        cov = [c for c in df.columns if c not in ("duration", "event")]
        out[name] = (df[cov].astype(float).values, df["duration"].values,
                     df["event"].astype(int).values, cov)
    return out


SK_MODELS = {
    "Cox PH": lambda: CoxPHSurvivalAnalysis(alpha=1.0),
    "RSF": lambda: RandomSurvivalForest(n_estimators=200, min_samples_leaf=15,
                                        max_features="sqrt", n_jobs=-1, random_state=RNG),
    "Gradient Boost": lambda: GradientBoostingSurvivalAnalysis(
        n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=RNG),
}


def bench(X, dur, ev, n_splits=5, n_repeats=2):
    y = Surv.from_arrays(ev.astype(bool), dur)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RNG)
    scores = {m: [] for m in list(SK_MODELS) + ["DeepSurv"]}
    for tr, te in cv.split(X, ev):
        sc = StandardScaler().fit(X[tr])
        Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
        for name, ctor in SK_MODELS.items():
            m = ctor().fit(Ztr, y[tr])
            r = m.predict(Zte)
            scores[name].append(concordance_index_censored(y[te]["event"], y[te]["time"], r)[0])
        risk, _ = fit_deepsurv(Ztr, y[tr], Zte, seed=int(tr[0]) % 1000)
        if risk is not None:
            scores["DeepSurv"].append(
                concordance_index_censored(y[te]["event"], y[te]["time"], risk)[0])
    return {m: (np.nanmean(v), np.nanstd(v)) for m, v in scores.items() if v}


if __name__ == "__main__":
    data = load_datasets()
    rows = []
    for name, (X, dur, ev, cov) in data.items():
        print(f"\n=== {name}: n={len(X)}, p={len(cov)}, events={ev.sum()} "
              f"({100*ev.mean():.0f}%) ===")
        res = bench(X, dur, ev)
        cox_c = res["Cox PH"][0]
        for m, (mean, sd) in res.items():
            delta = mean - cox_c
            print(f"  {m:<16} C={mean:.3f} (sd {sd:.3f})  Δvs Cox={delta:+.3f}")
            rows.append({"dataset": name, "n": len(X), "model": m,
                         "c_index": round(mean, 3), "delta_vs_cox": round(delta, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "multidataset.csv", index=False)
    # also append the cirrhosis (this study) death-endpoint numbers for context
    print("\nSaved analysis/results/multidataset.csv")
    print("\nMax improvement of any complex model over Cox, per dataset:")
    for name in data:
        sub = df[(df.dataset == name) & (df.model != "Cox PH")]
        print(f"  {name}: {sub['delta_vs_cox'].max():+.3f}")
