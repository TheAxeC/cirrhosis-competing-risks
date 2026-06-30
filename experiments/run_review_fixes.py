"""Reviewer-requested analyses:
(1) In-sample-baseline fix: compare the fixed Mayo score (1989 coefficients) against
    a Cox model refit on the SAME five Mayo variables INSIDE each fold, to test
    whether 'nothing beats Mayo' is an in-sample artefact of the externally-derived score.
(2) Paired per-fold delta-C (Mayo vs each ML model) with bootstrap CI, to substantiate
    'discrimination is statistically indistinguishable' rather than eyeballing overlapping CIs.
(3) Proportional-hazards check for the main Cox model.
"""
import os, sys, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from core.data_prep import load_clean

warnings.filterwarnings("ignore")
RNG = 42
EDEMA = {"N": 0.0, "S": 0.5, "Y": 1.0}
MAYO5 = ["log_bili", "albumin", "age", "log_protime", "edema"]


def mayo5_design(df):
    X = pd.DataFrame({
        "log_bili": np.log(df["Bilirubin"]),
        "albumin": df["Albumin"],
        "age": df["Age_years"],
        "log_protime": np.log(df["Prothrombin"]),
        "edema": df["Edema"].map(EDEMA).fillna(0.0)})
    return X


def boot_ci(v, n=2000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)])
    rng = np.random.default_rng(seed)
    m = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def part1_refit(df):
    X5 = mayo5_design(df)
    mayo = df["mayo_score"].values.reshape(-1, 1)
    y = Surv.from_arrays(df["event_death"].values, df["time"].values)
    strat = df["event_cr"].values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=RNG)
    fixed, refit = [], []
    for tr, te in cv.split(X5, strat):
        # fixed Mayo score (single covariate; only baseline hazard fit in-fold)
        imp = SimpleImputer(strategy="median")
        m1 = CoxPHSurvivalAnalysis(alpha=1e-4).fit(imp.fit_transform(mayo[tr]), y[tr])
        r1 = m1.predict(imp.transform(mayo[te]))
        fixed.append(concordance_index_censored(y[te]["event"], y[te]["time"], r1)[0])
        # refit 5 Mayo variables in-fold (coefficients estimated on training fold)
        pre = SimpleImputer(strategy="median")
        Ztr = StandardScaler().fit(pre.fit_transform(X5.iloc[tr]))
        Xtr = Ztr.transform(pre.transform(X5.iloc[tr]))
        Xte = Ztr.transform(pre.transform(X5.iloc[te]))
        m2 = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xtr, y[tr])
        r2 = m2.predict(Xte)
        refit.append(concordance_index_censored(y[te]["event"], y[te]["time"], r2)[0])
    fixed, refit = np.array(fixed), np.array(refit)
    dm, dlo, dhi = boot_ci(refit - fixed)
    print("=== (1) In-sample-baseline fix (death endpoint, CV Harrell C) ===")
    print(f"  Mayo score (fixed 1989 coefficients) : {fixed.mean():.3f}")
    print(f"  Cox refit on same 5 vars in-fold     : {refit.mean():.3f}")
    print(f"  paired Delta (refit - fixed)         : {dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
    return {"mayo_fixed": round(fixed.mean(), 3), "mayo_refit5": round(refit.mean(), 3),
            "delta": round(dm, 3), "delta_lo": round(dlo, 3), "delta_hi": round(dhi, 3)}


def part2_paired():
    df = pd.read_csv(RESULTS_DIR / "cv_folds_advanced.csv")
    piv = df.pivot_table(index="fold", columns="model", values="c_harrell")
    base = piv["Mayo PBC score"]
    print("\n=== (2) Paired per-fold Delta-C vs Mayo (bootstrap 95% CI) ===")
    rows = []
    for m in ["Cox PH", "Coxnet (elastic net)", "RSF", "Gradient Boost", "DeepSurv", "DeepHit"]:
        d = (piv[m] - base).dropna().values
        dm, dlo, dhi = boot_ci(d)
        excl = "excludes 0" if (dlo > 0 or dhi < 0) else "includes 0"
        print(f"  {m:<22} {dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}]  ({excl})")
        rows.append({"model": m, "delta_vs_mayo": round(dm, 3),
                     "lo": round(dlo, 3), "hi": round(dhi, 3), "ci_excludes_0": dlo > 0 or dhi < 0})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "paired_delta_vs_mayo.csv", index=False)


def part3_ph(df):
    X5 = mayo5_design(df)
    for c in ["log_bili", "albumin", "age", "log_protime"]:
        X5[c] = (X5[c].fillna(X5[c].median()) - X5[c].mean()) / X5[c].std()
    d = X5.copy(); d["time"] = df["time"].values; d["event"] = df["event_death"].astype(int).values
    cph = CoxPHFitter(penalizer=0.05).fit(d, "time", "event")
    pht = proportional_hazard_test(cph, d, time_transform="rank")
    print("\n=== (3) Proportional-hazards test (Schoenfeld, rank) ===")
    print(pht.summary[["test_statistic", "p"]].round(3).to_string())
    gp = pht.summary["p"].min()
    print(f"  global: smallest covariate p = {gp:.3f} "
          f"({'no strong violation' if gp > 0.05 else 'some violation'})")


if __name__ == "__main__":
    df = load_clean()
    r1 = part1_refit(df)
    pd.DataFrame([r1]).to_csv(RESULTS_DIR / "refit_baseline.csv", index=False)
    part2_paired()
    part3_ph(df)
    print("\nSaved refit_baseline.csv + paired_delta_vs_mayo.csv")
