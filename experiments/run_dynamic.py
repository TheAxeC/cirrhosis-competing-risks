"""Lever 1: dynamic competing-risks survival on the longitudinal PBC data (pbcseq).

The static UCI benchmark uses one snapshot per patient. The sequential Mayo PBC
data (pbcseq: 312 patients, 1945 serial lab measurements) lets us model the
biomarker TRAJECTORY. We (a) fit time-varying cause-specific Cox models showing
the most recent labs carry the prognostic signal, and (b) landmark at 2/4/6
years to show that updating covariates (last value carried forward) discriminates
better than the baseline measurement -- the dynamic prediction gain that neither
the classification framing nor prior single-event survival work on this cohort
captures.
"""
import os, sys, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifelines import CoxTimeVaryingFitter, CoxPHFitter
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")
YEAR = 365.25
RNG = 42
COVS = ["log_bili", "albumin", "log_protime", "age", "edema"]


def load():
    df = pd.read_csv(DATA_DIR / "pbcseq.csv")
    df["log_bili"] = np.log(df["bili"])
    df["log_protime"] = np.log(df["protime"])
    df = df.rename(columns={"age": "age"})
    # forward-fill within patient, then median-impute any leading gaps
    for c in ["log_bili", "albumin", "log_protime", "edema"]:
        df[c] = df.groupby("id")[c].ffill()
        df[c] = df[c].fillna(df[c].median())
    return df


def counting_process(df, cause):
    """Long-format (start, stop, event] intervals for CoxTimeVaryingFitter.
    cause: 2 = death (transplant censored), 1 = transplant (death censored)."""
    rows = []
    for pid, g in df.groupby("id"):
        g = g.sort_values("day").reset_index(drop=True)
        ft = g["futime"].iloc[0]
        st = g["status"].iloc[0]
        for i in range(len(g)):
            start = g["day"].iloc[i]
            stop = g["day"].iloc[i + 1] if i + 1 < len(g) else ft
            if stop <= start:
                continue
            ev = int(i == len(g) - 1 and st == cause)
            rows.append({"id": pid, "start": start, "stop": stop, "event": ev,
                         **{c: g[c].iloc[i] for c in COVS}})
    return pd.DataFrame(rows)


def time_varying_cox(df):
    out = {}
    for name, cause in [("death", 2), ("transplant", 1)]:
        cp = counting_process(df, cause)
        # standardize continuous covariates for per-SD HRs
        z = cp.copy()
        for c in ["log_bili", "albumin", "log_protime", "age"]:
            z[c] = (z[c] - z[c].mean()) / z[c].std()
        ctv = CoxTimeVaryingFitter(penalizer=0.05)
        ctv.fit(z[["id", "start", "stop", "event"] + COVS],
                id_col="id", event_col="event", start_col="start", stop_col="stop")
        out[name] = ctv.summary[["exp(coef)", "exp(coef) lower 95%",
                                 "exp(coef) upper 95%", "p"]].copy()
    return out


def landmark_cohort(df, L, horizon):
    """Build baseline vs dynamic (LVCF) covariate sets at landmark L (days)."""
    rows_base, rows_dyn = [], []
    for pid, g in df.groupby("id"):
        ft = g["futime"].iloc[0]; st = g["status"].iloc[0]
        if ft <= L:
            continue  # not at risk at the landmark
        g = g.sort_values("day")
        base = g.iloc[0]
        recent = g[g["day"] <= L].iloc[-1]
        res = min(ft - L, horizon)
        event = int(st == 2 and (ft - L) <= horizon)  # death within horizon
        rows_base.append({"time": res, "event": event, **{c: base[c] for c in COVS}})
        rows_dyn.append({"time": res, "event": event, **{c: recent[c] for c in COVS}})
    return pd.DataFrame(rows_base), pd.DataFrame(rows_dyn)


def cv_paired(base, dyn, n_rep=20):
    """Paired repeated 5-fold CV: fit baseline and dynamic on the SAME splits.
    Returns mean C for each and a bootstrap 95% CI on the per-fold paired delta."""
    ev = base["event"].values.astype(bool)
    t = base["time"].values
    if ev.sum() < 8:
        return np.nan, np.nan, (np.nan, np.nan)
    cb_f, cd_f, deltas = [], [], []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=RNG)
    for tr, te in cv.split(np.zeros(len(ev)), ev):
        def fit_c(data):
            d = data.iloc[tr]
            m = CoxPHFitter(penalizer=0.1).fit(d[COVS + ["time", "event"]], "time", "event")
            r = m.predict_partial_hazard(data.iloc[te][COVS]).values.ravel()
            return concordance_index_censored(ev[te], t[te], r)[0]
        try:
            cb, cd = fit_c(base), fit_c(dyn)
            cb_f.append(cb); cd_f.append(cd); deltas.append(cd - cb)
        except Exception:
            pass
    deltas = np.array(deltas)
    rng = np.random.default_rng(RNG)
    boot = [rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(2000)]
    return (float(np.mean(cb_f)), float(np.mean(cd_f)),
            (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))))


if __name__ == "__main__":
    df = load()
    print(f"pbcseq: {df['id'].nunique()} patients, {len(df)} measurements\n")

    print("=== Time-varying cause-specific Cox (per-SD HR) ===")
    tvc = time_varying_cox(df)
    for cause, s in tvc.items():
        s = s.round(3)
        s.columns = ["HR", "HR_lo", "HR_hi", "p"]
        print(f"\n[{cause}]"); print(s.to_string())
    pd.concat({k: v for k, v in tvc.items()}).to_csv(RESULTS_DIR / "dynamic_tvcox.csv")

    print("\n=== Landmark analysis: baseline vs dynamic (LVCF), death within 3y ===")
    H = 3 * YEAR
    res = []
    for Lyr in [2, 4, 6]:
        L = Lyr * YEAR
        base, dyn = landmark_cohort(df, L, H)
        n = len(base); nev = int(base["event"].sum())
        cb, cd, (dlo, dhi) = cv_paired(base, dyn)
        res.append({"landmark_y": Lyr, "n_at_risk": n, "deaths_3y": nev,
                    "C_baseline": round(cb, 3), "C_dynamic": round(cd, 3),
                    "delta": round(cd - cb, 3),
                    "delta_lo": round(dlo, 3), "delta_hi": round(dhi, 3)})
        print(f"  LM {Lyr}y: n={n}, deaths={nev} | "
              f"baseline C={cb:.3f}  dynamic C={cd:.3f}  "
              f"delta={cd-cb:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
    pd.DataFrame(res).to_csv(RESULTS_DIR / "dynamic_landmark.csv", index=False)
    print("\nSaved dynamic_tvcox.csv + dynamic_landmark.csv")
