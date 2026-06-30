"""External validation on an independent cirrhosis cohort (SUPPORT study).

We train a cause-specific death model on the Mayo PBC cohort (UCI-878) and test
its discrimination on the Cirrhosis subgroup of the SUPPORT study (508 patients;
different hospitals, era and case-mix). Only markers common to both cohorts are
used (bilirubin, albumin, age); SUPPORT lacks prothrombin time and edema, and its
patients are acutely hospitalised, so we validate DISCRIMINATION (transportability
of risk ordering), not absolute-risk calibration. Death is the endpoint
(transplant is not ascertained as a competing event in SUPPORT).
"""
import os, sys, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import RepeatedStratifiedKFold
from core.data_prep import load_clean

warnings.filterwarnings("ignore")
RNG = 42
SHARED = ["log_bili", "albumin", "age"]  # markers present in both cohorts


def pbc_frame():
    df = load_clean()
    out = pd.DataFrame({
        "log_bili": np.log(df["Bilirubin"]),
        "albumin": df["Albumin"],
        "age": df["Age_years"],
        "time": df["time"].values,
        "event": df["event_death"].astype(int).values})
    return out


def support_cirrhosis_frame(train_medians):
    df = pd.read_csv(DATA_DIR / "support2_full.csv")
    df = df[df["dzgroup"] == "Cirrhosis"].copy()
    out = pd.DataFrame({
        "log_bili": np.log(df["bili"]),
        "albumin": df["alb"],
        "age": df["age"],
        "time": df["d.time"].astype(float).values,
        "event": df["death"].astype(int).values})
    # impute external missing with PBC training medians (deploying the PBC model)
    for c in SHARED:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(train_medians[c])
    out = out[out["time"] > 0]
    return out


def cindex(frame, risk):
    return concordance_index_censored(frame["event"].astype(bool),
                                      frame["time"].values, risk)[0]


if __name__ == "__main__":
    pbc = pbc_frame()
    medians = pbc[SHARED].median()
    sup = support_cirrhosis_frame(medians)

    print(f"PBC (train): n={len(pbc)}, deaths={int(pbc.event.sum())}")
    print(f"SUPPORT cirrhosis (external): n={len(sup)}, deaths={int(sup.event.sum())} "
          f"({100*sup.event.mean():.0f}%), median follow-up {sup.time.median():.0f} d\n")

    # Train 3-covariate cause-specific death Cox on PBC, apply to SUPPORT
    cph = CoxPHFitter(penalizer=0.01).fit(pbc[SHARED + ["time", "event"]], "time", "event")
    risk_ext = cph.predict_partial_hazard(sup[SHARED]).values.ravel()
    c_ext = cindex(sup, risk_ext)

    # Internal benchmark: CV C-index of the same 3-covariate model on PBC
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RNG)
    c_int = []
    for tr, te in cv.split(pbc, pbc["event"]):
        m = CoxPHFitter(penalizer=0.01).fit(pbc.iloc[tr][SHARED + ["time", "event"]], "time", "event")
        r = m.predict_partial_hazard(pbc.iloc[te][SHARED]).values.ravel()
        c_int.append(cindex(pbc.iloc[te], r))
    c_int = np.mean(c_int)

    # SUPPORT internal (refit on SUPPORT) for an upper reference + HR direction check
    cph_s = CoxPHFitter(penalizer=0.01).fit(sup[SHARED + ["time", "event"]], "time", "event")
    c_sup_internal = cindex(sup, cph_s.predict_partial_hazard(sup[SHARED]).values.ravel())

    print("=== External validation (death endpoint, 3 shared markers) ===")
    print(f"  PBC internal CV C-index            : {c_int:.3f}")
    print(f"  PBC model -> SUPPORT (external) C   : {c_ext:.3f}")
    print(f"  SUPPORT internal-refit C (ceiling) : {c_sup_internal:.3f}")
    print("\n  Hazard-ratio direction consistency (per the shared markers):")
    hr = pd.DataFrame({
        "PBC_HR": np.exp(cph.params_),
        "SUPPORT_HR": np.exp(cph_s.params_)}).round(3)
    print(hr.to_string())

    pd.DataFrame([{"pbc_internal_cv": round(c_int, 3),
                   "external_support": round(c_ext, 3),
                   "support_internal": round(c_sup_internal, 3),
                   "support_n": len(sup), "support_deaths": int(sup.event.sum())}
                  ]).to_csv(RESULTS_DIR / "external_validation.csv", index=False)
    hr.to_csv(RESULTS_DIR / "external_hr.csv")
    print("\nSaved external_validation.csv + external_hr.csv")
