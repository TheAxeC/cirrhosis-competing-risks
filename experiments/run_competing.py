"""Competing-risks analysis: death vs. liver transplant.

Transplant ('CL') is a competing event: it removes a patient from the at-risk
set for death, so it is modelled as a second cause, not as a third
classification label. The script (1) estimates the cumulative incidence of each
cause (Aalen-Johansen) and (2) fits cause-specific Cox models that show which
clinical features drive death and transplant differently, an effect a
three-class classifier (run_classification.py) cannot express.
"""
import os, sys, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifelines import AalenJohansenFitter, CoxPHFitter
from core.data_prep import load_clean
warnings.filterwarnings("ignore")

YEAR = 365.25

# Parsimonious, clinically interpretable covariate set.
CONT = ["Age_years", "Bilirubin", "Albumin", "Prothrombin", "Copper",
        "SGOT", "Platelets"]
LOG = ["Bilirubin", "Prothrombin", "Copper", "SGOT"]
BIN = {"Ascites": "Y", "Hepatomegaly": "Y", "Spiders": "Y", "Sex": "F"}
EDEMA = {"N": 0.0, "S": 0.5, "Y": 1.0}


def design_matrix(df):
    """Build a standardized, interpretable covariate frame for Cox inference."""
    X = pd.DataFrame(index=df.index)
    for c in CONT:
        v = df[c].copy()
        if c in LOG:
            v = np.log(v)
        X[c if c not in LOG else f"log_{c}"] = v
    # median-impute then standardize continuous -> HR per 1 SD
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median())
        X[c] = (X[c] - X[c].mean()) / X[c].std()
    for c, pos in BIN.items():
        X[c] = (df[c] == pos).astype(float)
    X["Edema"] = df["Edema"].map(EDEMA).fillna(0.0)
    X["Stage4"] = (df["Stage"] == 4).astype(float)
    return X


def cif_table(df):
    """Aalen-Johansen cumulative incidence at 1/3/5/8 years for each cause."""
    rows = []
    for cause, code in [("Death", 1), ("Transplant", 2)]:
        ajf = AalenJohansenFitter(calculate_variance=True)
        ajf.fit(df["time"], df["event_cr"], event_of_interest=code)
        cif = ajf.cumulative_density_
        col = cif.columns[0]
        for yr in [1, 3, 5, 8]:
            t = yr * YEAR
            val = cif.loc[:t, col].iloc[-1] if (cif.index <= t).any() else np.nan
            rows.append({"cause": cause, "year": yr, "cif": round(float(val), 3)})
    return pd.DataFrame(rows), df


def cause_specific_cox(df):
    """Fit cause-specific Cox for death and (parsimonious) for transplant."""
    X = design_matrix(df)
    results = {}

    # Death: competing event (transplant) treated as censored.
    dd = X.copy()
    dd["time"] = df["time"].values
    dd["event"] = (df["event_cr"] == 1).astype(int).values
    cph_d = CoxPHFitter(penalizer=0.05)
    cph_d.fit(dd, "time", "event")
    results["death"] = cph_d

    # Transplant: only 25 events -> parsimonious model (3 covariates).
    tt = X[["Age_years", "log_Bilirubin", "Albumin"]].copy()
    tt["time"] = df["time"].values
    tt["event"] = (df["event_cr"] == 2).astype(int).values
    cph_t = CoxPHFitter(penalizer=0.1)
    cph_t.fit(tt, "time", "event")
    results["transplant"] = cph_t
    return results


def summarize_cox(cph, label):
    s = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%",
                     "exp(coef) upper 95%", "p"]].copy()
    s.columns = ["coef", "HR", "HR_lo", "HR_hi", "p"]
    s.insert(0, "cause", label)
    s = s.round(3)
    return s.reset_index().rename(columns={"covariate": "variable", "index": "variable"})


if __name__ == "__main__":
    df = load_clean()
    cif, _ = cif_table(df)
    print("=== Cumulative incidence (Aalen-Johansen) ===")
    print(cif.pivot(index="year", columns="cause", values="cif"))

    cox = cause_specific_cox(df)
    print("\n=== Cause-specific Cox: DEATH (n_events="
          f"{int((df['event_cr']==1).sum())}) ===")
    sd = summarize_cox(cox["death"], "death")
    print(sd.to_string(index=False))
    print(f"  concordance = {cox['death'].concordance_index_:.3f}")

    print("\n=== Cause-specific Cox: TRANSPLANT (n_events="
          f"{int((df['event_cr']==2).sum())}) ===")
    st = summarize_cox(cox["transplant"], "transplant")
    print(st.to_string(index=False))
    print(f"  concordance = {cox['transplant'].concordance_index_:.3f}")

    cif.to_csv(RESULTS_DIR / "cif.csv", index=False)
    pd.concat([sd, st]).to_csv(RESULTS_DIR / "cause_specific_cox.csv", index=False)
    print("\nSaved CIF + cause-specific Cox to results/")
