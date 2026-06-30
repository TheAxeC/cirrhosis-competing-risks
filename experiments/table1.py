"""Cohort characteristics (Table 1), overall and by competing outcome."""
import os, sys
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_prep import load_clean

CONT = [("Age_years", "Age, years"), ("Bilirubin", "Bilirubin, mg/dL"),
        ("Albumin", "Albumin, g/dL"), ("Prothrombin", "Prothrombin time, s"),
        ("Copper", "Copper, ug/day"), ("SGOT", "SGOT, U/mL"),
        ("Platelets", "Platelets, x1000/mL"), ("mayo_score", "Mayo risk score")]
CAT = [("Sex", "F", "Female"), ("Ascites", "Y", "Ascites"),
       ("Hepatomegaly", "Y", "Hepatomegaly"), ("Edema", "Y", "Edema (Y)"),
       ("Stage", 4, "Fibrosis stage 4")]


def med_iqr(s):
    s = s.dropna()
    return f"{s.median():.1f} [{s.quantile(.25):.1f}-{s.quantile(.75):.1f}]"


def pct(df, c, v):
    n = (df[c] == v).sum()
    return f"{n} ({100*n/len(df):.0f}%)"


if __name__ == "__main__":
    df = load_clean()
    groups = {"Overall": df,
              "Alive (C)": df[df.event_cr == 0],
              "Death (D)": df[df.event_cr == 1],
              "Transplant (CL)": df[df.event_cr == 2]}
    rows = []
    rows.append({"Characteristic": "n", **{g: str(len(d)) for g, d in groups.items()}})
    for c, lbl in CONT:
        rows.append({"Characteristic": lbl,
                     **{g: med_iqr(d[c]) for g, d in groups.items()}})
    for c, v, lbl in CAT:
        rows.append({"Characteristic": lbl,
                     **{g: pct(d, c, v) for g, d in groups.items()}})
    t1 = pd.DataFrame(rows)
    t1.to_csv(RESULTS_DIR / "table1.csv", index=False)
    print(t1.to_string(index=False))
