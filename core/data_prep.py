"""
Data preparation for the cirrhosis (Mayo PBC) survival re-analysis.

Reframes the UCI 878 / Mayo PBC cohort (418 patients) as a time-to-event
problem with competing risks (death vs. transplant), and constructs the
historical Mayo PBC natural-history risk score as a clinical baseline.

The Mayo score cannot be replaced by MELD here because the dataset contains
neither serum creatinine nor INR; the Mayo PBC model was, however, derived on
exactly this cohort (Dickson et al., 1989) and uses only columns we have.
"""
import numpy as np
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

DATA_CSV = DATA_DIR / "cirrhosis.csv"

# Columns that are strongly right-skewed lab values -> log1p helps linear models.
LOG_COLS = ["Bilirubin", "Cholesterol", "Copper", "Alk_Phos", "SGOT",
            "Tryglicerides", "Prothrombin"]
NUM_COLS = ["Age_years", "Bilirubin", "Cholesterol", "Albumin", "Copper",
            "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]
CAT_COLS = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"]


def load_clean():
    """Load the raw CSV and engineer the survival targets + Mayo score."""
    # The raw CSV marks missing values either as an empty field or as the
    # string 'NaNN', in numeric and categorical columns alike; both are read
    # as NaN, so the lab columns parse as floats and 'NaNN' never becomes a
    # category of its own.
    df = pd.read_csv(DATA_CSV, na_values=["NaNN"])
    df = df.drop(columns=["ID"])
    df["Age_years"] = df["Age"] / 365.25

    # --- Survival targets -------------------------------------------------
    # Status: C = censored/alive, CL = censored at transplant, D = death.
    # Time is N_Days (days from registration to event/last contact).
    df["time"] = df["N_Days"].astype(float)

    # Overall survival ("death" cause-specific): transplant treated as censored.
    df["event_death"] = (df["Status"] == "D").astype(bool)

    # Competing-risks event code: 0 = censored, 1 = death, 2 = transplant.
    code = {"C": 0, "D": 1, "CL": 2}
    df["event_cr"] = df["Status"].map(code).astype(int)

    # --- Mayo PBC natural-history risk score (Dickson 1989) ---------------
    # R = 0.871*ln(bili) - 2.53*ln(albumin) + 0.039*age(yr)
    #     + 2.38*ln(protime) + 0.859*edema
    edema_score = df["Edema"].map({"N": 0.0, "S": 0.5, "Y": 1.0})
    df["mayo_score"] = (
        0.871 * np.log(df["Bilirubin"])
        - 2.53 * np.log(df["Albumin"])
        + 0.039 * df["Age_years"]
        + 2.38 * np.log(df["Prothrombin"])
        + 0.859 * edema_score
    )
    return df


def make_preprocessor():
    """ColumnTransformer fit INSIDE each CV fold (no leakage)."""
    log_idx = [NUM_COLS.index(c) for c in LOG_COLS if c in NUM_COLS]

    def log_skewed(X):
        X = np.asarray(X, dtype=float).copy()
        X[:, log_idx] = np.log1p(np.clip(X[:, log_idx], 0, None))
        return X

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(log_skewed)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUM_COLS),
        ("cat", categorical, CAT_COLS),
    ])


if __name__ == "__main__":
    df = load_clean()
    n = len(df)
    print(f"n = {n} patients")
    print("\nStatus distribution:")
    print(df["Status"].value_counts())
    print("\nCompeting-risks events (0=cens,1=death,2=transplant):")
    print(df["event_cr"].value_counts().sort_index())
    print(f"\nDeaths: {df['event_death'].sum()} "
          f"({100*df['event_death'].mean():.1f}%)")
    print(f"Median follow-up (days): {df['time'].median():.0f} "
          f"[{df['time'].min():.0f}-{df['time'].max():.0f}]")
    print(f"\nMayo risk score: mean {df['mayo_score'].mean():.2f}, "
          f"sd {df['mayo_score'].std():.2f}, "
          f"missing {df['mayo_score'].isna().sum()}")
    # Sanity: Mayo score should rank-correlate with shorter survival in deaths.
    deaths = df[df["event_death"]]
    rho = deaths[["mayo_score", "time"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman(mayo, time|death) = {rho:.3f} (expect negative)")
    print(f"\nMissing values per feature:")
    print(df[NUM_COLS + CAT_COLS].isna().sum()[lambda s: s > 0])
