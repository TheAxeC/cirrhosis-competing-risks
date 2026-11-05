"""Fine-Gray subdistribution hazards for death and transplant.

scikit-survival and lifelines have no Fine-Gray model, so it is implemented here with the
weighted-data approach of R's `survival::finegray` (Geskus, Biometrics 2011): a subject who
had the competing event stays in the risk set after that event with an inverse-probability-
of-censoring weight G(t-)/G(T_i-), where G is the Kaplan-Meier estimate of the censoring
distribution (events placed before censorings at tied times, as `finegray` does). The
weighted data are fitted with a lifelines `CoxPHFitter` (Efron ties) with a subject-clustered
robust variance.

lifelines ignores delayed entry in its score residuals and baseline hazard, so the expanded
data are written without delayed entry: a subject's time-varying weight w(t), which is
non-increasing, is the telescoping sum of rows that all start at 0, one row per step of w(t)
with duration at the step and weight equal to the drop. Summed over a subject's rows the risk
sets, score residuals and Breslow sums equal those of the (start, stop] form.

Outputs
  (i)   results/finegray.csv: full-cohort subdistribution hazard ratios (95% CI, robust SE)
        for death and transplant on the cause-specific Cox covariate sets of
        run_competing.py, next to the coefficients of R `cmprsk::crr` and of R
        `survival::finegray` + `coxph` on the same data when Rscript has those packages.
  (ii)  results/finegray_cv.csv: cross-validated discrimination of the predicted 5-year
        cumulative incidence (CIF), same CV and predictors as run_endpoints.py, scored with
        the truncated IPCW concordance for competing risks of Wolbers et al. (Biostatistics
        2014; the `pec::cindex` / `riskRegression` C for a cause), and Harrell's C with the
        competing event censored for comparison with endpoints_cv.csv.
  (iii) results/finegray_calibration.csv + figures/fig_finegray_calibration.png: mean
        out-of-fold predicted 5-year CIF against the Aalen-Johansen CIF by predicted-risk
        quintile, for each cause.

Imported by run_simulation.py, run_dca.py and run_deephit_cr.py: `fit_finegray`,
`aalen_johansen_cif`, `cr_cindex`.
"""
import os, sys, shutil, subprocess, tempfile, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifelines import CoxPHFitter, AalenJohansenFitter
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.metrics import concordance_index_censored

from core.data_prep import load_clean
from core.eval_utils import bootstrap_ci
from experiments.run_competing import design_matrix
from experiments.run_endpoints import NUM as ENDPOINT_NUM, prep as endpoint_prep

warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
RNG = 42
YEAR = 365.25
HORIZON = 5 * YEAR
CAUSES = {"Death": 1, "Transplant": 2}
# Cause-specific Cox covariate sets of run_competing.py: every design_matrix column for
# death, the parsimonious three for transplant (25 events).
TRANSPLANT_COVS = ["Age_years", "log_Bilirubin", "Albumin"]
C_DEATH, C_TX = "#c0392b", "#2980b9"


# ---------------------------------------------------------------- censoring and AJ --------
def censoring_km(time, status):
    """Kaplan-Meier estimate of the censoring distribution, events before tied censorings.

    Returns (times, G) with G[k] = P(C > times[k]); G(t-) of a time t is the product over
    censoring times strictly before t, which `G_left` evaluates.
    """
    time = np.asarray(time, float); status = np.asarray(status)
    ct = np.unique(time[status == 0])
    at_risk = np.array([(time >= c).sum() - ((time == c) & (status != 0)).sum() for c in ct])
    d_c = np.array([((time == c) & (status == 0)).sum() for c in ct])
    G = np.cumprod(1.0 - d_c / np.maximum(at_risk, 1))
    return ct, G


def G_left(ct, G, t):
    """G(t-): censoring survival just before each t (product over censorings < t)."""
    k = np.searchsorted(ct, np.asarray(t, float), side="left")
    return np.r_[1.0, G][k]


def G_right(ct, G, t):
    """G(t): censoring survival at each t (product over censorings <= t)."""
    k = np.searchsorted(ct, np.asarray(t, float), side="right")
    return np.r_[1.0, G][k]


def aalen_johansen_cif(time, status, cause, t):
    """Aalen-Johansen cumulative incidence of `cause` at time t (status 0 = censored)."""
    time = np.asarray(time, float); status = np.asarray(status)
    if len(time) == 0:
        return np.nan
    ut = np.unique(time[(status != 0) & (time <= t)])
    if len(ut) == 0:
        return 0.0
    n = np.array([(time >= s).sum() for s in ut], float)
    d_all = np.array([((time == s) & (status != 0)).sum() for s in ut], float)
    d_k = np.array([((time == s) & (status == cause)).sum() for s in ut], float)
    S_left = np.r_[1.0, np.cumprod(1.0 - d_all / n)[:-1]]
    return float((S_left * d_k / n).sum())


# ---------------------------------------------------------------- Fine-Gray model ---------
def expand_finegray(time, status, cause):
    """Weighted Fine-Gray data for one cause, as rows that all start at time 0.

    Returns (subject index, duration, event, weight) arrays. Subjects with the event of
    interest or censored keep one row with weight 1. A subject with the competing event at
    T_i is at risk with weight 1 up to T_i and with weight G(t-)/G(T_i-) at each event-of-
    interest time t > T_i; that step function is written as one row per step.
    """
    time = np.asarray(time, float); status = np.asarray(status).astype(int)
    ct, G = censoring_km(time, status)
    tau = np.unique(time[status == cause])                    # event-of-interest times
    g_tau = G_left(ct, G, tau)
    rows_i, rows_t, rows_e, rows_w = [], [], [], []
    comp = (status != 0) & (status != cause)
    for i in np.where(~comp)[0]:
        rows_i.append(i); rows_t.append(time[i])
        rows_e.append(int(status[i] == cause)); rows_w.append(1.0)
    for i in np.where(comp)[0]:
        after = tau > time[i]
        w = g_tau[after] / G_left(ct, G, time[i])            # weight at each later tau
        w_full = np.r_[1.0, w]                                # weight 1 up to T_i
        t_full = np.r_[time[i], tau[after]]
        drop = w_full - np.r_[w_full[1:], 0.0]                # telescoping decrements
        keep = drop > 1e-12
        rows_i.extend([i] * int(keep.sum())); rows_t.extend(t_full[keep])
        rows_e.extend([0] * int(keep.sum())); rows_w.extend(drop[keep])
    return (np.asarray(rows_i), np.asarray(rows_t, float),
            np.asarray(rows_e, int), np.asarray(rows_w, float))


def breslow(durations, events, weights, lp):
    """Weighted Breslow cumulative baseline hazard at the distinct event times."""
    order = np.argsort(durations, kind="mergesort")
    d, e, w, r = durations[order], events[order], weights[order], np.exp(lp[order])
    risk_rev = np.cumsum((w * r)[::-1])[::-1]                 # sum over rows with dur >= d[k]
    ev_t = np.unique(d[e == 1])
    first = np.searchsorted(d, ev_t, side="left")
    num = np.array([w[(d == s) & (e == 1)].sum() for s in ev_t])
    return ev_t, np.cumsum(num / risk_rev[first])


class FineGray:
    """Fitted Fine-Gray model: coefficients, robust SEs, Breslow baseline, CIF prediction."""

    def __init__(self, cph, columns, center, base_t, base_H):
        self.cph, self.columns, self.center = cph, list(columns), center
        self.base_t, self.base_H = base_t, base_H
        self.params_ = cph.params_.loc[self.columns]

    def summary(self):
        s = self.cph.summary.loc[self.columns]
        return pd.DataFrame({"coef": s["coef"], "se": s["se(coef)"],
                             "SHR": s["exp(coef)"], "SHR_lo": s["exp(coef) lower 95%"],
                             "SHR_hi": s["exp(coef) upper 95%"], "p": s["p"]})

    def linear_predictor(self, X):
        X = np.asarray(X, float)
        return (X - self.center) @ self.params_.values

    def predict_cif(self, X, t):
        """Predicted cumulative incidence at time t: 1 - exp(-H0(t) exp(x'b))."""
        k = np.searchsorted(self.base_t, t, side="right") - 1
        H0 = self.base_H[k] if k >= 0 else 0.0
        return 1.0 - np.exp(-H0 * np.exp(self.linear_predictor(X)))


def fit_finegray(X, time, status, cause, robust=True, alpha=0.0):
    """Fit a Fine-Gray model for `cause` (status codes: 0 censored, other ints causes).

    X: DataFrame or 2-D array of covariates, one row per subject. With robust=True the SEs
    are the subject-clustered sandwich (the Fine-Gray variance of `survival::finegray` +
    `coxph`); robust=False skips that O(rows^2) step when only point estimates and
    predictions are needed. alpha > 0 adds the ridge penalty 0.5 * alpha * ||beta||^2 on
    the scale of X, the penalty of sksurv CoxPHSurvivalAnalysis(alpha); lifelines applies
    n_rows * p_j * 0.5 * (beta_j * sd_j)^2 to its internally standardized coefficients, so
    p_j = alpha / (n_rows * sd_j^2). Returns a FineGray.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X, float), columns=[f"x{j}" for j in range(np.shape(X)[1])])
    cols = list(X.columns)
    Xv = X.values.astype(float)
    center = Xv.mean(axis=0)
    idx, dur, ev, w = expand_finegray(time, status, cause)
    data = pd.DataFrame(Xv[idx] - center, columns=cols)
    data["fg_time"], data["fg_event"], data["fg_wt"], data["fg_id"] = dur, ev, w, idx
    penalizer = 0.0
    if alpha > 0:
        sd = data[cols].std(axis=0).values                  # lifelines' _norm_std
        penalizer = alpha / (len(data) * np.maximum(sd, 1e-12) ** 2)
    cph = CoxPHFitter(penalizer=penalizer)
    if robust:
        cph.fit(data, "fg_time", "fg_event", weights_col="fg_wt", cluster_col="fg_id", robust=True)
    else:
        cph.fit(data.drop(columns="fg_id"), "fg_time", "fg_event", weights_col="fg_wt")
    lp = (Xv[idx] - center) @ cph.params_.loc[cols].values
    base_t, base_H = breslow(dur, ev, w, lp)
    return FineGray(cph, cols, center, base_t, base_H)


# ---------------------------------------------------------------- concordance -------------
def cr_cindex(time, status, risk, cause, tau):
    """Truncated IPCW concordance for one cause under competing risks (Wolbers et al. 2014).

    Pairs (i, j) with i having `cause` at T_i <= tau are comparable when T_j > T_i (weight
    1/(G(T_i-)G(T_i))) or when j had a competing event at T_j <= T_i (weight
    1/(G(T_i-)G(T_j-))); a pair is concordant when risk_i > risk_j, half for ties. G is the
    marginal Kaplan-Meier censoring distribution of the evaluation data, as in
    `pec::cindex` and `riskRegression::Score`.
    """
    time = np.asarray(time, float); status = np.asarray(status).astype(int)
    risk = np.asarray(risk, float)
    ct, G = censoring_km(time, status)
    Gl, Gr = G_left(ct, G, time), G_right(ct, G, time)
    comp = (status != 0) & (status != cause)
    num = den = 0.0
    for i in np.where((status == cause) & (time <= tau))[0]:
        a = time > time[i]
        b = comp & (time <= time[i])
        wa = 1.0 / (Gl[i] * Gr[i])
        wb = 1.0 / (Gl[i] * Gl[b])
        q = (risk[i] > risk).astype(float) + 0.5 * (risk[i] == risk)
        num += wa * q[a].sum() + (wb * q[b]).sum()
        den += wa * a.sum() + wb.sum()
    return num / den if den > 0 else np.nan


# ---------------------------------------------------------------- (i) full cohort ---------
def full_cohort(df):
    """Fine-Gray SHRs (unpenalized, robust 95% CI) on the run_competing covariate sets."""
    X = design_matrix(df)
    sets = {"Death": list(X.columns), "Transplant": TRANSPLANT_COVS}
    out, fits = [], {}
    for cause, code in CAUSES.items():
        fg = fit_finegray(X[sets[cause]], df["time"].values, df["event_cr"].values, code)
        s = fg.summary().reset_index().rename(columns={"index": "variable", "covariate": "variable"})
        s.insert(0, "cause", cause.lower())
        out.append(s); fits[cause] = (fg, sets[cause])
    return pd.concat(out, ignore_index=True), X, sets


R_SCRIPT = r"""
suppressMessages({library(cmprsk); library(survival)})
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1]); code <- as.integer(args[2])
covs <- setdiff(names(d), c("time", "status", "id"))
crr_fit <- crr(d$time, d$status, as.matrix(d[, covs]), failcode = code, cencode = 0)
d$st <- factor(d$status, 0:2, labels = c("censor", "death", "transplant"))
et <- c("death", "transplant")[code]
fgd <- finegray(as.formula(paste("Surv(time, st) ~", paste(c(covs, "id"), collapse = "+"))),
                data = d, etype = et)
fit <- coxph(as.formula(paste("Surv(fgstart, fgstop, fgstatus) ~", paste(covs, collapse = "+"))),
             weights = fgwt, cluster = id, data = fgd)
res <- data.frame(variable = covs, crr_coef = unname(crr_fit$coef),
                  crr_se = sqrt(diag(crr_fit$var)),
                  r_finegray_coef = unname(coef(fit)), r_finegray_se = unname(sqrt(diag(vcov(fit)))))
write.csv(res, args[3], row.names = FALSE)
"""


def r_reference(df, X, sets):
    """Coefficients from R cmprsk::crr and survival::finegray + coxph, if R has them."""
    rs = shutil.which("Rscript")
    if rs is None or subprocess.run(
            [rs, "-e", "quit(status = !(requireNamespace('cmprsk', quietly=TRUE) && "
                       "requireNamespace('survival', quietly=TRUE)))"],
            capture_output=True).returncode != 0:
        print("  [R reference skipped: Rscript with cmprsk + survival not found]")
        return None
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "fg_ref.R")
        with open(script, "w") as f:
            f.write(R_SCRIPT)
        for cause, code in CAUSES.items():
            d = X[sets[cause]].copy()
            d["time"], d["status"], d["id"] = df["time"].values, df["event_cr"].values, np.arange(len(df))
            inp, res = os.path.join(tmp, "in.csv"), os.path.join(tmp, "out.csv")
            d.to_csv(inp, index=False)
            subprocess.run([rs, script, inp, str(code), res], check=True, capture_output=True)
            r = pd.read_csv(res); r.insert(0, "cause", cause.lower())
            out.append(r)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- (ii) CV -----------------
def cv_finegray(df, n_repeats=10):
    """Same CV and predictors as run_endpoints.py; returns per-fold scores + OOF CIFs."""
    X = df[ENDPOINT_NUM].copy()
    t, st = df["time"].values, df["event_cr"].values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RNG)
    rows = []
    oof = {c: np.zeros((n_repeats, len(df))) for c in CAUSES}
    for fold, (tr, te) in enumerate(cv.split(X, st)):
        pre = endpoint_prep()
        Ztr = pre.fit_transform(X.iloc[tr]); Zte = pre.transform(X.iloc[te])
        for cause, code in CAUSES.items():
            fg = fit_finegray(Ztr, t[tr], st[tr], code, robust=False)
            cif = fg.predict_cif(Zte, HORIZON)
            oof[cause][fold // 5, te] = cif
            rows.append({"endpoint": cause, "fold": fold,
                         "c_cr": cr_cindex(t[te], st[te], cif, code, HORIZON),
                         "c_harrell": concordance_index_censored(
                             st[te] == code, t[te], cif)[0]})
    return pd.DataFrame(rows), oof


# ---------------------------------------------------------------- (iii) calibration -------
def calibration(df, oof, n_groups=5, n_boot=1000):
    """Mean OOF predicted 5-year CIF vs Aalen-Johansen CIF by predicted-risk quintile."""
    t, st = df["time"].values, df["event_cr"].values
    rng = np.random.default_rng(RNG)
    rows = []
    for cause, code in CAUSES.items():
        pred = oof[cause].mean(axis=0)                        # mean over CV repeats
        grp = pd.qcut(pred, n_groups, labels=False, duplicates="drop")
        for g in np.unique(grp):
            m = np.where(grp == g)[0]
            obs = aalen_johansen_cif(t[m], st[m], code, HORIZON)
            boots = []
            for _ in range(n_boot):
                b = rng.choice(m, len(m), replace=True)
                boots.append(aalen_johansen_cif(t[b], st[b], code, HORIZON))
            rows.append({"endpoint": cause, "group": int(g) + 1, "n": len(m),
                         "pred_cif_5y": round(float(pred[m].mean()), 4),
                         "obs_cif_5y": round(obs, 4),
                         "obs_lo": round(float(np.percentile(boots, 2.5)), 4),
                         "obs_hi": round(float(np.percentile(boots, 97.5)), 4)})
    return pd.DataFrame(rows)


def fig_calibration(cal):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    for ax, (cause, col) in zip(axes, [("Death", C_DEATH), ("Transplant", C_TX)]):
        c = cal[cal.endpoint == cause]
        hi = max(c["obs_hi"].max(), c["pred_cif_5y"].max()) * 1.1
        ax.plot([0, hi], [0, hi], "k--", lw=0.8, label="Perfect calibration")
        ax.errorbar(c["pred_cif_5y"], c["obs_cif_5y"],
                    yerr=[c["obs_cif_5y"] - c["obs_lo"], c["obs_hi"] - c["obs_cif_5y"]],
                    fmt="o-", color=col, capsize=3, label="Fine-Gray (out-of-fold)")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel("Predicted 5-year cumulative incidence")
        ax.set_ylabel("Observed (Aalen-Johansen)")
        ax.set_title(f"{cause}: 5-year calibration by quintile")
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_finegray_calibration.png"); plt.close(fig)


if __name__ == "__main__":
    df = load_clean()

    print("=== Aalen-Johansen check: own estimator vs lifelines at 5 years ===")
    for cause, code in CAUSES.items():
        ajf = AalenJohansenFitter(seed=RNG).fit(df["time"], df["event_cr"], event_of_interest=code)
        cd = ajf.cumulative_density_.iloc[:, 0]
        print(f"  {cause:<10} own {aalen_johansen_cif(df['time'], df['event_cr'], code, HORIZON):.4f}"
              f"   lifelines {cd.loc[:HORIZON].iloc[-1]:.4f}")

    print("\n=== (i) Full-cohort Fine-Gray subdistribution hazard ratios (robust 95% CI) ===")
    fgtab, X, sets = full_cohort(df)
    ref = r_reference(df, X, sets)
    if ref is not None:
        fgtab = fgtab.merge(ref, on=["cause", "variable"], how="left")
        for c in ["crr", "r_finegray"]:
            fgtab[f"{c}_absdiff"] = (fgtab["coef"] - fgtab[f"{c}_coef"]).abs()
    print(fgtab.round(4).to_string(index=False))
    if ref is not None:
        print("\n  Reference comparison, max |coef difference| per cause:")
        for cause in fgtab.cause.unique():
            s = fgtab[fgtab.cause == cause]
            print(f"    {cause:<10} vs cmprsk::crr {s['crr_absdiff'].max():.2e}   "
                  f"vs survival::finegray+coxph {s['r_finegray_absdiff'].max():.2e}   "
                  f"(robust SE max |diff| vs coxph "
                  f"{(s['se'] - s['r_finegray_se']).abs().max():.2e}, vs crr "
                  f"{(s['se'] - s['crr_se']).abs().max():.2e})")
    fgtab.round(5).to_csv(RESULTS_DIR / "finegray.csv", index=False)

    print("\n=== (ii) Cross-validated discrimination of the 5-year CIF (5x10 CV) ===")
    folds, oof = cv_finegray(df)
    rows = []
    for cause, code in CAUSES.items():
        f = folds[folds.endpoint == cause]
        cm, clo, chi = bootstrap_ci(f["c_cr"].values)
        hm, hlo, hhi = bootstrap_ci(f["c_harrell"].values)
        rows.append({"endpoint": cause, "model": "Fine-Gray",
                     "n_events": int((df["event_cr"] == code).sum()),
                     "n_folds_scored": int(np.isfinite(f["c_cr"]).sum()),
                     "c_cr_5y": round(cm, 3), "c_cr_5y_lo": round(clo, 3), "c_cr_5y_hi": round(chi, 3),
                     "c_harrell": round(hm, 3), "c_harrell_lo": round(hlo, 3),
                     "c_harrell_hi": round(hhi, 3)})
        print(f"  {cause:<10} competing-risks C(5y) {cm:.3f} [{clo:.3f}, {chi:.3f}]   "
              f"Harrell C {hm:.3f} [{hlo:.3f}, {hhi:.3f}]")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "finegray_cv.csv", index=False)

    print("\n=== (iii) Calibration of the 5-year CIF by predicted-risk quintile ===")
    cal = calibration(df, oof)
    print(cal.to_string(index=False))
    cal.to_csv(RESULTS_DIR / "finegray_calibration.csv", index=False)
    fig_calibration(cal)
    print("\nSaved finegray.csv, finegray_cv.csv, finegray_calibration.csv, "
          "fig_finegray_calibration.png")
