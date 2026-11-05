"""Decision-curve analysis at 5 years for the death endpoint, under competing risks.

Out-of-fold predicted 5-year death risks come from the paper's main CV (repeated stratified
5-fold, 4 repeats, stratified on the competing-risks event code, seed 42; run_models.py /
run_advanced.py). The risk of every model is the predicted 5-year cumulative incidence of
death with transplant as a competing event, so a threshold on it is a threshold on the
absolute 5-year risk of death. 1 - S(5y) from a death model with transplant censored is
not that risk: it describes a world without transplant and exceeds the death incidence.

Mayo PBC score, Cox PH, RSF and gradient boosting are cause-specific: inside each training
fold the same model family, with the same settings and the same in-fold preprocessing, is
fit twice, once for death with transplant censored and once for transplant with death
censored (for the Mayo PBC score both are the one-covariate Cox model of
run_models.fit_predict_mayo; for the others the run_advanced.SKSURV_MODELS settings on the
make_preprocessor covariates). The two cumulative hazards, from each model's
predict_cumulative_hazard_function, are evaluated on one grid, the training fold's death and
transplant times up to 5 years, and combined into the death incidence by
cause_specific_cif. Fine-Gray for death (run_finegray.fit_finegray on the same in-fold
preprocessed covariates with the Cox PH model's ridge penalty, alpha = 1.0) gives the
5-year incidence directly. Treat-all and treat-none are the reference strategies.

Net benefit at threshold p_t (Vickers et al.; competing-risks form):
    NB(p_t) = P(r >= p_t) * F_high - P(r >= p_t) * (1 - F_high) * p_t / (1 - p_t)
where F_high is the Aalen-Johansen cumulative incidence of death at 5 years among patients
with predicted risk r >= p_t, so censoring is handled and transplant is a competing event,
not a censoring. Thresholds 0.05 to 0.50 in steps of 0.01. Each CV repeat yields one
out-of-fold risk per patient; NB is computed per repeat and averaged over the 4 repeats.
The 95% interval is the percentile interval over 1000 patient bootstrap resamples of that
repeat-averaged NB.

Transplant endpoint: not analysed. Its 5-year Aalen-Johansen incidence (printed) lies
below the lowest threshold, so treat-all has negative net benefit over the whole range.
Only the 0.05 to 0.10 end of the range classes a sizeable share of patients as high-risk
(the share whose out-of-fold Fine-Gray transplant risk reaches each threshold is printed),
and over 0.20 to 0.50 the high-risk group is a few patients, against a total of 17
transplants within 5 years (printed), so the net benefit there rests on one or two
events. Transplant is also an allocation outcome, decided by listing and organ supply,
with no intervention whose harm-benefit trade-off a 5-year transplant-probability
threshold expresses.

Outputs: results/dca.csv and figures/fig_dca.png.
"""
import os, sys, time, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from core.data_prep import load_clean, make_preprocessor, NUM_COLS, CAT_COLS
from core.eval_utils import serial_predict, survival_at_times
from experiments.run_advanced import SKSURV_MODELS
from experiments.run_finegray import (fit_finegray, aalen_johansen_cif, ENDPOINT_NUM,
                                      endpoint_prep)

warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
RNG = 42
YEAR = 365.25
HORIZON = 5 * YEAR
THRESHOLDS = np.round(np.arange(0.05, 0.5001, 0.01), 2)
N_REPEATS = 4
N_BOOT = 1000
# Ridge for the Fine-Gray model on the full one-hot covariates, the same penalty as the
# Cox PH comparator (sksurv CoxPHSurvivalAnalysis(alpha=1.0) in run_advanced.py); the
# unpenalized fit does not converge on this design in most folds.
FG_ALPHA = 1.0
MODELS = ["Mayo PBC score", "Cox PH", "RSF", "Gradient Boost", "Fine-Gray"]
STRATEGIES = MODELS + ["Treat all", "Treat none"]
COLORS = {"Mayo PBC score": "#c0392b", "Cox PH": "#34495e", "RSF": "#2980b9",
          "Gradient Boost": "#8e44ad", "Fine-Gray": "#27ae60", "Treat all": "#7f8c8d",
          "Treat none": "k"}


def aj_cif_weighted(time, status, cause, t, W):
    """Aalen-Johansen CIF of `cause` at t for each row of a subject-weight matrix W."""
    order = np.argsort(time, kind="mergesort")
    tm, st, W = time[order], status[order], W[:, order]
    ut, start = np.unique(tm, return_index=True)
    n_risk = np.cumsum(W[:, ::-1], axis=1)[:, ::-1][:, start]          # weight with T >= s
    d_all = np.add.reduceat(W * (st != 0), start, axis=1)
    d_k = np.add.reduceat(W * (st == cause), start, axis=1)
    keep = ut <= t
    n_risk, d_all, d_k = n_risk[:, keep], d_all[:, keep], d_k[:, keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        haz_all = np.where(n_risk > 0, d_all / n_risk, 0.0)
        haz_k = np.where(n_risk > 0, d_k / n_risk, 0.0)
    S_left = np.hstack([np.ones((W.shape[0], 1)), np.cumprod(1 - haz_all, axis=1)[:, :-1]])
    return (S_left * haz_k).sum(axis=1)


def net_benefit(time, status, risks, w):
    """NB for every model (rows of `risks`) and threshold, plus treat-all, for weights w."""
    masks = [(r[None, :] >= THRESHOLDS[:, None]) for r in risks]       # each (thr, n)
    M = np.vstack(masks + [np.ones((1, len(w)), bool)]) * w[None, :]
    F = aj_cif_weighted(time, status, 1, HORIZON, M)
    P = M.sum(axis=1) / w.sum()
    odds = np.r_[np.tile(THRESHOLDS / (1 - THRESHOLDS), len(risks)), np.nan]
    nb_models = (P[:-1] * F[:-1] - P[:-1] * (1 - F[:-1]) * odds[:-1]).reshape(len(risks), -1)
    nb_all = P[-1] * F[-1] - P[-1] * (1 - F[-1]) * THRESHOLDS / (1 - THRESHOLDS)
    prop = P[:-1].reshape(len(risks), -1)
    return nb_models, nb_all, prop


def cause_specific_cif(H_k, H_other):
    """Cumulative incidence of cause k at the last grid time, from cause-specific hazards.

    H_k, H_other: (n, J) cumulative hazards of cause k and of the competing cause at the
    grid times t_1 < ... < t_J, with H = 0 before t_1. The overall survival is
    S(t) = exp(-(H_k(t) + H_other(t))). At each t_j the drop S(t_j-) - S(t_j) is split
    between the causes in proportion to their hazard increments, so
        F_k(t_J) = sum_j S(t_j-) * (1 - exp(-dH(t_j))) * dH_k(t_j) / dH(t_j),
    with dH = dH_k + dH_other. To first order in dH this is sum_j S(t_j-) dH_k(t_j); the
    proportional split makes F_k + F_other = 1 - S(t_J) and, with H_other = 0,
    F_k = 1 - exp(-H_k(t_J)) hold exactly.
    """
    H_all = H_k + H_other
    dH_k = np.diff(H_k, axis=1, prepend=0.0)
    dH = np.diff(H_all, axis=1, prepend=0.0)
    S_left = np.exp(-(H_all - dH))
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(dH > 0, dH_k / dH, 0.0)
    return (S_left * -np.expm1(-dH) * share).sum(axis=1)


def chf_on_grid(model, Z, grid):
    """Cumulative hazard of each row of Z at the grid times, 0 before the model's first time."""
    out = np.empty((len(Z), len(grid)))
    for i, f in enumerate(model.predict_cumulative_hazard_function(Z)):
        k = np.searchsorted(f.x, grid, side="right") - 1
        out[i] = np.where(k >= 0, f.a * f.y[np.maximum(k, 0)] + f.b, 0.0)
    return out


def fit_mayo_cox(ztr, y):
    """The one-covariate Cox model of run_models.fit_predict_mayo."""
    return CoxPHSurvivalAnalysis(alpha=1e-4).fit(ztr, y)


def oof_risks(df):
    """Out-of-fold 5-year death incidences, one (repeat, patient) array per model.

    Also returns the largest absolute error, over all folds and test patients, of the
    arithmetic checks of cause_specific_cif: F_death with the transplant hazard set to 0
    against 1 - exp(-H_death(5y)); F_death + F_transplant against 1 - S(5y); and, for the
    models whose survival function is exp(-H) (all but RSF, whose survival function is a
    Kaplan-Meier product), F_death with no transplant hazard against 1 - S_death(5y) from
    the death model's own predict_survival_function.
    """
    X = df[NUM_COLS + CAT_COLS + ["mayo_score"]].copy()
    t, st = df["time"].values, df["event_cr"].values
    y_death = Surv.from_arrays(st == 1, t)
    y_tx = Surv.from_arrays(st == 2, t)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=RNG)
    risk = {m: np.full((N_REPEATS, len(df)), np.nan) for m in MODELS}
    check = {}
    for fold, (tr, te) in enumerate(cv.split(X, st)):
        rep = fold // 5
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        grid = np.unique(t[tr][(st[tr] != 0) & (t[tr] <= HORIZON)])
        imp = SimpleImputer(strategy="median")
        ztr = imp.fit_transform(Xtr[["mayo_score"]]); zte = imp.transform(Xte[["mayo_score"]])
        pre = make_preprocessor()
        Ztr = np.asarray(pre.fit_transform(Xtr), float); Zte = np.asarray(pre.transform(Xte), float)
        fits = {"Mayo PBC score": (lambda y: fit_mayo_cox(ztr, y), zte)}
        for name in ["Cox PH", "RSF", "Gradient Boost"]:
            fits[name] = (lambda y, name=name: serial_predict(SKSURV_MODELS[name]().fit(Ztr, y)), Zte)
        for name, (fit, Zt) in fits.items():
            m_d = fit(y_death[tr])
            H_d = chf_on_grid(m_d, Zt, grid)
            H_t = chf_on_grid(fit(y_tx[tr]), Zt, grid)
            F_d = cause_specific_cif(H_d, H_t)
            risk[name][rep, te] = F_d
            F_alone = cause_specific_cif(H_d, np.zeros_like(H_t))
            e = [np.abs(F_alone - (1 - np.exp(-H_d[:, -1]))).max(),
                 np.abs(F_d + cause_specific_cif(H_t, H_d)
                        - (1 - np.exp(-(H_d[:, -1] + H_t[:, -1])))).max()]
            if name != "RSF":   # Breslow models: the model's own S(t) is exp(-H(t))
                e.append(np.abs(F_alone - (1 - survival_at_times(m_d, Zt, [HORIZON]).ravel())).max())
            check[name] = [max(c, x) for c, x in zip(check[name], e)] if name in check else e
        fg = fit_finegray(Ztr, t[tr], st[tr], 1, robust=False, alpha=FG_ALPHA)
        risk["Fine-Gray"][rep, te] = fg.predict_cif(Zte, HORIZON)
        print(f"  fold {fold + 1}/{5 * N_REPEATS} done")
    return risk, check


def transplant_threshold_check(df):
    """5-year transplant incidence and the share of OOF Fine-Gray risks above thresholds."""
    X = df[ENDPOINT_NUM].copy(); t, st = df["time"].values, df["event_cr"].values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=RNG)
    r = np.zeros((N_REPEATS, len(df)))
    for fold, (tr, te) in enumerate(cv.split(X, st)):
        pre = endpoint_prep()
        fg = fit_finegray(pre.fit_transform(X.iloc[tr]), t[tr], st[tr], 2, robust=False)
        r[fold // 5, te] = fg.predict_cif(pre.transform(X.iloc[te]), HORIZON)
    inc = aalen_johansen_cif(t, st, 2, HORIZON)
    print(f"  5-year transplant incidence (Aalen-Johansen): {inc:.4f}; "
          f"transplants within 5 years: {int(((st == 2) & (t <= HORIZON)).sum())}")
    for p in [0.05, 0.10, 0.20, 0.30, 0.50]:
        print(f"  share with OOF Fine-Gray 5-year transplant risk >= {p:.2f}: "
              f"{(r >= p).mean():.3f}")


def fig_dca(res):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for s in STRATEGIES:
        d = res[res.strategy == s]
        ls = "--" if s in ("Treat all", "Treat none") else "-"
        ax.plot(d.threshold, d.net_benefit, ls, color=COLORS[s], lw=1.8, label=s)
    top = res.net_benefit.max()
    ax.set_ylim(-0.05, top * 1.1); ax.set_xlim(THRESHOLDS[0], THRESHOLDS[-1])
    ax.set_xlabel("Threshold probability (5-year death)"); ax.set_ylabel("Net benefit")
    ax.set_title("Decision curves, 5-year death (competing-risks, out-of-fold)")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.savefig(FIG_DIR / "fig_dca.png"); plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    df = load_clean()
    t, st = df["time"].values.astype(float), df["event_cr"].values
    one = np.ones((1, len(df)))
    print(f"Aalen-Johansen check at 5y (weighted vs run_finegray): "
          f"{aj_cif_weighted(t, st, 1, HORIZON, one)[0]:.6f} vs "
          f"{aalen_johansen_cif(t, st, 1, HORIZON):.6f}")

    print("\n=== Out-of-fold 5-year death risks (5x4 CV) ===")
    risk, check = oof_risks(df)
    print("\nArithmetic checks of cause_specific_cif (max abs error over folds and patients):")
    for m, e in check.items():
        own = f"; vs the model's own 1 - S_death(5y): {e[2]:.2e}" if len(e) > 2 else ""
        print(f"  {m:<16} no transplant hazard vs 1 - exp(-H_death(5y)): {e[0]:.2e}; "
              f"F_death + F_transplant vs 1 - S(5y): {e[1]:.2e}{own}")

    nb_m, nb_a, prop = zip(*[net_benefit(t, st, [risk[m][r] for m in MODELS], np.ones(len(df)))
                             for r in range(N_REPEATS)])
    nb_m, nb_a, prop = np.mean(nb_m, axis=0), np.mean(nb_a, axis=0), np.mean(prop, axis=0)

    rng = np.random.default_rng(RNG)
    boot_m = np.empty((N_BOOT,) + nb_m.shape); boot_a = np.empty((N_BOOT, len(THRESHOLDS)))
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(df), len(df)), minlength=len(df)).astype(float)
        bm, ba, _ = zip(*[net_benefit(t, st, [risk[m][r] for m in MODELS], w)
                          for r in range(N_REPEATS)])
        boot_m[b], boot_a[b] = np.mean(bm, axis=0), np.mean(ba, axis=0)

    rows = []
    for i, m in enumerate(MODELS):
        for j, p in enumerate(THRESHOLDS):
            rows.append({"endpoint": "Death", "threshold": p, "strategy": m,
                         "net_benefit": nb_m[i, j], "nb_lo": np.percentile(boot_m[:, i, j], 2.5),
                         "nb_hi": np.percentile(boot_m[:, i, j], 97.5), "prop_high_risk": prop[i, j]})
    for j, p in enumerate(THRESHOLDS):
        rows.append({"endpoint": "Death", "threshold": p, "strategy": "Treat all",
                     "net_benefit": nb_a[j], "nb_lo": np.percentile(boot_a[:, j], 2.5),
                     "nb_hi": np.percentile(boot_a[:, j], 97.5), "prop_high_risk": 1.0})
        rows.append({"endpoint": "Death", "threshold": p, "strategy": "Treat none",
                     "net_benefit": 0.0, "nb_lo": 0.0, "nb_hi": 0.0, "prop_high_risk": 0.0})
    res = pd.DataFrame(rows).round(4)
    res.to_csv(RESULTS_DIR / "dca.csv", index=False)
    fig_dca(res)

    print("\n=== Net benefit, death at 5 years (mean over CV repeats [95% bootstrap CI]) ===")
    show = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    print(f"{'Strategy':<16}" + "".join(f"{p:>22.2f}" for p in show))
    for s in STRATEGIES:
        d = res[res.strategy == s].set_index("threshold")
        print(f"{s:<16}" + "".join(
            f"{d.loc[p, 'net_benefit']:>8.3f} [{d.loc[p, 'nb_lo']:.3f},{d.loc[p, 'nb_hi']:.3f}]".rjust(22)
            for p in show))

    print("\n=== Transplant endpoint: threshold range check ===")
    transplant_threshold_check(df)
    print(f"\nSaved dca.csv + fig_dca.png ({time.time() - t0:.0f} s)")
