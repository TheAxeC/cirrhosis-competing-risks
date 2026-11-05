"""Simulation study: which approach recovers transplant risk and the cause-specific effects?

Data-generating mechanism, fitted once to the real cohort and fixed before any comparison:
  - Covariates: real patients' baseline vectors resampled with replacement. The set is the
    five Mayo PBC model variables (age in years, log bilirubin, albumin, log prothrombin
    time, edema 0 / 0.5 / 1), the covariates of core.data_prep's Mayo score,
    run_review_fixes.mayo5_design and run_dynamic.COVS. Complete cases on these five
    (416 of 418) are the resampling pool.
  - Event times: cause-specific Weibull proportional-hazards models for death and for
    transplant, H_k(t | x) = exp(b_k + x'beta_k) t^rho_k, each fitted by maximum likelihood
    (lifelines WeibullAFTFitter, converted to the PH form) with the other event censored.
    Latent times T_k = (E_k / exp(b_k + x'beta_k))^(1/rho_k), E_k ~ Exp(1), so the
    cause-specific hazards of the simulated data are exactly these.
  - Censoring: independent of covariates and event times, drawn from the Kaplan-Meier
    estimate of the real censoring distribution (deaths and transplants treated as
    censored for it); mass the estimate leaves beyond the last censoring time is placed at
    that time, the end of follow-up.
  - Truth: each patient's 5-year cumulative incidence F_k(5y | x) = integral of
    S(u | x) dH_k(u | x), by 200-node Gauss-Legendre quadrature in u^rho_k (smooth
    integrand); F_1 + F_2 = 1 - S(5y) is printed as the quadrature check.

Training sizes n = 418, 1000, 4000 with 200 replicates each; every replicate is scored on
one fixed simulated test set of 5000 patients. Approaches, all on the same five covariates:
  (a) three-class random forest (run_classification.RF_PARAMS), transplant risk = P(CL);
  (b) the same with SMOTE on the training data (run_classification.SMOTE_PARAMS);
  (c) unpenalized multinomial logistic regression on standardized covariates, P(CL);
      age coefficients of CL vs C and of D vs C;
  (d) cause-specific Cox for transplant and for death (lifelines, Breslow baselines),
      risk = the 5-year CIF from both cause-specific hazards;
  (e) Fine-Gray for transplant and for death (run_finegray.fit_finegray), 5-year CIF.
Classifier death risk is P(D). All coefficients are reported per year of age.

Metrics per replicate on the test set: Spearman correlation and Harrell's C (sksurv, every
pair ordered by the true CIF) between each risk score and the true 5-year CIF, for
transplant and for death; the transplant-class F1 of (a) to (c) against status at end of
follow-up; the estimated age effect on transplant and on death. The true sign is that of
the mechanism's cause-specific age coefficient. Targets of the mean estimate: the
cause-specific Cox coefficient estimates the mechanism's coefficient; the Fine-Gray and
multinomial coefficients estimate their large-sample (least-false) values under this
mechanism, computed on one simulated sample of 200000 patients (Fine-Gray on the
complete data truncated at the end of follow-up, which the censoring-weighted estimator
targets; multinomial on labels with the mechanism's censoring).

Outputs: results/simulation.csv (mean and 2.5 / 97.5 percentiles over replicates),
results/simulation_mechanism.csv (fitted parameters, simulated vs real event rates),
figures/fig_simulation.png (transplant-risk recovery against n by approach).
"""
import os, sys, time, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from lifelines import WeibullAFTFitter, CoxPHFitter, KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sksurv.metrics import concordance_index_censored
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from core.data_prep import load_clean
from experiments.run_classification import RF_PARAMS, SMOTE_PARAMS
from experiments.run_finegray import fit_finegray, breslow

warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
RNG = 42
YEAR = 365.25
HORIZON = 5 * YEAR
SIZES = [418, 1000, 4000]
N_REP = 200
N_TEST = 5000
N_LARGE = 200_000
COVS = ["age", "log_bili", "albumin", "log_protime", "edema"]
CAUSES = {"death": 1, "transplant": 2}
LABELS = np.array(["C", "D", "CL"])                    # index = event code 0 / 1 / 2
APPROACHES = ["(a) RF", "(b) RF + SMOTE", "(c) Multinomial logistic",
              "(d) Cause-specific Cox", "(e) Fine-Gray"]
COLORS = dict(zip(APPROACHES, ["#7f8c8d", "#e67e22", "#8e44ad", "#c0392b", "#2980b9"]))
N_JOBS = max(1, (os.cpu_count() or 2) - 1)


# ---------------------------------------------------------------- mechanism ---------------
def real_frame():
    df = load_clean()
    d = pd.DataFrame({"age": df["Age_years"], "log_bili": np.log(df["Bilirubin"]),
                      "albumin": df["Albumin"], "log_protime": np.log(df["Prothrombin"]),
                      "edema": df["Edema"].map({"N": 0.0, "S": 0.5, "Y": 1.0}),
                      "time": df["time"], "status": df["event_cr"]})
    return d.dropna().reset_index(drop=True)


def fit_mechanism(real):
    """Cause-specific Weibull PH fits and the censoring distribution, from the real cohort."""
    mech = {}
    for cause, code in CAUSES.items():
        d = real[COVS + ["time"]].copy(); d["event"] = (real["status"] == code).astype(int)
        aft = WeibullAFTFitter().fit(d, "time", "event")
        lam = aft.params_["lambda_"]; rho = float(np.exp(aft.params_["rho_"]["Intercept"]))
        # S = exp(-(t / exp(a0 + x'a))^rho)  =>  H = exp(-rho a0 - rho x'a) t^rho
        mech[cause] = {"rho": rho, "b0": float(-rho * lam["Intercept"]),
                       "beta": np.array([-rho * lam[c] for c in COVS])}
    km = KaplanMeierFitter().fit(real["time"], (real["status"] == 0).astype(int))
    sf = km.survival_function_.iloc[:, 0]
    mech["cens_t"], mech["cens_S"] = sf.index.values.astype(float), sf.values
    return mech


def draw_censoring(mech, n, rng):
    """Inverse-CDF draw from the censoring Kaplan-Meier step function."""
    t, S = mech["cens_t"], mech["cens_S"]
    u = rng.uniform(size=n)
    k = np.searchsorted(-S, -u, side="left")           # first time with S(t) <= u
    return t[np.minimum(k, len(t) - 1)]


def simulate(mech, pool, n, rng, censor=True):
    X = pool[rng.integers(0, len(pool), n)]
    lat = []
    for cause in CAUSES:
        m = mech[cause]
        a = np.exp(m["b0"] + X @ m["beta"])
        lat.append((rng.exponential(size=n) / a) ** (1.0 / m["rho"]))
    T = np.column_stack(lat)
    t_ev = T.min(axis=1); code = T.argmin(axis=1) + 1
    C = draw_censoring(mech, n, rng) if censor else np.full(n, mech["cens_t"].max())
    time_ = np.minimum(t_ev, C)
    status = np.where(t_ev <= C, code, 0)
    return X, time_, status


def true_cif(mech, X, t=HORIZON, nodes=200):
    """F_k(t | x) for both causes by Gauss-Legendre quadrature in v = u^rho_k."""
    xg, wg = np.polynomial.legendre.leggauss(nodes)
    a = {c: np.exp(mech[c]["b0"] + X @ mech[c]["beta"]) for c in CAUSES}
    out = {}
    for c in CAUSES:
        rho = mech[c]["rho"]
        vmax = t ** rho
        v = 0.5 * vmax * (xg + 1.0); w = 0.5 * vmax * wg
        u = v ** (1.0 / rho)
        H = sum(a[k][:, None] * u[None, :] ** mech[k]["rho"] for k in CAUSES)
        out[c] = a[c] * (np.exp(-H) * w[None, :]).sum(axis=1)
    S_t = np.exp(-sum(a[k] * t ** mech[k]["rho"] for k in CAUSES))
    return out, S_t


# ---------------------------------------------------------------- approaches --------------
def harrell_vs_truth(score, truth):
    return concordance_index_censored(np.ones(len(truth), bool), -truth, score)[0]


def cs_cox_cif(fits, Xte, t=HORIZON):
    """5-year CIF of both causes from cause-specific Cox fits (Breslow baselines)."""
    grid = np.unique(np.concatenate([fits[c]["bt"] for c in CAUSES]))
    grid = grid[grid <= t]
    dH, H = {}, {}
    for c in CAUSES:
        f = fits[c]
        k = np.searchsorted(f["bt"], grid, side="right") - 1
        Hb = np.where(k >= 0, f["bH"][np.maximum(k, 0)], 0.0)
        dHb = np.diff(np.r_[0.0, Hb])
        r = np.exp((Xte - f["center"]) @ f["beta"])
        dH[c] = r[:, None] * dHb[None, :]
        H[c] = np.cumsum(dH[c], axis=1)
    Htot = H["death"] + H["transplant"]
    S_left = np.exp(-np.hstack([np.zeros((len(Xte), 1)), Htot[:, :-1]]))
    return {c: (S_left * dH[c]).sum(axis=1) for c in CAUSES}


def fit_cs_cox(X, time_, status):
    fits = {}
    center = X.mean(axis=0)
    for c, code in CAUSES.items():
        d = pd.DataFrame(X - center, columns=COVS)
        d["time"], d["event"] = time_, (status == code).astype(int)
        cph = CoxPHFitter().fit(d, "time", "event")
        beta = cph.params_.loc[COVS].values
        bt, bH = breslow(time_, (status == code).astype(int), np.ones(len(time_)),
                         (X - center) @ beta)
        fits[c] = {"beta": beta, "center": center, "bt": bt, "bH": bH}
    return fits


def run_replicate(mech, pool, test, n, seed):
    rng = np.random.default_rng(seed)
    X, t, st = simulate(mech, pool, n, rng)
    y = LABELS[st]
    Xte, truth, y_te = test
    out = {}

    def score(name, s_tx, s_d, f1=np.nan, age_tx=np.nan, age_d=np.nan):
        out[name] = {"spearman_transplant": spearmanr(s_tx, truth["transplant"])[0],
                     "harrell_transplant": harrell_vs_truth(s_tx, truth["transplant"]),
                     "spearman_death": spearmanr(s_d, truth["death"])[0],
                     "harrell_death": harrell_vs_truth(s_d, truth["death"]),
                     "f1_transplant": f1, "age_transplant": age_tx, "age_death": age_d}

    def classifier(name, model):
        try:
            model.fit(X, y)
            P = model.predict_proba(Xte); cls = list(model.classes_)
            pred = model.classes_[P.argmax(axis=1)]
            score(name, P[:, cls.index("CL")], P[:, cls.index("D")],
                  f1=f1_score(y_te == "CL", pred == "CL", zero_division=0))
        except Exception as ex:
            out[name] = {"error": f"{type(ex).__name__}: {ex}"}
        return model

    classifier("(a) RF", RandomForestClassifier(n_jobs=1, **RF_PARAMS))
    classifier("(b) RF + SMOTE", ImbPipeline([("smote", SMOTE(**SMOTE_PARAMS)),
                                               ("clf", RandomForestClassifier(n_jobs=1, **RF_PARAMS))]))
    mnl = classifier("(c) Multinomial logistic",
                     make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=10000)))
    if "error" not in out["(c) Multinomial logistic"]:
        out["(c) Multinomial logistic"].update(mnl_age(mnl))
    try:
        fits = fit_cs_cox(X, t, st)
        cif = cs_cox_cif(fits, Xte)
        score("(d) Cause-specific Cox", cif["transplant"], cif["death"],
              age_tx=fits["transplant"]["beta"][0], age_d=fits["death"]["beta"][0])
    except Exception as ex:
        out["(d) Cause-specific Cox"] = {"error": f"{type(ex).__name__}: {ex}"}
    try:
        Xdf = pd.DataFrame(X, columns=COVS)
        fg = {c: fit_finegray(Xdf, t, st, code, robust=False) for c, code in CAUSES.items()}
        score("(e) Fine-Gray", fg["transplant"].predict_cif(Xte, HORIZON),
              fg["death"].predict_cif(Xte, HORIZON),
              age_tx=fg["transplant"].params_["age"], age_d=fg["death"].params_["age"])
    except Exception as ex:
        out["(e) Fine-Gray"] = {"error": f"{type(ex).__name__}: {ex}"}
    events = {"deaths": int((st == 1).sum()), "transplants": int((st == 2).sum())}
    return out, events


def mnl_age(pipe):
    """Per-year age coefficients of CL vs C and D vs C from a scaler + multinomial fit."""
    sc, lr = pipe.steps[0][1], pipe.steps[-1][1]
    cls = list(lr.classes_)
    coef = lr.coef_[:, 0] / sc.scale_[0]
    return {"age_transplant": coef[cls.index("CL")] - coef[cls.index("C")],
            "age_death": coef[cls.index("D")] - coef[cls.index("C")]}


def pseudo_true(mech, pool, rng):
    """Large-sample targets of the Fine-Gray and multinomial age coefficients."""
    X, t, st = simulate(mech, pool, N_LARGE, rng, censor=False)   # truncated at end of follow-up
    Xdf = pd.DataFrame(X, columns=COVS)
    fg = {c: fit_finegray(Xdf, t, st, code, robust=False).params_["age"]
          for c, code in CAUSES.items()}
    X2, t2, st2 = simulate(mech, pool, N_LARGE, rng, censor=True)
    mnl = make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, max_iter=10000))
    mnl.fit(X2, LABELS[st2])
    m = mnl_age(mnl)
    return {"(e) Fine-Gray": {"age_transplant": fg["transplant"], "age_death": fg["death"]},
            "(c) Multinomial logistic": m}


# ---------------------------------------------------------------- driver ------------------
def summarize(results, mech, targets):
    true_cs = {"age_transplant": mech["transplant"]["beta"][0], "age_death": mech["death"]["beta"][0]}
    rows = []
    for n in SIZES:
        for a in APPROACHES:
            recs = [r[0][a] for r in results[n]]
            ok = [r for r in recs if "error" not in r]
            for metric in ["spearman_transplant", "harrell_transplant", "spearman_death",
                           "harrell_death", "f1_transplant", "age_transplant", "age_death"]:
                v = np.array([r[metric] for r in ok], float)
                if not np.isfinite(v).any():
                    continue
                v = v[np.isfinite(v)]
                row = {"n": n, "approach": a, "metric": metric, "n_replicates": len(v),
                       "n_failed": len(recs) - len(ok), "mean": v.mean(),
                       "p2_5": np.percentile(v, 2.5), "p97_5": np.percentile(v, 97.5)}
                if metric.startswith("age_"):
                    row["true_cs_coef"] = true_cs[metric]
                    row["target"] = (true_cs[metric] if a == "(d) Cause-specific Cox"
                                     else targets.get(a, {}).get(metric, np.nan))
                    row["prop_true_sign"] = float(np.mean(np.sign(v) == np.sign(true_cs[metric])))
                rows.append(row)
    return pd.DataFrame(rows)


def fig_simulation(summ):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, (metric, lbl) in zip(axes, [("spearman_transplant", "Spearman correlation"),
                                        ("harrell_transplant", "Harrell's C")]):
        for i, a in enumerate(APPROACHES):
            s = summ[(summ.approach == a) & (summ.metric == metric)].sort_values("n")
            x = np.array(s["n"], float) * (1 + 0.04 * (i - 2))
            ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["p2_5"], s["p97_5"] - s["mean"]],
                        fmt="o-", color=COLORS[a], capsize=3, lw=1.6, ms=5, label=a)
        ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels(SIZES)
        ax.set_xlabel("Training size n"); ax.set_ylabel(lbl)
        ax.set_title(f"Transplant risk vs true 5-year CIF: {lbl}", fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_simulation.png"); plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    real = real_frame()
    pool = real[COVS].values.astype(float)
    mech = fit_mechanism(real)
    ss = np.random.SeedSequence(RNG)
    s_test, s_large, s_rep = ss.spawn(3)

    print("=== Data-generating mechanism (fitted to the real cohort, n=%d) ===" % len(real))
    mrows = []
    for c in CAUSES:
        m = mech[c]
        print(f"  {c:<10} rho={m['rho']:.4f}  b0={m['b0']:.4f}  beta=" +
              ", ".join(f"{k}={b:+.4f}" for k, b in zip(COVS, m["beta"])))
        mrows += [{"quantity": f"{c}_rho", "value": m["rho"]}, {"quantity": f"{c}_b0", "value": m["b0"]}]
        mrows += [{"quantity": f"{c}_beta_{k}", "value": b} for k, b in zip(COVS, m["beta"])]
    print(f"  censoring: Kaplan-Meier of real censoring times, {len(mech['cens_t'])} steps, "
          f"residual mass {mech['cens_S'][-1]:.4f} at {mech['cens_t'][-1]:.0f} d")
    mrows.append({"quantity": "censoring_residual_mass_at_end", "value": mech["cens_S"][-1]})

    rng_test = np.random.default_rng(s_test)
    Xte, t_te, st_te = simulate(mech, pool, N_TEST, rng_test)
    truth, S5 = true_cif(mech, Xte)
    qerr = np.abs(truth["death"] + truth["transplant"] - (1 - S5)).max()
    print(f"  quadrature check: max |F_death + F_transplant - (1 - S(5y))| = {qerr:.2e}")
    print(f"  test set (n={N_TEST}): mean true 5y CIF death {truth['death'].mean():.4f}, "
          f"transplant {truth['transplant'].mean():.4f}")
    mrows.append({"quantity": "quadrature_max_abs_error", "value": qerr})

    targets = pseudo_true(mech, pool, np.random.default_rng(s_large))
    print("  large-sample targets (per year of age): " + "; ".join(
        f"{a} {k} {v:+.4f}" for a, d in targets.items() for k, v in d.items()))
    for a, d in targets.items():
        mrows += [{"quantity": f"target_{a}_{k}", "value": v} for k, v in d.items()]

    results = {}
    seeds = s_rep.spawn(len(SIZES) * N_REP)
    for si, n in enumerate(SIZES):
        t1 = time.time()
        results[n] = Parallel(n_jobs=N_JOBS)(
            delayed(run_replicate)(mech, pool, (Xte, truth, LABELS[st_te]), n,
                                   seeds[si * N_REP + r]) for r in range(N_REP))
        print(f"  n={n}: {N_REP} replicates in {time.time() - t1:.0f} s")

    print("\n=== Simulated vs real event counts ===")
    print(f"  real (n={len(real)}): deaths {(real.status == 1).sum()}, transplants {(real.status == 2).sum()}")
    for n in SIZES:
        d = np.array([r[1]["deaths"] for r in results[n]]); tx = np.array([r[1]["transplants"] for r in results[n]])
        print(f"  simulated n={n}: deaths mean {d.mean():.1f} ({100*d.mean()/n:.1f}%), "
              f"transplants mean {tx.mean():.1f} ({100*tx.mean()/n:.1f}%)")
        mrows += [{"quantity": f"sim_n{n}_death_rate", "value": d.mean() / n},
                  {"quantity": f"sim_n{n}_transplant_rate", "value": tx.mean() / n}]
    mrows += [{"quantity": "real_death_rate", "value": (real.status == 1).mean()},
              {"quantity": "real_transplant_rate", "value": (real.status == 2).mean()},
              {"quantity": "test_death_rate", "value": (st_te == 1).mean()},
              {"quantity": "test_transplant_rate", "value": (st_te == 2).mean()}]
    errs = sorted({r[0][a]["error"] for n in SIZES for r in results[n] for a in APPROACHES
                   if "error" in r[0][a]})
    if errs:
        print("  fit errors:", errs)

    summ = summarize(results, mech, targets)
    summ.round(4).to_csv(RESULTS_DIR / "simulation.csv", index=False)
    pd.DataFrame(mrows).to_csv(RESULTS_DIR / "simulation_mechanism.csv", index=False)
    fig_simulation(summ)

    print("\n=== Results (mean [2.5, 97.5 percentile] over replicates) ===")
    for metric in summ.metric.unique():
        print(f"\n[{metric}]")
        for _, r in summ[summ.metric == metric].iterrows():
            extra = (f"  target {r['target']:+.4f}  true CS {r['true_cs_coef']:+.4f}  "
                     f"true sign {r['prop_true_sign']:.3f}" if metric.startswith("age_") else "")
            print(f"  n={r['n']:<5} {r['approach']:<26} {r['mean']:+.4f} "
                  f"[{r['p2_5']:+.4f}, {r['p97_5']:+.4f}]{extra}")
    print(f"\nSaved simulation.csv, simulation_mechanism.csv, fig_simulation.png "
          f"({time.time() - t0:.0f} s)")
