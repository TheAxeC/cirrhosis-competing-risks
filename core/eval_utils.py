"""Evaluation helpers for the survival re-analysis.

Metrics (all from scikit-survival, the standard validated implementations):
  - Harrell's C-index (concordance_index_censored)
  - Uno's IPCW C-index (concordance_index_ipcw)
  - Time-dependent AUC at fixed horizons (cumulative_dynamic_auc)
  - Integrated Brier Score over a within-follow-up grid (integrated_brier_score)

scikit-survival requires every evaluation time point to lie strictly inside the
follow-up support of BOTH the training and the test fold; helpers below clip the
horizons / grid accordingly so individual folds never raise.
"""
import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

YEAR = 365.25
HORIZONS = {"1y": 1 * YEAR, "3y": 3 * YEAR, "5y": 5 * YEAR}


def _common_support(y_train, y_test):
    """Largest time strictly inside both folds' observed-time support."""
    t_tr, t_te = y_train["time"], y_test["time"]
    lo = max(t_tr.min(), t_te.min())
    hi = min(t_tr.max(), t_te.max())
    return lo, hi


def survival_at_times(model, X, times):
    """S(t) matrix (n x len(times)) from a sksurv predict_survival_function."""
    fns = model.predict_survival_function(X)
    return np.asarray([[fn(t) for t in times] for fn in fns])


def evaluate_fold(y_train, y_test, risk, surv_fn=None):
    """Compute discrimination + calibration metrics for one fold.

    risk     : array, higher = higher risk (for C-index / AUC)
    surv_fn  : callable(times)->S(t) matrix, or None to skip Brier/IBS
    Returns dict of scalar metrics (NaN where a horizon is out of fold support).
    """
    out = {}
    # Harrell C
    out["c_harrell"] = concordance_index_censored(
        y_test["event"], y_test["time"], risk)[0]
    lo, hi = _common_support(y_train, y_test)
    # Uno IPCW C (truncated at common support)
    try:
        out["c_uno"] = concordance_index_ipcw(
            y_train, y_test, risk, tau=hi)[0]
    except Exception:
        out["c_uno"] = np.nan

    # Time-dependent AUC at fixed horizons (drop horizons outside support)
    for name, t in HORIZONS.items():
        if lo < t < hi:
            try:
                auc, _ = cumulative_dynamic_auc(y_train, y_test, risk, t)
                out[f"auc_{name}"] = float(np.atleast_1d(auc)[0])
            except Exception:
                out[f"auc_{name}"] = np.nan
        else:
            out[f"auc_{name}"] = np.nan

    # Integrated Brier Score over an interior grid
    if surv_fn is not None:
        t_lo = max(lo, HORIZONS["1y"]) + 1
        t_hi = hi - 1
        if t_hi > t_lo:
            grid = np.linspace(t_lo, t_hi, 20)
            try:
                S = surv_fn(grid)
                out["ibs"] = integrated_brier_score(y_train, y_test, S, grid)
            except Exception:
                out["ibs"] = np.nan
        else:
            out["ibs"] = np.nan
    return out


def bootstrap_ci(values, n_boot=2000, seed=0):
    """Percentile 95% CI of the mean of a list of fold scores."""
    v = np.asarray([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
