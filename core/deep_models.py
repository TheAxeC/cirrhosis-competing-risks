"""Advanced (deep) survival models via pycox, wrapped to match the harness.

Each wrapper returns (risk, surv_fn) where risk is higher = worse prognosis and
surv_fn(times) -> S(t) matrix, so they slot straight into eval_utils.evaluate_fold.
`seed` fixes the inner validation split and every random draw of training, so one
seed gives one result.
All wrappers are defensive: on any failure they return NaNs for that fold rather
than killing the whole CV run.
"""
import random

import numpy as np
import torch
import torchtuples as tt
from pycox.models import CoxPH, DeepHitSingle

torch.set_num_threads(2)
# Deterministic kernels, so a fixed seed gives the same network on every run (CPU only).
torch.use_deterministic_algorithms(True)
_NAN = (None, None)


def _seed_all(seed):
    """Seed every random source a fit draws from: torch (weight initialisation, dropout,
    the torchtuples batch shuffle), numpy and Python `random` (torchtuples names its
    early-stopping checkpoint file with it)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _split(n, frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * frac)
    return idx[k:], idx[:k]


def _surv_from_df(sdf, times):
    idx = sdf.index.values
    M = sdf.values  # (n_times, n_samples)
    out = np.empty((M.shape[1], len(times)))
    for j, t in enumerate(times):
        k = max(0, min(np.searchsorted(idx, t, side="right") - 1, len(idx) - 1))
        out[:, j] = M[k, :]
    return out


def _net(in_features, out_features, nodes=(32, 32), dropout=0.3):
    return tt.practical.MLPVanilla(in_features, list(nodes), out_features,
                                   batch_norm=True, dropout=dropout,
                                   output_bias=(out_features > 1))


def fit_deepsurv(Ztr, ytr, Zte, seed=0):
    """DeepSurv = neural Cox proportional hazards (Katzman et al., 2018)."""
    _seed_all(seed)
    try:
        Ztr = np.asarray(Ztr, "float32"); Zte = np.asarray(Zte, "float32")
        t = ytr["time"].astype("float32"); e = ytr["event"].astype("float32")
        tr, va = _split(len(Ztr), seed=seed)
        net = _net(Ztr.shape[1], 1)
        model = CoxPH(net, tt.optim.Adam(0.01))
        model.fit(Ztr[tr], (t[tr], e[tr]), batch_size=64, epochs=256,
                  callbacks=[tt.callbacks.EarlyStopping(patience=15)],
                  val_data=(Ztr[va], (t[va], e[va])), verbose=False)
        model.compute_baseline_hazards()
        risk = model.predict(Zte).ravel()
        sdf = model.predict_surv_df(Zte)
        return risk, (lambda times: _surv_from_df(sdf, times))
    except Exception as ex:
        print("  [deepsurv fail]", type(ex).__name__, ex)
        return _NAN


def fit_deephit(Ztr, ytr, Zte, seed=0, num_durations=40):
    """DeepHit (single-risk): discrete-time deep survival (Lee et al., 2018)."""
    _seed_all(seed)
    try:
        Ztr = np.asarray(Ztr, "float32"); Zte = np.asarray(Zte, "float32")
        t = ytr["time"].astype("float64"); e = ytr["event"].astype("float64")
        labtrans = DeepHitSingle.label_transform(num_durations)
        tr, va = _split(len(Ztr), seed=seed)
        y_tr = labtrans.fit_transform(t[tr], e[tr])
        y_va = labtrans.transform(t[va], e[va])
        net = _net(Ztr.shape[1], labtrans.out_features)
        model = DeepHitSingle(net, tt.optim.Adam(0.01), alpha=0.2, sigma=0.1,
                              duration_index=labtrans.cuts)
        model.fit(Ztr[tr], y_tr, batch_size=64, epochs=256,
                  callbacks=[tt.callbacks.EarlyStopping(patience=15)],
                  val_data=(Ztr[va], y_va), verbose=False)
        sdf = model.predict_surv_df(Zte)
        # risk = negative area under predicted survival curve (higher = worse)
        risk = -sdf.values.sum(axis=0)
        return risk, (lambda times: _surv_from_df(sdf, times))
    except Exception as ex:
        print("  [deephit fail]", type(ex).__name__, ex)
        return _NAN
