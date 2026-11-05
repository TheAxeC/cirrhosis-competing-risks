"""Competing-risks DeepHit: death and transplant as two causes of one network.

pycox's two-event `DeepHit` (Lee et al., AAAI 2018), not `DeepHitSingle`. Architecture and
training follow core/deep_models.py: an MLP with 2 x 32 units, batch norm, dropout 0.3,
Adam (lr 0.01), alpha 0.2, sigma 0.1, 40 equidistant time cuts, batch 64, at most 256
epochs, early stopping with patience 15 on a 20% inner validation split of the training
fold (deep_models._split). The test fold is never seen during training or early stopping.
The output layer has 2 x 40 units, read as one 40-cut probability mass per cause.

CV: the same as run_endpoints.py (repeated stratified 5-fold, 10 repeats, stratified on the
competing-risks event code, seed 42). Two covariate sets are run: the run_endpoints.py
predictors (age, log bilirubin, albumin, log prothrombin; the Fine-Gray CV set of
run_finegray.py) and the full preprocessed covariates used by the existing deep models
(core.data_prep.make_preprocessor).

Discrimination per cause: the truncated IPCW competing-risks C-index of Wolbers et al.
(2014) on the predicted 5-year cumulative incidence (run_finegray.cr_cindex, the metric of
results/finegray_cv.csv), and Harrell's C with the competing event censored.
Output: results/deephit_cr.csv.
"""
import os, sys, time, tempfile, warnings
from core.paths import DATA_DIR, RESULTS_DIR, FIG_DIR
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.metrics import concordance_index_censored

from core.data_prep import load_clean, make_preprocessor, NUM_COLS, CAT_COLS
from core.eval_utils import bootstrap_ci
from core.deep_models import _split
from experiments.run_finegray import cr_cindex, ENDPOINT_NUM, endpoint_prep

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
RNG = 42
YEAR = 365.25
HORIZON = 5 * YEAR
NUM_DURATIONS = 40
CAUSES = {"Death": 1, "Transplant": 2}
FEATURE_SETS = {
    "endpoint predictors (4)": (ENDPOINT_NUM, endpoint_prep),
    "full covariates": (NUM_COLS + CAT_COLS, make_preprocessor),
}


class LabTransform(LabTransDiscreteTime):
    """Discrete-time labels that keep the cause code (0 censored, 1 death, 2 transplant)."""

    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events = events.copy(); events[is_event == 0] = 0
        return durations, events.astype("int64")


class TwoCauseNet(torch.nn.Module):
    """core.deep_models._net with 2 x num_durations outputs, reshaped per cause."""

    def __init__(self, in_features, num_durations, nodes=(32, 32), dropout=0.3):
        super().__init__()
        self.nd = num_durations
        self.mlp = tt.practical.MLPVanilla(in_features, list(nodes), 2 * num_durations,
                                           batch_norm=True, dropout=dropout, output_bias=True)

    def forward(self, x):
        return self.mlp(x).view(x.shape[0], 2, self.nd)


def fit_deephit_cr(Ztr, t, e, Zte, seed):
    """Train on (Ztr, t, e); return the predicted 5-year CIF of each cause for Zte."""
    torch.manual_seed(seed); np.random.seed(seed)
    Ztr = np.asarray(Ztr, "float32"); Zte = np.asarray(Zte, "float32")
    labtrans = LabTransform(NUM_DURATIONS)
    tr, va = _split(len(Ztr), seed=seed)
    y_tr = labtrans.fit_transform(t[tr].astype("float64"), e[tr].astype("int64"))
    y_va = labtrans.transform(t[va].astype("float64"), e[va].astype("int64"))
    model = DeepHit(TwoCauseNet(Ztr.shape[1], NUM_DURATIONS), tt.optim.Adam(0.01),
                    alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
    ckpt = os.path.join(tempfile.gettempdir(), f"deephit_cr_{os.getpid()}_{seed}.pt")
    model.fit(Ztr[tr], y_tr, batch_size=64, epochs=256,
              callbacks=[tt.callbacks.EarlyStopping(patience=15, file_path=ckpt)],
              val_data=(Ztr[va], y_va), verbose=False)
    cif = model.predict_cif(Zte)                     # (causes, cuts, n)
    k = max(0, np.searchsorted(labtrans.cuts, HORIZON, side="right") - 1)
    return {1: cif[0, k, :], 2: cif[1, k, :]}


def run(n_repeats=10):
    df = load_clean()
    t, st = df["time"].values, df["event_cr"].values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RNG)
    rows = []
    for fs, (cols, prep) in FEATURE_SETS.items():
        X = df[cols].copy()
        for fold, (tr, te) in enumerate(cv.split(X, st)):
            pre = prep()
            Ztr = np.asarray(pre.fit_transform(X.iloc[tr]), float)
            Zte = np.asarray(pre.transform(X.iloc[te]), float)
            cif = fit_deephit_cr(Ztr, t[tr], st[tr], Zte, seed=RNG + fold)
            for cause, code in CAUSES.items():
                rows.append({"feature_set": fs, "endpoint": cause, "fold": fold,
                             "c_cr": cr_cindex(t[te], st[te], cif[code], code, HORIZON),
                             "c_harrell": concordance_index_censored(
                                 st[te] == code, t[te], cif[code])[0]})
            if (fold + 1) % 10 == 0:
                print(f"  [{fs}] fold {fold + 1}/{5 * n_repeats} done")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    folds = run()
    out = []
    for (fs, cause), f in folds.groupby(["feature_set", "endpoint"], sort=False):
        cm, clo, chi = bootstrap_ci(f["c_cr"].values)
        hm, hlo, hhi = bootstrap_ci(f["c_harrell"].values)
        out.append({"model": "DeepHit (competing risks)", "feature_set": fs, "endpoint": cause,
                    "n_folds_scored": int(np.isfinite(f["c_cr"]).sum()),
                    "c_cr_5y": round(cm, 3), "c_cr_5y_lo": round(clo, 3), "c_cr_5y_hi": round(chi, 3),
                    "c_harrell": round(hm, 3), "c_harrell_lo": round(hlo, 3),
                    "c_harrell_hi": round(hhi, 3)})
        print(f"  {fs:<25} {cause:<10} competing-risks C(5y) {cm:.3f} [{clo:.3f}, {chi:.3f}]   "
              f"Harrell C {hm:.3f} [{hlo:.3f}, {hhi:.3f}]")
    pd.DataFrame(out).to_csv(RESULTS_DIR / "deephit_cr.csv", index=False)
    print(f"\nSaved: results/deephit_cr.csv ({time.time() - t0:.0f} s)")
