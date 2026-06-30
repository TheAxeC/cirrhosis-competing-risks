# Competing-Risks Survival Framework for Cirrhosis Outcome Prediction

Reproducible analysis pipeline for the paper *"From Classification to Competing Risks: A Critical Re-Examination of a Widely-Used Cirrhosis Outcome Benchmark"* (Kerga & Faes).

It recasts the UCI "Cirrhosis Patient Survival Prediction" benchmark (Mayo PBC cohort) from three-class classification into competing-risks survival, benchmarks seven survival models against the historical Mayo clinical score, adds a dynamic (longitudinal) analysis and a multi-dataset generalization, and tests external transportability. All preprocessing is fit **inside** cross-validation folds (no leakage).

## Key results (reproduced by this code)
- No ML model, deep networks included, beats the five-variable Mayo PBC score (Harrell's C 0.833); all seven models sit at C 0.81-0.83. The ceiling is the variables, not model capacity.
- Recast as a competing event, the transplant outcome (poorly predicted as a class, F1 about 0.21) reaches cross-validated concordance 0.82.
- Cause-specific Cox recovers the established opposite age effects: older age raises the death hazard (HR 1.36/SD) but lowers the transplant hazard (HR 0.74/SD).
- A baseline measurement's death-risk concordance decays 0.87 to 0.64 over six years and is restored to 0.78 by the most recent measurement.

## Layout + how to run (this repository is a Python package)

Paths are resolved by **`core/paths.py`** from the file location: inputs from `data/`; generated result CSVs and figures are written to `results/` and `figures/` inside this repository (created if absent). There is no `analysis/` prefix or symlink. Run drivers as modules from this directory:

```bash
pip install -r requirements.txt    # scikit-survival 0.27, lifelines 0.30, pycox 0.3, Python 3.13
python3 paper.py                        # rebuild EVERYTHING in order (the 13 drivers below)
# or run a single driver:
python -m experiments.table1            # cohort characteristics            -> results/table1.csv
python -m experiments.run_competing     # Aalen-Johansen CIF + cause-specific Cox -> results/{cif,cause_specific_cox}.csv
python -m experiments.run_endpoints     # CV C-index, death & transplant    -> results/endpoints_cv.csv
python -m experiments.run_models        # 4-model CV (Mayo/Cox/RSF/GBM)      -> results/{cv_folds_death,summary_death}.csv
python -m experiments.run_advanced      # 7-model CV (adds Coxnet/DeepSurv/DeepHit) -> results/summary_advanced.csv + fig_metrics_all.png
python -m experiments.run_augment       # Mayo-score augmentation            -> results/{cv_folds_augment,summary_augment}.csv
python -m experiments.run_review_fixes  # refit-in-fold + paired delta-C vs Mayo + Schoenfeld PH (PH p printed)
python -m experiments.run_dynamic       # time-varying Cox + landmarking (pbcseq) -> results/{dynamic_tvcox,dynamic_landmark}.csv
python -m experiments.run_multidataset  # METABRIC/GBSG/SUPPORT simple-vs-complex (pycox downloads) -> results/multidataset.csv
python -m experiments.run_external      # transportability on SUPPORT        -> results/{external_validation,external_hr}.csv
python -m experiments.figures           # fig_cif/forest/stratification/calibration
python -m experiments.figures_ext       # fig_landmark, fig_multidataset
python -m experiments.graphical_abstract # graphical_abstract.png
```

### `core/`: shared library
- `paths.py`: input/results/figure anchors, resolved from the file location.
- `data_prep.py`: data loading, the Mayo PBC risk score, competing-risks targets, leakage-free preprocessing (imported by the run scripts; writes no CSV). `DATA_CSV` resolves to `data/cirrhosis.csv`.
- `eval_utils.py`: survival metrics (Harrell/Uno C, time-dependent AUC, integrated Brier, bootstrap CIs).
- `deep_models.py`: DeepSurv / DeepHit wrappers (pycox).

### `experiments/`: 13 keeper drivers
- `run_advanced.py`: the 7-model death-endpoint comparison (the paper's main benchmark table) + `fig_metrics_all`.
- `run_models.py`: 4-model comparison; kept because `run_advanced` imports `fit_predict_standard`/ `fit_predict_mayo` from it.
- `run_competing` (CIF + cause-specific Cox), `run_endpoints` (CV C-index), `run_augment`, `run_review_fixes` (reviewer analyses; PH prothrombin p=0.002 printed), `run_dynamic` (tvc + landmark), `run_multidataset`, `run_external` (transportability), `table1`, `figures`, `figures_ext`, `graphical_abstract`.

### `data/`: committed public inputs
- `cirrhosis.csv`: Mayo PBC, n=418 (UCI ML Repository id=878; Dickson et al., *Hepatology* 1989).
- `pbcseq.csv`: the serial Mayo PBC cohort, n=312 / 1945 measurements (R `survival` package).
- `support2_full.csv`: the SUPPORT cirrhosis subgroup, n=508 (auton-survival).

All three are small + public; committed for offline reproduction. The multi-dataset arm (`run_multidataset.py`) downloads METABRIC / GBSG / SUPPORT via `pycox` at run time.

## Code -> results -> paper mapping

| Script | Result file(s) | Figure(s) | Paper element |
|---|---|---|---|
| `table1.py` | `table1.csv` | none | Table 1 (cohort) |
| `run_competing.py` | `cif.csv`, `cause_specific_cox.csv` | `fig_cif`, `fig_forest` | Cumulative incidence; cause-specific HRs; in-sample C about 0.85 (stdout) |
| `run_advanced.py` | `summary_advanced.csv`, `cv_folds_advanced.csv` | `fig_metrics_all` | Main 7-model benchmark; Harrell's C 0.81-0.83 |
| `run_endpoints.py` | `endpoints_cv.csv` | none | transplant C=0.822, death C=0.826 |
| `run_augment.py` | `summary_augment.csv`, `cv_folds_augment.csv` | none | augmentation delta-C +0.0006 |
| `run_review_fixes.py` | `refit_baseline.csv`, `paired_delta_vs_mayo.csv` | none | refit delta-C -0.001; PH prothrombin p=0.002 (stdout) |
| `run_dynamic.py` | `dynamic_tvcox.csv`, `dynamic_landmark.csv` | `fig_landmark` | landmark 0.87 to 0.64 to 0.78 |
| `run_multidataset.py` | `multidataset.csv` | `fig_multidataset` | generalisation (complexity grows with n) |
| `run_external.py` | `external_validation.csv`, `external_hr.csv` | none | transportability 0.823 to 0.558 (refit 0.594) |
| `run_models.py` | `summary_death.csv`, `cv_folds_death.csv` | `fig_metrics` (via `figures.py`) | none in the paper (subset) |

Secondary numbers print to stdout only (not persisted): in-sample death concordance about 0.85 (`run_competing`), the Schoenfeld PH p-values (`run_review_fixes`). The main paper uses fig_cif and fig_landmark; the supplement uses fig_metrics_all, fig_multidataset, fig_forest, fig_calibration, and fig_stratification.

## Reproducibility
Fixed random seed (42) throughout; Python 3.13 (see `requirements.txt`). Public inputs committed in `data/`. MIT license (`LICENSE`). Cite the paper (on acceptance) and the dataset (Dickson et al., *Hepatology* 1989; UCI ML Repository id=878).
