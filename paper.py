"""Single entry point: rebuild every result and figure for the paper.

Runs the 13 experiment drivers in dependency order (run_models before run_advanced,
which imports fit_predict_* from it; the rest are independent). Each driver writes its
CSVs to `results/` and its figures to `figures/`. See README.md for the full
driver -> result -> paper-element map.

Run from this directory:
    python3 paper.py            # rebuild everything (a few minutes; deep models + a pycox download)

Notes:
  - Fixed seed 42 throughout; the Mayo / Cox / Coxnet / RSF / GBM numbers are deterministic.
    DeepSurv / DeepHit wiggle slightly between runs (torch), within the reported noise.
  - run_multidataset downloads METABRIC / GBSG / SUPPORT via pycox on first run (needs internet).
"""
import runpy
import sys

# Dependency order: run_models exports fit_predict_* used by run_advanced; figures read the CSVs.
DRIVERS = [
    "table1",            # Table 1 (cohort)
    "run_competing",     # Aalen-Johansen CIF + cause-specific Cox
    "run_endpoints",     # CV C-index, death & transplant
    "run_models",        # 4-model CV (also exports fit_predict_* for run_advanced)
    "run_advanced",      # 7-model CV -> Table 2
    "run_augment",       # Mayo-score augmentation
    "run_review_fixes",  # refit-in-fold + paired delta-C vs Mayo + Schoenfeld PH
    "run_dynamic",       # time-varying Cox + landmarking (pbcseq)
    "run_multidataset",  # METABRIC/GBSG/SUPPORT simple-vs-complex (pycox download)
    "run_external",      # transportability on SUPPORT cirrhosis subgroup
    "figures",           # fig_cif / forest / stratification / calibration
    "figures_ext",       # fig_landmark, fig_multidataset
    "graphical_abstract",
]


def main(drivers=DRIVERS):
    for name in drivers:
        print("\n" + "=" * 78 + f"\n[{name}]\n" + "=" * 78)
        runpy.run_module(f"experiments.{name}", run_name="__main__")
    print("\nDone. Results in results/, figures in figures/.")


if __name__ == "__main__":
    main(sys.argv[1:] or DRIVERS)
