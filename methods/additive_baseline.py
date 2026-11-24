"""Additive (ridge on one-hot residues) fitness baseline, and the ORBIT verification experiment.

Usage (from the repository root, after data/prepare_data.py and data/build_cubes.py):
    python -m methods.additive_baseline

Model: fitness = b + sum over the 4 sites of w[site, residue]  (one-hot encoding, 4 x 20 = 80 inputs).
Ridge strength is chosen on the official FLIP validation variants, then the model is refit on
train + validation and evaluated once on the FLIP test split (3- and 4-mutants).

Verification of the interaction methodology: an additive model has no interactions, so the WHT of its
predictions over every cube must have exactly zero order-2/3/4 coefficients, and "functional recovery"
R^2_k (how well predicted Walsh coefficients match measured ones, ORBIT eq. 8, pooled over the
613 probe-test cubes) must be ~0 or slightly negative for k >= 2.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from methods.gb1 import AMINO_ACIDS, METRICS_DIR, PROCESSED_DIR
from methods.wht import pooled_r2, subsets_of_order, walsh_transform

ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]   # same grid as ORBIT's probes

# Numbers printed in the ORBIT paper (arXiv 2608.24953, Table 1 and Table 3, ridge row), for comparison only.
PAPER_REPORTED = {"test_r2": 0.1084, "test_spearman": 0.5617, "test_rmse": 1.1619,
                  "functional_recovery_r2": {"1": 0.4126, "2": -0.0012, "3": -0.0147, "4": -0.0218}}


def one_hot(variants):
    """(N, 80) one-hot matrix: column 20*site + amino-acid index."""
    features = np.zeros((len(variants), 4 * len(AMINO_ACIDS)))
    for row, variant in enumerate(variants):
        for site, residue in enumerate(variant):
            features[row, site * len(AMINO_ACIDS) + AMINO_ACIDS.index(residue)] = 1.0
    return features


def fit_additive_model(variants):
    """Fit on FLIP train, choose alpha on FLIP validation, refit on train+val. Returns (model, alpha)."""
    split = variants["split"].to_numpy()
    features, target = one_hot(variants["variant"]), variants["fitness"].to_numpy()
    train, val = split == "train", split == "val"
    val_mse = {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha).fit(features[train], target[train])
        val_mse[alpha] = np.mean((model.predict(features[val]) - target[val]) ** 2)
    best_alpha = min(val_mse, key=val_mse.get)
    refit = train | val
    return Ridge(alpha=best_alpha).fit(features[refit], target[refit]), best_alpha


def functional_recovery(true_fitness, predicted_fitness):
    """R^2_k for k=1..4 between predicted and measured Walsh coefficients, pooled over cubes and subsets of order k.
    Both inputs are (n_cubes, 16) vertex values."""
    true_coefficients, predicted_coefficients = walsh_transform(true_fitness), walsh_transform(predicted_fitness)
    result = {}
    for order in range(1, 5):
        subsets = subsets_of_order(order)
        result[order] = pooled_r2(true_coefficients[:, subsets], predicted_coefficients[:, subsets])
    return result


def main():
    variants = pd.read_csv(PROCESSED_DIR / "variants.csv")
    cubes = np.load(PROCESSED_DIR / "cubes.npz")
    model, alpha = fit_additive_model(variants)

    # ---- fitness prediction on the official FLIP test split ----
    test = (variants["split"] == "test").to_numpy()
    prediction = model.predict(one_hot(variants["variant"][test]))
    truth = variants["fitness"].to_numpy()[test]
    fitness_metrics = {"alpha": alpha, "n_test": int(test.sum()),
                       "test_r2": float(1 - np.sum((truth - prediction) ** 2) / np.sum((truth - truth.mean()) ** 2)),
                       "test_spearman": float(spearmanr(truth, prediction)[0]),
                       "test_rmse": float(np.sqrt(np.mean((truth - prediction) ** 2)))}

    # ---- interaction verification on cubes ----
    predicted_all = model.predict(one_hot(variants["variant"]))
    vertices, probe_split = cubes["vertices"], cubes["probe_split"]
    true_cubes, predicted_cubes = variants["fitness"].to_numpy()[vertices], predicted_all[vertices]
    higher_order = walsh_transform(predicted_cubes)[:, np.concatenate([subsets_of_order(k) for k in (2, 3, 4)])]
    max_abs_higher_order = float(np.abs(higher_order).max())
    assert max_abs_higher_order < 1e-9, "additive model must have no order>=2 Walsh coefficients"
    test_cubes = probe_split == "test"
    recovery = functional_recovery(true_cubes[test_cubes], predicted_cubes[test_cubes])

    report = {"our_reproduction": {**fitness_metrics, "max_abs_predicted_order_2_to_4_coefficient": max_abs_higher_order,
                                   "functional_recovery_r2_probe_test_cubes": {str(k): v for k, v in recovery.items()}},
              "reported_in_orbit_paper": PAPER_REPORTED}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "additive_baseline.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
