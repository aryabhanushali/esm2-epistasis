"""Interaction accessibility of cached ESM-2 representations (ORBIT protocol).

Examples (from the repository root):
    python -m analysis.accessibility --size 8M --init pretrained --stage pre
    python -m analysis.accessibility --size 8M --init pretrained --stage post --seed 42
    python -m analysis.accessibility --size 8M --init pretrained --stage pre --protocol random   # leaky sensitivity analysis

For every layer l and interaction order k = 1..4:
  1. For each cube, take the 16 cached vertex representations H(v) and apply the WHT -> r_S (one vector per subset S).
  2. Ground truth: apply the WHT to the 16 measured fitness values -> eps_S (one number per subset).
  3. Fit ONE ridge probe per (layer, order) that maps r_S -> eps_S, pooling all cubes AND all subsets with |S| = k.
     (Pooling prevents the probe from using subset identity, as in the official ORBIT code.)
     Features with ~zero variance are dropped; features are standardized with training statistics only; the ridge
     strength is chosen on probe-validation cubes; the probe is then refit on train+validation cubes.
  4. Accessibility A_{l,k} = pooled R^2 of the refit probe on the held-out probe-test cubes (can be negative).
Also reported: the Walsh energy of the representation at each order (how much order-k variation exists at all).

Protocols (--protocol):
  strict (primary, = ORBIT): amino-acid-identity-strict split from data/build_cubes.py. Train/val/test cubes use
         disjoint alternative amino acids at every site, so they share only the wild-type variant.
  random (sensitivity): the same 8,210 cubes split at random in the same proportions. Cubes in different parts
         then share many variants (and the same FLIP train/test variants), so accessibility can be inflated.
         The CSV records the fraction of test-cube variants that also occur in probe-training cubes.

Output: results/metrics/accessibility_<condition>_<protocol>.csv (one row per layer x order; resumable per layer).
"""
import argparse
import json

import numpy as np
import pandas as pd

from methods.gb1 import METRICS_DIR, PROCESSED_DIR
from methods.wht import energy_by_order, pooled_r2, subsets_of_order, walsh_transform
from models.extract_activations import ACTIVATION_DIR, condition_name

ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]    # grid of the official ORBIT code
MIN_FEATURE_STD = 1e-6
SPLIT_NAMES = ["train", "val", "test"]


# ---------------- probe ----------------
def standardize(x_fit, *others):
    """Scale features with the mean/std of x_fit only (no information from other arrays)."""
    mean, std = x_fit.mean(axis=0), x_fit.std(axis=0)
    return [(x - mean) / std for x in (x_fit, *others)]


def ridge_path(x, y, alphas):
    """Ridge solutions for many alphas from ONE eigendecomposition (x already standardized).

    Minimizes ||y - b - x w||^2 + alpha ||w||^2 with an unpenalized intercept; identical to
    sklearn.linear_model.Ridge (checked in test_methods.py) but ~10x faster for the alpha grid.
    Returns a list of (weights, intercept)."""
    x_mean, y_mean = x.mean(axis=0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    eigenvalues, eigenvectors = np.linalg.eigh(xc.T @ xc)
    projected = eigenvectors.T @ (xc.T @ yc)
    eigenvalues = np.clip(eigenvalues, 0, None)
    solutions = []
    for alpha in alphas:
        weights = eigenvectors @ (projected / (eigenvalues + alpha))
        solutions.append((weights, y_mean - x_mean @ weights))
    return solutions


def fit_probe(x_train, y_train, x_val, y_val, x_test):
    """Ridge probe with validation-selected strength. Returns (test prediction, chosen alpha, n_features_used).

    No test information is used for scaling, feature selection or alpha. If no feature varies, the best
    possible linear model is the training mean (as in the official code)."""
    x_train, x_val, x_test = (np.asarray(a, dtype=np.float64) for a in (x_train, x_val, x_test))
    active = x_train.std(axis=0) > MIN_FEATURE_STD               # drop ~constant features
    if not active.any():
        return np.full(len(x_test), np.concatenate([y_train, y_val]).mean()), np.nan, 0
    x_train, x_val, x_test = x_train[:, active], x_val[:, active], x_test[:, active]
    x_train_scaled, x_val_scaled = standardize(x_train, x_val)
    validation_error = [np.mean((x_val_scaled @ w + b - y_val) ** 2)
                        for w, b in ridge_path(x_train_scaled, y_train, ALPHAS)]
    best_alpha = ALPHAS[int(np.argmin(validation_error))]
    x_refit, y_refit = np.concatenate([x_train, x_val]), np.concatenate([y_train, y_val])
    x_refit_scaled, x_test_scaled = standardize(x_refit, x_test)  # refit on probe-train + probe-val only
    (weights, intercept), = ridge_path(x_refit_scaled, y_refit, [best_alpha])
    return x_test_scaled @ weights + intercept, best_alpha, int(active.sum())


def accessibility_by_order(representation, truth):
    """Accessibility of one layer.

    representation: dict split -> (n_cubes, 16, dim) Walsh coefficients r_S of the representation.
    truth:          dict split -> (n_cubes, 16) Walsh coefficients eps_S of measured fitness.
    Returns one dict per order k = 1..4."""
    rows = []
    for order in range(1, 5):
        subsets = subsets_of_order(order)
        dim = representation["train"].shape[2]
        x = {s: representation[s][:, subsets, :].reshape(-1, dim) for s in SPLIT_NAMES}   # pool cubes and subsets
        y = {s: truth[s][:, subsets].reshape(-1) for s in SPLIT_NAMES}
        prediction, alpha, n_features = fit_probe(x["train"], y["train"], x["val"], y["val"], x["test"])
        rows.append({"order": order, "accessibility_r2": pooled_r2(y["test"], prediction),
                     "rmse": float(np.sqrt(np.mean((y["test"] - prediction) ** 2))),
                     "alpha": alpha, "n_active_features": n_features, "n_test_coefficients": len(y["test"])})
    return rows


# ---------------- data ----------------
def cube_split_indices(probe_split, protocol, seed=42):
    """Cube row indices of probe-train / val / test cubes."""
    if protocol == "strict":
        return {s: np.flatnonzero(probe_split == s) for s in SPLIT_NAMES}
    pool = np.flatnonzero(probe_split != "mixed")                # same 8,210 cubes, random assignment
    shuffled = np.random.default_rng(seed).permutation(pool)
    sizes = [int((probe_split == s).sum()) for s in SPLIT_NAMES]
    bounds = np.cumsum([0] + sizes)
    return {s: shuffled[bounds[i]:bounds[i + 1]] for i, s in enumerate(SPLIT_NAMES)}


def seen_variant_fraction(vertices, split_cubes):
    """Fraction of distinct non-wild-type variants of test cubes that also occur in probe-train cubes."""
    train = set(vertices[split_cubes["train"]].ravel())
    test = set(vertices[split_cubes["test"]].ravel()) - {0}
    return len(test & train) / len(test)


def representation_coefficients(layer_values, rows, chunk=512):
    """Walsh-transform the cached vectors of cubes. layer_values: (n_variants, dim); rows: (n_cubes, 16) row ids.
    Done in chunks and stored as float32 to bound memory. Returns (n_cubes, 16, dim)."""
    out = np.empty((len(rows), 16, layer_values.shape[1]), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        block = layer_values[rows[start:start + chunk]].astype(np.float64)
        out[start:start + chunk] = walsh_transform(block)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", required=True)
    parser.add_argument("--init", choices=["pretrained", "random"], required=True)
    parser.add_argument("--stage", choices=["pre", "post"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pooling", choices=["sites", "mean"], default="sites")
    parser.add_argument("--tag", default="")
    parser.add_argument("--protocol", choices=["strict", "random"], default="strict")
    parser.add_argument("--layers", default="all", help="comma-separated layer indices, or 'all'")
    args = parser.parse_args()

    condition = condition_name(args.size, args.init, args.seed, args.stage, args.pooling, args.tag)
    cache_dir = ACTIVATION_DIR / condition
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    n_layers = metadata["n_layers_including_embedding"]
    layers = range(n_layers) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    output_path = METRICS_DIR / f"accessibility_{condition}_{args.protocol}.csv"
    finished = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    done_layers = set(finished["layer"]) if len(finished) else set()

    # ---- cubes, row lookup and ground-truth coefficients (independent of the layer) ----
    variants = pd.read_csv(PROCESSED_DIR / "variants.csv")
    cubes = np.load(PROCESSED_DIR / "cubes.npz")
    vertices = cubes["vertices"]
    split_cubes = cube_split_indices(cubes["probe_split"], args.protocol)
    cached_variants = np.load(cache_dir / "variant_indices.npy")
    row_of_variant = np.full(len(variants), -1, dtype=np.int64)
    row_of_variant[cached_variants] = np.arange(len(cached_variants))
    rows = {s: row_of_variant[vertices[split_cubes[s]]] for s in SPLIT_NAMES}
    assert all((r >= 0).all() for r in rows.values()), "cache does not contain every variant of the probe cubes"
    fitness = variants["fitness"].to_numpy()
    truth = {s: walsh_transform(fitness[vertices[split_cubes[s]]]) for s in SPLIT_NAMES}
    overlap = seen_variant_fraction(vertices, split_cubes)
    print(f"{condition} / {args.protocol}: cubes " + ", ".join(f"{s}={len(split_cubes[s])}" for s in SPLIT_NAMES)
          + f"; test variants also in probe-train cubes: {overlap:.3f}")

    # ---- one layer at a time ----
    results = [finished] if len(finished) else []
    for layer in layers:
        if layer in done_layers:
            continue
        values = np.load(cache_dir / f"layer_{layer:02d}.npy", mmap_mode="r")
        representation = {s: representation_coefficients(values, rows[s]) for s in SPLIT_NAMES}
        _, energy_fraction = energy_by_order(representation["test"].astype(np.float64))
        raw_energy, _ = energy_by_order(representation["test"].astype(np.float64))
        layer_rows = accessibility_by_order(representation, truth)
        for row in layer_rows:
            row.update({"size": args.size, "init": args.init, "stage": args.stage, "seed": args.seed,
                        "pooling": args.pooling, "protocol": args.protocol, "layer": layer,
                        "relative_depth": layer / (n_layers - 1),
                        "energy_fraction": float(np.nanmean(energy_fraction[:, row["order"]])),
                        "raw_energy": float(raw_energy[:, row["order"]].mean()),
                        "test_variants_seen_in_train": overlap})
        results.append(pd.DataFrame(layer_rows))
        pd.concat(results).to_csv(output_path, index=False)
        print("layer %2d  " % layer + "  ".join(f"A{r['order']}={r['accessibility_r2']:+.3f}" for r in layer_rows))
    print("saved", output_path)


if __name__ == "__main__":
    main()
