"""Unit tests for cube construction, indexing and the Walsh-Hadamard transform.

Run from the repository root:   python -m pytest methods/test_methods.py -q
Tests marked 'data' need data/processed/*.{csv,npz} (run the data scripts first) and are skipped otherwise.
"""
import itertools

import numpy as np
import pytest

from data.build_cubes import assign_probe_split, cube_vertex_variants, identity_partition
from methods.gb1 import (POSITIONS, WT_SEQUENCE, WT_VARIANT, PROCESSED_DIR, alternatives_at,
                         esm_token_index, variant_to_sequence)
from methods.wht import (SUBSET_ORDERS, WALSH_MATRIX, energy_by_order, inverse_walsh_transform,
                         pooled_r2, subsets_of_order, walsh_transform)


def sign(vertex, site):
    """Walsh coordinate x_i of a vertex: +1 if site i is mutated, else -1."""
    return 1.0 if (vertex >> site) & 1 else -1.0


def landscape(function):
    """Vertex values f(v) for a function of the four coordinates (x0..x3)."""
    return np.array([function(*[sign(v, i) for i in range(4)]) for v in range(16)])


def reference_walsh(values):
    """Direct, loop-based definition: eps_S = 1/16 * sum_v f(v) * prod_{i in S} x_i(v)."""
    coefficients = np.zeros(16)
    for subset in range(16):
        for vertex in range(16):
            character = 1.0
            for site in range(4):
                if (subset >> site) & 1:
                    character *= sign(vertex, site)
            coefficients[subset] += values[vertex] * character / 16
    return coefficients


# ---------------- cube construction and indexing ----------------
def test_vertex_ordering():
    cube = cube_vertex_variants("ACDE")
    assert cube[0] == WT_VARIANT == "VDGV"
    assert cube[1] == "ADGV"          # bit 0 -> site 0 (position 39)
    assert cube[2] == "VCGV"          # bit 1 -> site 1 (position 40)
    assert cube[4] == "VDDV"
    assert cube[8] == "VDGE"
    assert cube[5] == "ADDV"          # 0b0101 -> sites 0 and 2
    assert cube[15] == "ACDE"
    assert len(set(cube)) == 16


def test_biological_vs_python_indexing():
    sequence = variant_to_sequence("ACDE")
    assert len(sequence) == 56 and variant_to_sequence(WT_VARIANT) == WT_SEQUENCE
    for position, residue in zip(POSITIONS, "ACDE"):
        assert sequence[position - 1] == residue            # 1-indexed biology -> 0-indexed Python
    changed = [i for i, (a, b) in enumerate(zip(sequence, WT_SEQUENCE)) if a != b]
    assert changed == [38, 39, 40, 53]


def test_esm_tokenization_positions():
    """ESM-2 token index of biological position p is p (the <cls> token takes index 0)."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    except Exception as error:                                # offline machine
        pytest.skip(f"tokenizer unavailable: {error}")
    sequence = variant_to_sequence("ACDE")
    ids = tokenizer(sequence)["input_ids"]
    assert len(ids) == len(sequence) + 2                      # <cls> ... <eos>
    assert ids[0] == tokenizer.cls_token_id and ids[-1] == tokenizer.eos_token_id
    for position, residue in zip(POSITIONS, "ACDE"):
        assert ids[esm_token_index(position)] == tokenizer.convert_tokens_to_ids(residue)


# ---------------- Walsh-Hadamard transform ----------------
def test_walsh_matrix_is_orthogonal():
    assert np.allclose(WALSH_MATRIX.T @ WALSH_MATRIX, 16 * np.eye(16))
    assert list(SUBSET_ORDERS[[0, 1, 3, 7, 15]]) == [0, 1, 2, 3, 4]
    assert [len(subsets_of_order(k)) for k in range(5)] == [1, 4, 6, 4, 1]


def test_constant_landscape():
    coefficients = walsh_transform(np.full((1, 16), 3.5))[0]
    assert coefficients[0] == pytest.approx(3.5)
    assert np.allclose(coefficients[1:], 0)


def test_additive_landscape_has_no_higher_order_terms():
    a = [0.7, -1.2, 0.3, 2.0]
    f = landscape(lambda x0, x1, x2, x3: 5 + a[0] * x0 + a[1] * x1 + a[2] * x2 + a[3] * x3)
    coefficients = walsh_transform(f[None])[0]
    assert coefficients[0] == pytest.approx(5)
    assert [coefficients[1 << i] for i in range(4)] == pytest.approx(a)
    for order in (2, 3, 4):
        assert np.allclose(coefficients[subsets_of_order(order)], 0, atol=1e-12)


@pytest.mark.parametrize("sites", [(0, 1), (1, 3), (0, 1, 2), (0, 2, 3), (0, 1, 2, 3)])
def test_pure_interaction_lands_on_one_coefficient(sites):
    f = landscape(lambda *x: np.prod([x[i] for i in sites]) * 2.0)
    coefficients = walsh_transform(f[None])[0]
    mask = sum(1 << i for i in sites)
    assert coefficients[mask] == pytest.approx(2.0)
    assert np.allclose(np.delete(coefficients, mask), 0, atol=1e-12)
    assert SUBSET_ORDERS[mask] == len(sites)


def test_mixed_orders_and_perfect_reconstruction():
    f = landscape(lambda x0, x1, x2, x3: 1 + 0.5 * x0 - 2 * x1 * x2 + 0.25 * x0 * x2 * x3 + 3 * x0 * x1 * x2 * x3)
    coefficients = walsh_transform(f[None])[0]
    expected = {0: 1, 0b0001: 0.5, 0b0110: -2, 0b1101: 0.25, 0b1111: 3}
    for subset in range(16):
        assert coefficients[subset] == pytest.approx(expected.get(subset, 0), abs=1e-12)
    assert np.allclose(inverse_walsh_transform(coefficients[None])[0], f)


def test_vectorized_matches_reference_implementation():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(20, 16))
    fast = walsh_transform(values)
    for row in range(20):
        assert np.allclose(fast[row], reference_walsh(values[row]))
    assert np.allclose(walsh_transform(values[3], axis=0), reference_walsh(values[3]))


def test_scalar_and_multidimensional_transforms_agree():
    rng = np.random.default_rng(1)
    hidden = rng.normal(size=(7, 16, 5))
    transformed = walsh_transform(hidden)
    assert transformed.shape == (7, 16, 5)
    for dimension in range(5):                                 # each hidden dimension is a scalar landscape
        assert np.allclose(transformed[:, :, dimension], walsh_transform(hidden[:, :, dimension]))


def test_energy_by_order_parseval():
    rng = np.random.default_rng(2)
    coefficients = walsh_transform(rng.normal(size=(6, 16, 3)))
    raw, normalized = energy_by_order(coefficients)
    assert np.allclose(raw.sum(axis=1), (coefficients ** 2).sum(axis=(1, 2)))
    assert np.allclose(normalized[:, 1:].sum(axis=1), 1) and np.isnan(normalized[:, 0]).all()


def test_pooled_r2():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert pooled_r2(truth, truth) == 1.0
    assert pooled_r2(truth, np.full(4, truth.mean())) == pytest.approx(0.0)
    assert pooled_r2(truth, truth[::-1]) < 0                    # negative R^2 is preserved


# ---------------- amino-acid-identity-strict probe split ----------------
def test_identity_partition_properties():
    partition = identity_partition()
    assert identity_partition() == partition                   # deterministic
    for site, groups in enumerate(partition):
        assert [len(groups[g]) for g in ("train", "val", "test")] == [10, 4, 5]
        assert sorted(sum(groups.values(), [])) == sorted(alternatives_at(site))      # disjoint, covers 19 aa


def test_probe_split_assignment_rule():
    partition = identity_partition()
    aa = "ACDEFGHIKLMNPQRSTVWY"
    train_cube = [[aa.index(partition[s]["train"][0]) for s in range(4)]]
    mixed_cube = [[aa.index(partition[0]["train"][0])] + [aa.index(partition[s]["test"][0]) for s in range(1, 4)]]
    assert assign_probe_split(np.array(train_cube), partition)[0] == "train"
    assert assign_probe_split(np.array(mixed_cube), partition)[0] == "mixed"


# ---------------- tests on the real processed data ----------------
data = pytest.mark.skipif(not (PROCESSED_DIR / "cubes.npz").exists(), reason="run the data scripts first")


@data
def test_real_cubes_are_complete_and_consistent():
    import pandas as pd
    variants = pd.read_csv(PROCESSED_DIR / "variants.csv")
    cubes = np.load(PROCESSED_DIR / "cubes.npz")
    vertices, alternatives = cubes["vertices"], cubes["alternatives"]
    assert vertices.shape == (109_235, 16)
    for cube in [0, 1000, len(vertices) - 1]:                  # vertex strings match the construction rule
        alts = "".join("ACDEFGHIKLMNPQRSTVWY"[a] for a in alternatives[cube])
        assert [variants["variant"][i] for i in vertices[cube]] == cube_vertex_variants(alts)
    assert (variants["variant"][vertices[:, 0]] == WT_VARIANT).all()
    counts = {k: int((cubes["probe_split"] == k).sum()) for k in ["train", "val", "test", "mixed"]}
    assert counts == {"train": 7352, "val": 245, "test": 613, "mixed": 101_025}
    sets = {k: set(vertices[cubes["probe_split"] == k].ravel()) for k in ["train", "val", "test"]}
    assert sets["train"] & sets["val"] == sets["train"] & sets["test"] == {0}      # only wild type is shared


# ---------------- probing (analysis/accessibility.py) ----------------
def test_ridge_path_matches_sklearn():
    from sklearn.linear_model import Ridge
    from analysis.accessibility import ridge_path
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(200, 12)), rng.normal(size=200)
    for (weights, intercept), alpha in zip(ridge_path(x, y, [0.1, 10.0]), [0.1, 10.0]):
        reference = Ridge(alpha=alpha).fit(x, y)
        assert np.allclose(weights, reference.coef_) and intercept == pytest.approx(reference.intercept_)


def test_probe_recovers_linear_signal_and_not_noise():
    from analysis.accessibility import fit_probe
    rng = np.random.default_rng(4)
    w = rng.normal(size=8)
    x = {k: rng.normal(size=(n, 8)) for k, n in [("train", 400), ("val", 100), ("test", 200)]}
    y = {k: v @ w + 0.01 * rng.normal(size=len(v)) for k, v in x.items()}
    prediction, alpha, n_features = fit_probe(x["train"], y["train"], x["val"], y["val"], x["test"])
    assert pooled_r2(y["test"], prediction) > 0.99 and n_features == 8
    noise = rng.normal(size=200)                               # unrelated target: held-out R^2 must not be inflated
    prediction, _, _ = fit_probe(x["train"], rng.normal(size=400), x["val"], rng.normal(size=100), x["test"])
    assert pooled_r2(noise, prediction) < 0.1


def test_probe_ignores_constant_features_and_has_no_test_leakage():
    from analysis.accessibility import fit_probe
    rng = np.random.default_rng(5)
    x_train, x_val = rng.normal(size=(300, 4)), rng.normal(size=(60, 4))
    x_train[:, 2] = 7.0                                        # constant in training -> dropped
    y_train, y_val = x_train[:, 0] * 2, x_val[:, 0] * 2
    x_test = rng.normal(size=(50, 4))
    p1, _, n_features = fit_probe(x_train, y_train, x_val, y_val, x_test)
    p2, _, _ = fit_probe(x_train, y_train, x_val, y_val, np.vstack([x_test, 100 * rng.normal(size=(50, 4))]))
    assert n_features == 3
    assert np.allclose(p1, p2[:50])                            # extra (wild) test rows do not change earlier predictions
    zero, alpha, n = fit_probe(np.ones((10, 3)), np.arange(10.0), np.ones((4, 3)), np.arange(4.0), np.ones((5, 3)))
    assert n == 0 and np.isnan(alpha) and np.allclose(zero, np.concatenate([np.arange(10.0), np.arange(4.0)]).mean())


def test_accessibility_on_synthetic_cubes():
    """If the representation contains the measured Walsh coefficients, accessibility is ~1 at every order;
    if it is purely additive (no interactions), accessibility of orders >= 2 must not be positive."""
    from analysis.accessibility import accessibility_by_order
    rng = np.random.default_rng(6)
    sizes = {"train": 600, "val": 120, "test": 150}
    fitness = {k: rng.normal(size=(n, 16)) for k, n in sizes.items()}                 # random landscapes
    truth = {k: walsh_transform(v) for k, v in fitness.items()}
    informative = {k: np.repeat(truth[k][:, :, None], 3, axis=2) * [1.0, -2.0, 0.5] for k in sizes}
    assert all(row["accessibility_r2"] > 0.99 for row in accessibility_by_order(informative, truth))
    popcount = np.array([bin(v).count("1") for v in range(16)], dtype=float)
    additive_values = {k: popcount[None, :, None] * rng.normal(size=(n, 1, 1)) for k, n in sizes.items()}   # additive in the bits
    rep = {k: walsh_transform(v) for k, v in additive_values.items()}
    for row in accessibility_by_order(rep, truth):
        if row["order"] >= 2:
            assert row["accessibility_r2"] < 0.05


def test_cube_split_protocols():
    from analysis.accessibility import cube_split_indices, seen_variant_fraction
    labels = np.array(["train"] * 50 + ["val"] * 10 + ["test"] * 20 + ["mixed"] * 100)
    strict = cube_split_indices(labels, "strict")
    assert [len(strict[s]) for s in ("train", "val", "test")] == [50, 10, 20]
    shuffled = cube_split_indices(labels, "random")
    assert [len(shuffled[s]) for s in ("train", "val", "test")] == [50, 10, 20]
    assert set(np.concatenate(list(shuffled.values()))) == set(np.flatnonzero(labels != "mixed"))   # same pool, no mixed
    assert not (set(shuffled["train"]) & set(shuffled["test"]))                                      # cubes never shared
    vertices = np.zeros((80, 16), dtype=int)
    vertices[:, 1] = np.arange(80); vertices[:, 2] = np.arange(80) % 7 + 1000
    assert seen_variant_fraction(vertices, {"train": np.arange(50), "test": np.arange(60, 80)}) < 1
