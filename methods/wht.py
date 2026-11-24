"""Walsh-Hadamard transform (WHT) over a 4-dimensional mutation cube.

Conventions (identical to the official ORBIT code, ORBIT-GB1 `transform.py` / accessibility script):
  * A cube has 16 vertices, indexed v = 0..15. Bit i of v says whether site i
    (i = 0,1,2,3 <-> GB1 positions 39,40,41,54) carries the mutant (1) or wild-type (0) residue.
    So v=0 is wild type, v=15 is the quadruple mutant, v=5 (0b0101) mutates sites 0 and 2.
  * Walsh coordinates use x_i = -1 for wild type (bit 0) and +1 for mutant (bit 1).
  * A subset S of sites is also a 4-bit mask s = 0..15; its character is chi_S(x) = prod_{i in S} x_i.
  * Coefficients: eps_S = (1/16) * sum_v f(v) * chi_S(v)        (so f(v) = sum_S eps_S * chi_S(v)).
  * The interaction order of S is |S| = popcount(s):
      order 0 constant, order 1 single-site effects, order 2 pairwise, order 3 three-way, order 4 four-way.
"""
import numpy as np

N_SITES = 4
N_VERTICES = 2 ** N_SITES


def _build_walsh_matrix():
    """H[v, s] = chi_s(v). Rows: cube vertices, columns: subsets."""
    matrix = np.ones((N_VERTICES, N_VERTICES))
    for vertex in range(N_VERTICES):
        for subset in range(N_VERTICES):
            for site in range(N_SITES):
                if (subset >> site) & 1:
                    matrix[vertex, subset] *= 1.0 if (vertex >> site) & 1 else -1.0
    return matrix


WALSH_MATRIX = _build_walsh_matrix()
SUBSET_ORDERS = np.array([bin(s).count("1") for s in range(N_VERTICES)])    # popcount of each subset mask


def walsh_transform(values, axis=1):
    """WHT along the vertex axis (length 16). Works for scalars and hidden vectors alike.

    values: (N, 16) fitness landscapes -> (N, 16) coefficients, or
            (N, 16, P) representations  -> (N, 16, P) coefficients (use axis=1), or a single (16,) with axis=0.
    The output axis indexes subsets (mask s), not vertices.
    """
    values = np.asarray(values, dtype=np.float64)
    coefficients = np.tensordot(values, WALSH_MATRIX, axes=([axis], [0]))     # new axis is last
    return np.moveaxis(coefficients, -1, axis) / N_VERTICES


def inverse_walsh_transform(coefficients, axis=1):
    """Rebuild vertex values from coefficients: f(v) = sum_S eps_S chi_S(v)."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    values = np.tensordot(coefficients, WALSH_MATRIX, axes=([axis], [1]))
    return np.moveaxis(values, -1, axis)


def subsets_of_order(order):
    """Subset masks with |S| = order, e.g. order 2 -> the 6 site pairs."""
    return np.flatnonzero(SUBSET_ORDERS == order)


def subset_label(mask, positions=(39, 40, 41, 54)):
    """Readable name of a subset, e.g. mask 5 -> '39+41'; mask 0 -> 'constant'."""
    sites = [str(positions[i]) for i in range(N_SITES) if (mask >> i) & 1]
    return "+".join(sites) if sites else "constant"


def energy_by_order(coefficients):
    """Walsh energy per interaction order of representation coefficients.

    coefficients: (N, 16, P). Returns raw energies (N, 5) = sum_{|S|=k} ||r_S||^2 and the
    normalized spectrum (N, 5) = fraction of non-constant energy at each order (order 0 is
    excluded from the denominator, as in ORBIT; its normalized column is NaN).
    """
    squared_norm = (coefficients ** 2).sum(axis=2)                       # (N, 16)
    raw = np.stack([squared_norm[:, subsets_of_order(k)].sum(axis=1) for k in range(5)], axis=1)
    total = raw[:, 1:].sum(axis=1, keepdims=True)
    normalized = np.full_like(raw, np.nan)
    valid = total[:, 0] > 1e-15
    normalized[valid, 1:] = raw[valid, 1:] / total[valid]
    return raw, normalized


def pooled_r2(truth, prediction):
    """R^2 = 1 - SSE/SST with SST around the mean of `truth` (pooled over all entries).
    Can be negative. This is the pooled R^2 used by ORBIT for accessibility and functional recovery."""
    truth = np.asarray(truth, dtype=np.float64).ravel()
    prediction = np.asarray(prediction, dtype=np.float64).ravel()
    total = np.sum((truth - truth.mean()) ** 2)
    if total <= 1e-15:
        return float("nan")
    return float(1.0 - np.sum((truth - prediction) ** 2) / total)
