"""Build every complete 4-site mutation cube of GB1 and the amino-acid-identity-strict probe split.

Usage (from the repository root):
    python -m data.build_cubes

A cube is defined by choosing ONE alternative (non-wild-type) amino acid per site (19^4 = 130,321
candidates). Its 16 vertices are the combinations of {wild type, chosen alternative} at each site,
ordered as described in methods/wht.py (bit i of the vertex index <-> site i). A cube is kept only if all
16 vertices were measured (ORBIT reports 109,235).

Probe split (ORBIT 'AA-identity-strict'; official code ORBIT-GB1 `gb1_leakage_safe_split.py`, seed 42):
at every site, the 19 alternatives are shuffled and divided 10 / 4 / 5 into probe-train / probe-val /
probe-test identities. A cube is assigned to a split only if all four of its alternatives belong to the
same split; other ('mixed') cubes are not used. Expected counts: 7,352 / 245 / 613 (101,025 mixed).

Outputs:
    data/processed/cubes.npz              vertices (N,16) int32 indices into variants.csv,
                                          alternatives (N,4) int8 indices into AMINO_ACIDS,
                                          probe_split (N,) in {train,val,test,mixed}
    results/metrics/cube_summary.json     counts, the amino-acid partition and variant overlap
"""
import itertools
import json

import numpy as np
import pandas as pd

from methods.gb1 import AMINO_ACIDS, METRICS_DIR, PROCESSED_DIR, WT_VARIANT, alternatives_at

SPLIT_SEED = 42
SPLIT_SIZES = {"train": 10, "val": 4, "test": 5}
EXPECTED = {"complete": 109_235, "mixed": 101_025, "train": 7_352, "val": 245, "test": 613}


def cube_vertex_variants(alternatives):
    """16 variant strings of a cube; vertex v mutates site i iff bit i of v is set."""
    vertices = []
    for vertex in range(16):
        residues = [alternatives[i] if (vertex >> i) & 1 else WT_VARIANT[i] for i in range(4)]
        vertices.append("".join(residues))
    return vertices


def build_complete_cubes(variant_to_index):
    """Enumerate all 19^4 candidate cubes and keep those whose 16 vertices were all measured."""
    all_vertices, all_alternatives = [], []
    for combo in itertools.product(*[alternatives_at(site) for site in range(4)]):
        indices = [variant_to_index.get(v) for v in cube_vertex_variants(combo)]
        if None not in indices:
            all_vertices.append(indices)
            all_alternatives.append([AMINO_ACIDS.index(aa) for aa in combo])
    return np.array(all_vertices, dtype=np.int32), np.array(all_alternatives, dtype=np.int8)


def identity_partition(seed=SPLIT_SEED):
    """Per-site split of the 19 alternatives into train/val/test identities (same RNG use as ORBIT)."""
    rng = np.random.default_rng(seed)
    partition = []
    for site in range(4):                                  # one shared RNG, sites in order 39,40,41,54
        shuffled = rng.permutation(alternatives_at(site)).tolist()
        n_train, n_val = SPLIT_SIZES["train"], SPLIT_SIZES["val"]
        partition.append({"train": shuffled[:n_train], "val": shuffled[n_train:n_train + n_val],
                          "test": shuffled[n_train + n_val:]})
    return partition


def assign_probe_split(alternatives, partition):
    """'train'/'val'/'test' if all four alternatives lie in the same identity group, else 'mixed'."""
    group_of = [{aa: g for g, aas in site.items() for aa in aas} for site in partition]
    labels = []
    for row in alternatives:
        groups = {group_of[site][AMINO_ACIDS[aa]] for site, aa in enumerate(row)}
        labels.append(groups.pop() if len(groups) == 1 else "mixed")
    return np.array(labels)


def variant_overlap(vertices, probe_split, variants):
    """How many distinct variants (incl. wild type) are shared between probe splits and with FLIP splits."""
    members = {name: set(vertices[probe_split == name].ravel()) for name in ["train", "val", "test"]}
    overlap = {"unique_variants": {k: len(v) for k, v in members.items()}}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = members[a] & members[b]
        overlap[f"shared_{a}_{b}"] = len(shared)
        overlap[f"shared_{a}_{b}_excluding_wild_type"] = len(shared - {0})      # variant index 0 = wild type
    flip = variants["split"].to_numpy()
    for name, ids in members.items():
        counts = pd.Series(flip[sorted(ids)]).value_counts().to_dict()
        overlap[f"probe_{name}_variants_by_flip_split"] = {k: int(v) for k, v in counts.items()}
    return overlap


def main():
    variants = pd.read_csv(PROCESSED_DIR / "variants.csv")
    assert variants["variant"].iloc[0] == WT_VARIANT
    variant_to_index = {v: i for i, v in enumerate(variants["variant"])}

    vertices, alternatives = build_complete_cubes(variant_to_index)
    partition = identity_partition()
    probe_split = assign_probe_split(alternatives, partition)
    counts = {k: int((probe_split == k).sum()) for k in ["train", "val", "test", "mixed"]}
    counts["complete"] = len(vertices)

    # sanity checks against the numbers reported by ORBIT (printed, and asserted)
    for key, expected in EXPECTED.items():
        status = "OK" if counts[key] == expected else "MISMATCH"
        print(f"{key:>9}: {counts[key]:>7} (ORBIT: {expected}) {status}")
    assert all(counts[k] == v for k, v in EXPECTED.items()), "cube counts differ from ORBIT"
    assert (vertices[:, 0] == 0).all()          # vertex 0 of every cube is the wild type (variant index 0)

    np.savez_compressed(PROCESSED_DIR / "cubes.npz", vertices=vertices, alternatives=alternatives, probe_split=probe_split)
    summary = {"counts": counts, "amino_acid_partition_by_site": dict(zip(["V39", "D40", "G41", "V54"], partition)),
               "split_seed": SPLIT_SEED, "overlap": variant_overlap(vertices, probe_split, variants)}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "cube_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overlap"], indent=2))


if __name__ == "__main__":
    main()
