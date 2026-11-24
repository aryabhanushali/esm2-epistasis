"""Download and validate the GB1 four-site fitness landscape and the official FLIP splits.

Usage (from the repository root):
    python -m data.prepare_data

Outputs:
    data/raw/flip_gb1/...                      downloaded FLIP files (not committed)
    data/processed/variants.csv                one row per measured variant
    results/metrics/dataset_summary.txt        short dataset statistics (committed)

Source: FLIP repository, splits/gb1 (Wu et al. 2016 data + FLIP split columns).
Fitness is used exactly as measured (no transformation); FLIP's `target` equals `Fitness`.
"""
import io
import json
import zipfile

import pandas as pd
import requests

from methods.gb1 import (AMINO_ACIDS, METRICS_DIR, POSITIONS, PROCESSED_DIR, RAW_DIR,
                         WT_SEQUENCE, WT_VARIANT)

FLIP_URL = "https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/gb1/"
FLIP_DIR = RAW_DIR / "flip_gb1"
EXPECTED_VARIANTS = 149_361          # proposal / ORBIT
EXPECTED_FLIP_COUNTS = {"train": 381, "val": 43, "test": 8309}   # observed in the official file


def download_flip_files():
    """Download the two FLIP zip files once and extract them into data/raw/flip_gb1."""
    FLIP_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["four_mutations_full_data.csv.zip", "splits.zip"]:
        if not (FLIP_DIR / name).exists():
            print("downloading", name)
            response = requests.get(FLIP_URL + name, timeout=120)
            response.raise_for_status()
            (FLIP_DIR / name).write_bytes(response.content)
        with zipfile.ZipFile(FLIP_DIR / name) as archive:
            archive.extractall(FLIP_DIR)


def assign_split(row):
    """Map FLIP's two_vs_rest columns to train / val / test / unassigned."""
    if row["two_vs_rest"] == "test":
        return "test"
    if row["two_vs_rest"] == "train":
        return "val" if row["two_vs_rest_validation"] is True or row["two_vs_rest_validation"] == "True" else "train"
    return "unassigned"           # measured variants outside FLIP's filtered subset


def validate(full):
    """Check missing values, duplicates, alphabet, wild type and sequence reconstruction."""
    problems = []
    assert len(full) == EXPECTED_VARIANTS, f"expected {EXPECTED_VARIANTS} variants, found {len(full)}"
    assert full[["Variants", "HD", "Fitness", "sequence"]].notna().all().all(), "missing values"
    assert not full["Variants"].duplicated().any(), "duplicate variant identifiers"
    assert not full["sequence"].duplicated().any(), "duplicate sequences"
    assert (full["Variants"].str.len() == 4).all()
    assert set("".join(full["Variants"])) <= set(AMINO_ACIDS), "unexpected characters"
    assert (full["Fitness"] >= 0).all(), "negative fitness"
    assert full["Variants"].iloc[0] == WT_VARIANT and full["Fitness"].iloc[0] == 1.0, "WT row not found / WT fitness != 1"
    # HD must equal the number of positions that differ from wild type
    hd = full["Variants"].apply(lambda v: sum(a != b for a, b in zip(v, WT_VARIANT)))
    assert (hd == full["HD"]).all(), "HD column disagrees with variant string"
    # FLIP sequence must be the 5LDE construct with only the 4 sites changed; its first
    # 56 residues must equal our GB1 domain (with sites substituted)
    wt_full = full["sequence"].iloc[0]
    assert wt_full[:len(WT_SEQUENCE)] == WT_SEQUENCE, "GB1 domain is not the first 56 residues of FLIP WT sequence"
    for variant, sequence in zip(full["Variants"], full["sequence"]):
        site_residues = "".join(sequence[p - 1] for p in POSITIONS)      # biological -> Python index
        if site_residues != variant:
            problems.append(variant)
        elif sequence[:38] != wt_full[:38] or sequence[54:] != wt_full[54:]:
            problems.append(variant)
    assert not problems, f"{len(problems)} variants whose sequence disagrees with their identifier"
    # count how many measured variants exist of the 20^4 possible
    return {"possible_variants": 20 ** 4, "measured_variants": len(full)}


def main():
    download_flip_files()
    full = pd.read_csv(FLIP_DIR / "four_mutations_full_data.csv", low_memory=False)
    info = validate(full)

    # the standalone split file must agree with the split columns of the full file
    flip_split = pd.read_csv(FLIP_DIR / "splits" / "two_vs_rest.csv")
    full["split"] = full.apply(assign_split, axis=1)
    merged = flip_split.merge(full[["sequence", "Fitness", "split"]], on="sequence", how="left")
    assert len(merged) == len(flip_split) and merged["Fitness"].notna().all()
    assert (merged["target"] - merged["Fitness"]).abs().max() < 1e-9, "FLIP target != Fitness"
    assert (merged["set"].eq("test") == merged["split"].eq("test")).all()
    counts = full["split"].value_counts().to_dict()
    assert {k: counts[k] for k in EXPECTED_FLIP_COUNTS} == EXPECTED_FLIP_COUNTS, counts

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    variants = pd.DataFrame({
        "variant": full["Variants"],
        "n_mutations": full["HD"],
        "fitness": full["Fitness"],
        "split": full["split"],            # official FLIP two_vs_rest (+ official validation flag)
    })
    variants.to_csv(PROCESSED_DIR / "variants.csv", index_label="variant_index")

    # ---- short summary ----
    lines = [f"GB1 four-site landscape: {len(full)} measured of {info['possible_variants']} possible variants",
             "wild type VDGV at positions 39/40/41/54 verified (fitness 1.0); no missing values or duplicates",
             "fitness: untransformed; min %.4g median %.4g mean %.4g max %.4g; %d exact zeros"
             % (full.Fitness.min(), full.Fitness.median(), full.Fitness.mean(), full.Fitness.max(), (full.Fitness == 0).sum()),
             "", "variants by number of mutations:"]
    lines += [f"  {k}: {v}" for k, v in full["HD"].value_counts().sort_index().items()]
    lines += ["", "official FLIP two_vs_rest split (train = WT + singles + doubles measured in FLIP's filtered set):"]
    split_table = pd.crosstab(full["HD"], full["split"])
    lines += split_table.to_string().split("\n")
    lines += ["", "mean fitness by split: " + json.dumps(variants.groupby("split")["fitness"].mean().round(4).to_dict())]
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "dataset_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
