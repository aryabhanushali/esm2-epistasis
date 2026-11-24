"""Shared GB1 constants and paths. Everything that depends on GB1 indexing lives here.

Indexing conventions (a common source of bugs):
  * Biological positions 39, 40, 41, 54 are 1-indexed residue numbers of the GB1 domain.
  * Python string index of biological position p is p - 1.
  * ESM-2 tokenizes "<cls> R1 R2 ... RL <eos>", so residue p sits at token index p
    (the <cls> token occupies index 0). Hence the token index equals the biological position.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
POSITIONS = (39, 40, 41, 54)          # biological, 1-indexed
WT_VARIANT = "VDGV"                   # wild-type residues at POSITIONS

# 56-residue GB1 domain (residues 1-56 of the 5LDE chain used by FLIP). FLIP's own
# `sequence` column is the full 265-residue 5LDE construct (GB1 + tags + other domain);
# we feed only the GB1 domain to ESM-2. prepare_data.py verifies this matches FLIP.
WT_SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

for _position, _residue in zip(POSITIONS, WT_VARIANT):
    assert WT_SEQUENCE[_position - 1] == _residue, "wild-type sequence / position mismatch"


def variant_to_sequence(variant):
    """'ADGV' -> 56-residue GB1 sequence carrying those residues at positions 39,40,41,54."""
    residues = list(WT_SEQUENCE)
    for position, residue in zip(POSITIONS, variant):
        residues[position - 1] = residue
    return "".join(residues)


def esm_token_index(position):
    """Token index of biological position `position` in an ESM-2 input (<cls> is index 0)."""
    return position


def alternatives_at(site):
    """Non-wild-type amino acids at mutation site `site` (0..3), alphabetical order."""
    return [aa for aa in AMINO_ACIDS if aa != WT_VARIANT[site]]
