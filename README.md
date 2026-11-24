# What does a protein language model learn about epistasis?

How do pretraining and capacity shape what ESM-2 represents about epistasis (interactions between mutations)? We extend
[ORBIT](https://arxiv.org/abs/2608.24953) (Rahimimovassagh et al., 2026) to ESM-2 on the GB1 four-site fitness landscape.
Authors: Arya Bhanushali, Aryan Thakkar. The original proposal is in [proposal.md](proposal.md).

## 1. Research questions (our expected outcomes are hypotheses, not results)

1. At identical architecture and parameter count, is **pretrained** ESM-2 150M more *interaction-accessible* than **randomly initialized** ESM-2 150M?
2. Does accessibility of higher-order epistasis grow with **capacity** (8M → 35M → 150M, pretrained)?
3. At which **layers** does accessibility for each interaction order (1–4) appear?

## 2. Method in plain language

* **Data.** Wu et al. (2016) measured fitness of 149,361 of the 20⁴ variants of protein GB1 at four positions (39, 40, 41, 54; wild type V-D-G-V).
* **Cube.** Pick one alternative amino acid per position. The 16 variants that use, at each position, either the wild-type or the chosen residue form a *cube*. 109,235 of the 19⁴ possible cubes have all 16 variants measured.
* **Walsh-Hadamard transform (WHT).** Rewrites 16 fitness values as 16 coefficients ε_S, one per subset S of positions: constant (order 0), single-position effects (order 1), pairwise (2), three-way (3), four-way (4). A nonzero order-≥2 coefficient is epistasis. The same transform applied to a model's 16 hidden vectors gives representation coefficients r_S.
* **Accessibility.** For layer ℓ and order k, a linear **ridge probe** predicts the measured ε_S from r_S. Accessibility is the **held-out R²** of that probe (can be negative = worse than predicting the mean).
* **Models.** `facebook/esm2_t6_8M_UR50D`, `..._t12_35M_UR50D`, `..._t30_150M_UR50D` (pretrained) and a randomly initialized 150M of identical architecture. Each is adapted to predict fitness with LoRA + a linear head on the official FLIP split; hidden states are cached before and after adaptation.

### One cube, worked example (real data: alternatives I, Y, C, A; cube 41744)

Vertex *v* is a 4-bit number; bit *i* = 1 means position *i* (39, 40, 41, 54) carries the alternative residue.

| v | bits (54,41,40,39) | variant | fitness |   | v | bits | variant | fitness |
|---|---|---|---|---|---|---|---|---|
| 0 | 0000 | VDGV (wild type) | 1.000 | | 8  | 1000 | VDGA | 1.373 |
| 1 | 0001 | IDGV | 1.446 | | 9  | 1001 | IDGA | 1.512 |
| 2 | 0010 | VYGV | 3.891 | | 10 | 1010 | VYGA | 2.406 |
| 3 | 0011 | IYGV | 4.388 | | 11 | 1011 | IYGA | 2.435 |
| 4 | 0100 | VDCV | 0.061 | | 12 | 1100 | VDCA | 3.687 |
| 5 | 0101 | IDCV | 0.111 | | 13 | 1101 | IDCA | 4.130 |
| 6 | 0110 | VYCV | 0.143 | | 14 | 1110 | VYCA | 5.360 |
| 7 | 0111 | IYCV | 0.189 | | 15 | 1111 | IYCA | 5.073 |

WHT with ε_S = (1/16) Σ_v f(v)·χ_S(v), χ_S(v) = Π_{i∈S} x_i(v), x_i = −1 for wild type and +1 for mutant:
constant 2.325; order 1: 39→0.085, 40→0.660, 41→0.019, 54→0.922; order 2 includes **41+54 → 1.297** (G41C alone is damaging, V54A is mild, together fitness is 3.7: epistasis);
order 3 largest: 40+41+54 → 0.396; order 4: −0.035. The 16 coefficients reproduce the 16 fitness values exactly.

## 3. Pipeline

```
data/prepare_data.py ──► data/processed/variants.csv ─┬─► data/build_cubes.py ─► data/processed/cubes.npz
 (download FLIP GB1, validate)                         │     (109,235 cubes + AA-identity-strict probe split)
                                                       │                         │
methods/additive_baseline.py ◄─────────────────────────┤                         │
 (verify WHT / reproduce ORBIT ridge baseline)         ▼                         │
methods/test_methods.py        models/train.py ─► checkpoints/<run>/{best,last}.pt, results/metrics/train_<run>.json
 (run BEFORE any model work)          │
                                      ▼
              models/extract_activations.py  --stage pre | post   ─► data/processed/activations/<condition>/layer_XX.npy
                                      │
                                      ▼
              analysis/accessibility.py  ─► results/metrics/accessibility_<condition>_<protocol>.csv
                                      │
                                      ▼
              analysis/visualize.py  ─► results/figures/*.png|pdf, results/metrics/summary_table.csv
```

## 4. Repository structure

| Path | Purpose |
|---|---|
| `methods/gb1.py` | Constants (positions, WT, paths) and the biological/Python/ESM-token indexing helpers |
| `methods/wht.py` | WHT for scalars and hidden vectors, order grouping, Walsh energy, pooled R² |
| `methods/additive_baseline.py` | Ridge-on-one-hot additive baseline + ORBIT verification experiment |
| `methods/test_methods.py` | 24 pytest tests (cubes, indexing/tokenization, WHT, probe, splits) |
| `data/prepare_data.py` | Download + validate FLIP GB1 data and official split |
| `data/build_cubes.py` | Build all complete cubes, identity-strict probe split, overlap statistics |
| `models/train.py` | LoRA (or optional full fine-tuning) + regression head; resumable |
| `models/extract_activations.py` | Cache per-layer representations for each unique variant; resumable |
| `analysis/accessibility.py` | Walsh-transform representations, ridge probes, accessibility CSVs |
| `analysis/visualize.py` | Figures and summary table |
| `checkpoints/` | Trained weights (git-ignored) · `results/metrics/` small CSV/JSON (committed) · `results/figures/` |

## 5. Setup

Python 3.11 is required. On Apple Silicon use a native arm64 Python (`uv python install 3.11` gives one; an x86_64 Python under Rosetta cannot install the pinned torch).

```bash
git clone https://github.com/aryabhanushali/esm2-epistasis.git && cd esm2-epistasis
python3.11 -m venv .venv && source .venv/bin/activate      # or: uv venv --python 3.11 .venv
pip install -r requirements.txt                              # CPU / Apple MPS (macOS) or CUDA wheels (Linux default)
```
On a CUDA server, if the default wheel does not match your driver, install torch first from https://pytorch.org (e.g. `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`) and then the rest of `requirements.txt`. The device is chosen automatically (CUDA > Apple MPS > CPU); bfloat16 autocast is used on CUDA only. Run all commands from the repository root. Model weights (~600 MB for 150M) are downloaded from Hugging Face on first use.

## 6. Running the project

**Step 0 – data and math checks (CPU, minutes; do this first).**
```bash
python -m data.prepare_data            # downloads FLIP files, validates, writes data/processed/variants.csv
python -m data.build_cubes             # 109,235 cubes; asserts ORBIT's counts and writes cubes.npz
python -m pytest methods/test_methods.py -q
python -m methods.additive_baseline    # verification experiment -> results/metrics/additive_baseline.json
```

**Step 1 – one experimental condition** (sizes `8M|35M|150M`, init `pretrained|random`, any seed):
```bash
python -m models.train --size 150M --init random --seed 42                      # LoRA + head; add --resume to continue
python -m models.extract_activations --size 150M --init random --stage pre  --seed 42   # saved random init weights
python -m models.extract_activations --size 150M --init random --stage post --seed 42   # after adaptation
python -m analysis.accessibility --size 150M --init random --stage pre  --seed 42
python -m analysis.accessibility --size 150M --init random --stage post --seed 42
```
For a pretrained model the `pre` stage is the unmodified Hugging Face model and does not depend on the seed.

**Step 2 – all four primary conditions, three seeds** (this is the full experiment; run on a GPU server):
```bash
for cfg in "8M pretrained" "35M pretrained" "150M pretrained" "150M random"; do set -- $cfg
  for seed in 42 43 44; do
    python -m models.train --size $1 --init $2 --seed $seed
    for stage in pre post; do
      python -m models.extract_activations --size $1 --init $2 --stage $stage --seed $seed
      python -m analysis.accessibility     --size $1 --init $2 --stage $stage --seed $seed
    done
  done
done
python -m analysis.visualize            # figures for stage 'post'; add --stage pre for representations before adaptation
```
Everything is resumable: `train` with `--resume`; `extract_activations` and `accessibility` skip completed rows/layers.

**Optional.** `--pooling mean` (extract + accessibility + visualize) for mean-pooled representations; `--protocol random` (accessibility) for the leaky random-cube sensitivity analysis; `--full-finetune --tag _fullft` (train, and extract with the same flags) for the full-fine-tuning control.

**Smoke test (8M, minutes on a laptop):** `python -m models.train --size 8M --init pretrained --max-samples 64 --epochs 2 --tag _smoke`, then extract with `--tag _smoke --stage post --max-variants 300`. Outputs tagged `_smoke` are ignored by the figure script unless `--include-smoke`.

## 7. Data facts we verified (see `results/metrics/dataset_summary.txt`, `cube_summary.json`)

* 149,361 measured variants (of 160,000); no missing values or duplicates; wild type VDGV at fitness 1.0; fitness 0–8.76 with 29,477 exact zeros. Fitness is **used untransformed** (FLIP's `target` equals `Fitness`).
* FLIP's `sequence` column is the 265-residue 5LDE construct; positions 39/40/41/54 (1-indexed) match all 149,361 variant strings. We feed ESM-2 only the **56-residue GB1 domain** (residues 1–56 of that sequence) – a deliberate choice for ~5× lower cost; the mutated residues are unchanged. ESM-2 token index of residue *p* is *p* (CLS at index 0); tested.
* **Official FLIP `two_vs_rest`** covers only FLIP's filtered subset of 8,733 variants (not all 149,361): **train 381, validation 43** (FLIP's own validation flag, preserved; the 43 are the only model-selection data), **test 8,309** (3- and 4-mutants). The other 140,628 variants are unassigned (mostly near-zero fitness) and are used only through cubes.
* 109,235 complete cubes (matches the proposal and ORBIT).

## 8. Interaction-accessibility protocol (what is reproduced from ORBIT, and how it was verified)

Verified against ORBIT's **official code** (`maryam-rahimi/ORBIT-GB1`, which we read; we re-implemented, not copied) and the paper:

| Item | Implementation | Verification |
|---|---|---|
| Vertex order, ±1 coding | bit *i* ↔ position *i*; bit 0 → −1, bit 1 → +1 | official code; unit tests |
| WHT normalization | ε_S = (1/16)Σ f(v)χ_S(v) | paper eq. 6 + code; reconstruction/orthogonality tests |
| Probe | one shared ridge per (layer, order), pooled over cubes **and** subsets of that order; α grid 1e-6…1e3 chosen on probe-validation only; near-constant features dropped; standardization with training statistics; refit on train+val; pooled R² on test | official code |
| Probe split | AA-identity-strict: per position the 19 alternatives are permuted by `numpy.default_rng(42)` (one RNG, positions 39→54) and split 10/4/5; a cube is train/val/test only if all four alternatives fall in the same group, else "mixed" (discarded) | official code; **our counts match exactly: 7,352 / 245 / 613 / 101,025 mixed** |
| Additive baseline | ridge on one-hot residues | see §9 |

The probe is solved in closed form (one eigendecomposition per order) instead of calling sklearn 10×; test `test_ridge_path_matches_sklearn` checks equality.
**Leakage.** Probe train/val/test cubes share only the wild-type variant (verified, `cube_summary.json`), but ~11% of probe-test variants (146/1,282) are also FLIP-*test* variants and ~4% FLIP-train/validation (49/1,282), so a model fine-tuned on FLIP-train has seen few of them – this is ORBIT's own caveat. The `random` protocol (same 8,210 cubes, random assignment) shares many variants (in our 8M smoke run 81% of test-cube variants also appeared in probe-train cubes) and is reported only as a clearly labelled sensitivity analysis. Held-out R² is never replaced by training R²; negative values are kept.

## 9. Verification results (reproduced vs. paper-reported)

`python -m methods.additive_baseline` (ridge strength chosen on the 43 FLIP validation variants → α = 1). Paper values are as extracted from the ORBIT arXiv HTML (Tables 1 and 3, ridge row); please confirm them against the PDF.

| Quantity | Ours | Paper |
|---|---|---|
| Order 2 / 3 / 4 functional recovery R²ₖ (613 probe-test cubes) | −0.0012 / −0.0147 / −0.0218 | −0.0012 / −0.0147 / −0.0218 |
| Order 1 functional recovery R²₁ | 0.309 | 0.4126 |
| FLIP test Spearman / R² / RMSE | 0.594 / −0.130 / 1.308 | 0.5617 / 0.1084 / 1.1619 |
| max abs order 2–4 coefficient of additive predictions | 3e-16 | (should be 0) |

The order ≥ 2 agreement to four decimals confirms our cube construction, probe split and WHT conventions. The paper does not state its ridge α or exact training set; as a **post-hoc diagnostic only** (it uses test metrics, so it is not used anywhere), α ≈ 3 fit on all 424 FLIP training variants gives R² 0.100, Spearman 0.564, R²₁ 0.409, close to the paper. Our reported baseline keeps the validation-selected α.

## 10. Assumptions, deviations and limitations

* **Pooling.** Primary representation = concatenated hidden states at the 4 mutated positions (the same features the regression head reads; precedent: ORBIT-FMIB's ESM-2 analysis). ORBIT itself analysed small MLPs, not ESM-2, so there are **no published ESM-2 ORBIT numbers to reproduce**. Mean pooling is an optional secondary experiment.
* **Layers.** Index 0 = embeddings, 1..L = transformer layers. HF applies ESM-2's final LayerNorm to layer L only. Different depths are compared by relative depth (layer/L) and absolute index.
* **LoRA on a frozen random backbone is not equivalent to training a random network end-to-end**: the random weights stay fixed and only low-rank attention updates (query/key/value, rank 8) and the head adapt. A weak result for random init may reflect this limitation rather than the absence of pretraining. `--full-finetune` is provided as an optional control and is not part of the primary experiment. The initial random weights are saved (`checkpoints/<run>/initial_backbone.pt`) so before/after comparisons use identical initialization; matched pretrained/random runs also share LoRA/head initialization, data order, hyperparameters and seeds (initialization equality was checked with an ad-hoc script, not a unit test).
* **Hyperparameters are not tuned** (lr 1e-3, 30 epochs, batch 16, LoRA r=8/α=16): the validation set has only 43 variants, which is too small for reliable tuning. It is used only for checkpoint selection. Test data is evaluated once, at the end.
* The proposal says "train on ≤2 mutations from the 149,361 variants"; the official split trains on only the 424 FLIP-filtered ≤2-mutants (381 + 43).
* The 56-residue input (vs FLIP's 265-residue construct) is our choice. Fitness is raw and heavy-tailed; MSE on raw fitness is used.
* ORBIT-FMIB (a related paper using ESM-2) reports that its conclusions were unstable under replication; effects here may be small and sensitive to protocol, so we report all seeds and keep unfavorable results.
* Not verified: ESM-2 may have seen GB1 homologs in pretraining (confound for question 1).

## 11. Compute estimates (estimates unless marked measured)

* **Measured (Mac, MPS):** 8M extraction of 13,150 variants 25 s, 8M accessibility for 7 layers 58 s, 150M LoRA step on 64 samples ≈4 s/epoch, trainable parameters 924,161 of 149,309,402 (identical for pretrained and random).
* **Disk (fp16 'sites' activations):** 8M 0.23 GB, 35M ≈0.66 GB, 150M ≈2.1 GB per condition; the full plan (4 models × pre + post × 3 seeds) ≈ 20 GB in `data/processed/activations/`.
* **Memory:** training 150M needs a few GB of GPU memory (batch 16, length 58); accessibility for 150M peaks at roughly 6–8 GB of RAM per layer (only one layer is in memory).
* **Time:** training minutes per run; accessibility for 150M estimated tens of minutes per condition on CPU (dominated by 2560×2560 eigendecompositions).

## 12. Status checklist

| Component | Implemented | Tested | Run on real experiment |
|---|---|---|---|
| Data download/validation, cubes, probe split | ✅ | ✅ (counts match ORBIT) | ✅ |
| WHT, additive baseline | ✅ | ✅ 24 unit tests pass | ✅ (§9) |
| `models/train.py` (LoRA, resume, random init) | ✅ | ✅ 8M and 150M smoke runs (64 samples) | ⬜ |
| `models/extract_activations.py` | ✅ | ✅ numerically checked against independent forward pass | ⬜ |
| `analysis/accessibility.py` | ✅ | ✅ synthetic cubes + end-to-end 8M smoke | ⬜ |
| `analysis/visualize.py` | ✅ | ✅ rendered on smoke data (discarded) | ⬜ |
| Four primary conditions × 3 seeds, `pre` and `post` | – | – | ⬜ **not run** |
| Mean-pooling, random-split sensitivity, full-fine-tuning control | ✅ (flags) | partly (random split on 8M smoke) | ⬜ |

No ESM-2 accessibility result is reported in this repository yet; `results/figures/` currently contains only the additive-baseline verification figure.

## 13. References

1. Rahimimovassagh, Garibay, Yousefi (2026). Beyond tokens: probing higher-order epistasis in learned protein representations. arXiv:2608.24953. Code: github.com/maryam-rahimi/ORBIT-GB1.
2. Wu et al. (2016). Adaptation in protein fitness landscapes is facilitated by indirect paths. eLife 5:e16965.
3. Dallago et al. (2021). FLIP: benchmark tasks in fitness landscape inference for proteins. NeurIPS Datasets & Benchmarks. Data: github.com/J-SNACKKB/FLIP (`splits/gb1`).
4. Simon, Zou (2024). InterPLM. arXiv:2412.12101.
5. Ahlmann-Eltze, Huber, Anders (2025). Nature Methods 22(8):1657–1661.
6. Wittmann, Yue, Arnold (2021). Cell Systems 12(11):1026–1045.
7. Related: ORBIT-FMIB, arXiv:2610.00672 (ESM-2 with mutated-site representations).
