# What does a protein language model learn about epistasis?

We want to know how pretraining and model size affect what ESM-2 represents about epistasis (interactions between mutations). The setup follows ORBIT (Rahimimovassagh et al., 2026, arXiv:2608.24953) and applies it to ESM-2 on the GB1 four-site fitness landscape. Authors: Arya Bhanushali and Aryan Thakkar. The original proposal is in [proposal.md](proposal.md).

## Research questions

The outcomes we expected in the proposal are hypotheses. We have not tested them yet.

1. With the same architecture and parameter count, is pretrained ESM-2 150M more interaction-accessible than a randomly initialized ESM-2 150M?
2. Does accessibility of higher-order epistasis grow with model size (8M, 35M, 150M, all pretrained)?
3. At which layers does accessibility appear for each interaction order (1 to 4)?

## Method

Data: Wu et al. (2016) measured the fitness of 149,361 of the 160,000 possible variants of protein GB1 at four positions (39, 40, 41, 54). The wild type at these positions is V, D, G, V.

Cube: pick one alternative amino acid for each of the four positions. The 16 variants that use either the wild-type residue or the chosen residue at each position form a cube. Of the 130,321 possible cubes, 109,235 have all 16 variants measured.

Walsh-Hadamard transform (WHT): rewrites the 16 fitness values of a cube as 16 coefficients, one for each subset of positions. The constant is order 0, single-position effects are order 1, then pairwise (2), three-way (3) and four-way (4). A nonzero coefficient of order 2 or more is epistasis. Applying the same transform to a model's 16 hidden vectors gives the representation coefficients.

Accessibility: for a layer and an interaction order, a linear ridge probe predicts the measured fitness coefficients from the representation coefficients. Accessibility is the R squared of that probe on held-out cubes. It can be negative, which means the probe does worse than predicting the mean.

Models: `facebook/esm2_t6_8M_UR50D`, `facebook/esm2_t12_35M_UR50D` and `facebook/esm2_t30_150M_UR50D` (pretrained), plus a randomly initialized 150M model with the same architecture. Each model is adapted to predict fitness with LoRA and a linear head on the official FLIP split. Hidden states are cached before and after adaptation.

### Worked example (real data, cube 41744, alternatives I, Y, C, A)

Vertex v is a 4-bit number. Bit i is 1 when position i (39, 40, 41, 54 for bits 0 to 3) carries the alternative residue.

| v | bits | variant | fitness | v | bits | variant | fitness |
|---|---|---|---|---|---|---|---|
| 0 | 0000 | VDGV (wild type) | 1.000 | 8 | 1000 | VDGA | 1.373 |
| 1 | 0001 | IDGV | 1.446 | 9 | 1001 | IDGA | 1.512 |
| 2 | 0010 | VYGV | 3.891 | 10 | 1010 | VYGA | 2.406 |
| 3 | 0011 | IYGV | 4.388 | 11 | 1011 | IYGA | 2.435 |
| 4 | 0100 | VDCV | 0.061 | 12 | 1100 | VDCA | 3.687 |
| 5 | 0101 | IDCV | 0.111 | 13 | 1101 | IDCA | 4.130 |
| 6 | 0110 | VYCV | 0.143 | 14 | 1110 | VYCA | 5.360 |
| 7 | 0111 | IYCV | 0.189 | 15 | 1111 | IYCA | 5.073 |

The coefficient for a subset S is (1/16) times the sum over vertices of f(v) times the product of x_i for i in S, where x_i is -1 for the wild-type residue and +1 for the mutant. For this cube:

- constant: 2.325
- order 1: position 39 is 0.085, 40 is 0.660, 41 is 0.019, 54 is 0.922
- order 2: the largest is positions 41 and 54 at 1.297. G41C alone is damaging (0.061) and V54A alone is mild (1.373), but together fitness is 3.687.
- order 3: the largest is positions 40, 41 and 54 at 0.396
- order 4: -0.035

The 16 coefficients reproduce the 16 fitness values exactly.

## Pipeline

```
data/prepare_data.py -> data/processed/variants.csv
        |
        v
data/build_cubes.py -> data/processed/cubes.npz
        |                (109,235 cubes and the probe split)
        |
methods/additive_baseline.py   checks the WHT code against ORBIT
methods/test_methods.py        run these tests before any model work
        |
        v
models/train.py -> checkpoints/<run>/, results/metrics/train_<run>.json
        |
        v
models/extract_activations.py (stage pre or post) -> data/processed/activations/<condition>/
        |
        v
analysis/accessibility.py -> results/metrics/accessibility_<condition>_<protocol>.csv
        |
        v
analysis/visualize.py -> results/figures/, results/metrics/summary_table.csv
```

## Repository structure

| Path | Purpose |
|---|---|
| `methods/gb1.py` | Constants (positions, wild type, paths) and the indexing helpers |
| `methods/wht.py` | WHT for scalars and hidden vectors, grouping by order, Walsh energy, pooled R squared |
| `methods/additive_baseline.py` | Additive ridge baseline and the ORBIT verification experiment |
| `methods/test_methods.py` | 24 pytest tests for cubes, indexing, tokenization, WHT, the probe and the splits |
| `data/prepare_data.py` | Downloads and validates the FLIP GB1 data and the official split |
| `data/build_cubes.py` | Builds all complete cubes and the probe split, and counts shared variants |
| `models/train.py` | LoRA (or optional full fine-tuning) plus a regression head, resumable |
| `models/extract_activations.py` | Caches per-layer representations for each unique variant, resumable |
| `analysis/accessibility.py` | Walsh-transforms the representations and fits the ridge probes |
| `analysis/visualize.py` | Figures and the summary table |
| `checkpoints/` | Trained weights (not tracked by git) |
| `results/metrics/`, `results/figures/` | Small CSV and JSON files and figures (tracked) |

## Setup

Python 3.11 is required. On Apple Silicon you need a native arm64 Python. `uv python install 3.11` gives you one. An x86_64 Python running under Rosetta cannot install the pinned torch version.

```bash
git clone https://github.com/aryabhanushali/esm2-epistasis.git
cd esm2-epistasis
python3.11 -m venv .venv && source .venv/bin/activate    # or: uv venv --python 3.11 .venv
pip install -r requirements.txt
```

On a CUDA server, if the default torch wheel does not match the driver, install torch first from https://pytorch.org (for example `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`) and then install the rest of `requirements.txt`.

The device is picked automatically (CUDA, then Apple MPS, then CPU). bfloat16 autocast is used on CUDA only. Run every command from the repository root. The ESM-2 weights download from Hugging Face on first use (about 600 MB for the 150M model).

## Running the project

Step 0, data and checks (CPU, a few minutes). Do this first.

```bash
python -m data.prepare_data
python -m data.build_cubes
python -m pytest methods/test_methods.py -q
python -m methods.additive_baseline
```

Step 1, one experimental condition. Sizes are `8M`, `35M`, `150M`. Init is `pretrained` or `random`.

```bash
python -m models.train --size 150M --init random --seed 42        # add --resume to continue an interrupted run
python -m models.extract_activations --size 150M --init random --stage pre  --seed 42
python -m models.extract_activations --size 150M --init random --stage post --seed 42
python -m analysis.accessibility --size 150M --init random --stage pre  --seed 42
python -m analysis.accessibility --size 150M --init random --stage post --seed 42
```

For a pretrained model, the `pre` stage is the unmodified Hugging Face model, so it does not depend on the seed. For the random model, `pre` uses the initial weights saved before training.

Step 2, all four conditions with three seeds. This is the full experiment and should run on a GPU server.

```bash
for cfg in "8M pretrained" "35M pretrained" "150M pretrained" "150M random"; do
  set -- $cfg
  for seed in 42 43 44; do
    python -m models.train --size $1 --init $2 --seed $seed
    for stage in pre post; do
      python -m models.extract_activations --size $1 --init $2 --stage $stage --seed $seed
      python -m analysis.accessibility     --size $1 --init $2 --stage $stage --seed $seed
    done
  done
done
python -m analysis.visualize
```

`train` resumes with `--resume`. `extract_activations` and `accessibility` skip work that is already finished. `visualize` makes the figures for `post` by default. Add `--stage pre` for representations before adaptation.

Optional runs:

- `--pooling mean` on extract, accessibility and visualize for mean-pooled representations
- `--protocol random` on accessibility for the random-cube sensitivity analysis
- `--full-finetune --tag _fullft` on train (and on extract) for the full fine-tuning control

Smoke test on a laptop:

```bash
python -m models.train --size 8M --init pretrained --max-samples 64 --epochs 2 --tag _smoke
python -m models.extract_activations --size 8M --init pretrained --stage post --tag _smoke --max-variants 300
```

Outputs tagged `_smoke` are ignored by the figure script unless you pass `--include-smoke`.

## Data checks (see `results/metrics/dataset_summary.txt` and `cube_summary.json`)

- 149,361 measured variants out of 160,000. No missing values or duplicates. Wild type VDGV has fitness 1.0. Fitness ranges from 0 to 8.76 and 29,477 variants are exactly 0. We use fitness without any transformation (FLIP's `target` column equals `Fitness`).
- The FLIP `sequence` column is the 265-residue 5LDE construct. Positions 39, 40, 41 and 54 (1-indexed) match the variant string for all 149,361 rows. We give ESM-2 only the first 56 residues, the GB1 domain, which is about 5 times cheaper. The mutated residues are the same. In ESM-2 the token index of residue p is p, because the CLS token sits at index 0. A test checks this.
- The official FLIP `two_vs_rest` split covers only FLIP's filtered subset of 8,733 variants, not all 149,361. It has 381 training variants, 43 validation variants (FLIP's own validation flag, which we kept) and 8,309 test variants (3 and 4 mutations). The other 140,628 variants are not in the split and only enter through the cubes.
- 109,235 complete cubes, matching the proposal and ORBIT.

## Accessibility protocol

We checked these details against ORBIT's official code (`maryam-rahimi/ORBIT-GB1`) and the paper. We wrote our own implementation.

| Item | What we do | How it was checked |
|---|---|---|
| Vertex order and sign coding | Bit i is position i. Bit 0 is -1 and bit 1 is +1. | Official code and unit tests |
| WHT scaling | Coefficient = (1/16) times the sum of f(v) chi_S(v) | Paper eq. 6, official code, reconstruction and orthogonality tests |
| Probe | One ridge probe per layer and order, pooled over cubes and over all subsets of that order. Alpha grid 1e-6 to 1e3 chosen on probe-validation cubes only. Near-constant features dropped. Features standardized with training statistics. Refit on train plus validation. Pooled R squared on test cubes. | Official code |
| Probe split | For each position, the 19 alternative amino acids are shuffled with `numpy.random.default_rng(42)` (one generator, positions 39 to 54 in order) and split 10, 4 and 5 into train, validation and test groups. A cube belongs to a split only if all four of its alternatives are in that split's group. Other cubes are dropped. | Official code. Our counts match ORBIT exactly: 7,352, 245 and 613 cubes, 101,025 dropped. |

The probe is solved in closed form with one eigendecomposition per order instead of ten sklearn fits. A test checks it against sklearn.

Overlap between probe splits: the train, validation and test cubes share only the wild-type variant. About 11% of the variants in probe-test cubes (146 of 1,282) are FLIP test variants, and about 4% (49 of 1,282) are FLIP train or validation variants. ORBIT mentions this too.

The random-cube split uses the same 8,210 cubes, assigned at random. Many variants then appear in both probe train and test cubes (81% of test-cube variants in our 8M smoke run). We only report it as a sensitivity analysis. We never report training R squared as accessibility, and negative values are kept.

## Verification against ORBIT's additive baseline

`python -m methods.additive_baseline` fits a ridge model on one-hot residues, with the strength chosen on the 43 FLIP validation variants (alpha = 1). The paper values below were copied from the ORBIT arXiv HTML page (Tables 1 and 3, ridge row). Please compare them with the PDF before citing them.

| Quantity | Ours | Paper |
|---|---|---|
| Order 2, 3, 4 recovery (613 probe-test cubes) | -0.0012, -0.0147, -0.0218 | -0.0012, -0.0147, -0.0218 |
| Order 1 recovery | 0.309 | 0.4126 |
| FLIP test Spearman, R squared, RMSE | 0.594, -0.130, 1.308 | 0.5617, 0.1084, 1.1619 |
| Largest order 2 to 4 coefficient of the additive predictions | 3e-16 | should be 0 |

The agreement for orders 2 to 4 (four decimals) supports our cube construction, probe split and WHT conventions. The paper does not give its ridge alpha or exact training set. As a diagnostic after the fact, alpha around 3 fit on all 424 FLIP training variants gives R squared 0.100, Spearman 0.564 and order 1 recovery 0.409, which is close to the paper. This diagnostic looks at test metrics, so we do not use it. The baseline we report keeps the validation-selected alpha.

## Assumptions and limitations

- Representation. The primary representation is the hidden states at the four mutated positions, concatenated. This is the same input the regression head uses, and ORBIT-FMIB (arXiv:2610.00672) also uses mutated-site states for ESM-2. ORBIT itself studied small MLPs, so there are no published ESM-2 numbers from ORBIT to reproduce. Mean pooling is an optional secondary experiment.
- Layers. Index 0 is the embedding layer and 1 to L are the transformer layers. Hugging Face applies ESM-2's final LayerNorm to layer L only. We compare models of different depth by relative depth (layer divided by L) and by absolute layer index.
- LoRA on a frozen random backbone is not the same as training a random network end to end. The random weights stay fixed, and only the rank-8 updates on the attention query, key and value layers and the head are trained. A weak result for the random model could come from this limit and not from missing pretraining. `--full-finetune` is an optional control and is not part of the primary experiment. The initial random weights are saved in `checkpoints/<run>/initial_backbone.pt` so before and after comparisons use the same starting point. Matched pretrained and random runs also use the same LoRA and head initialization, data order and hyperparameters (we checked the initialization with a one-off script, not a unit test).
- Hyperparameters are not tuned (learning rate 1e-3, 30 epochs, batch size 16, LoRA rank 8, alpha 16). The validation set has only 43 variants, which is too small to tune on. It is only used to pick the best epoch. The test set is evaluated once at the end.
- The proposal says to train on all variants with at most 2 mutations. The official split trains only on the 424 such variants in FLIP's filtered set.
- Fitness is heavy-tailed. We train with MSE on the raw values.
- ORBIT-FMIB reports that its own conclusions changed under an independent replication. Effects here may be small and depend on the protocol, so we plan to report every seed and keep unfavorable results.
- ESM-2 may have seen GB1 or close homologs during pretraining, which could affect question 1.

## Compute estimates

Measured on a Mac (MPS): extracting 13,150 variants with the 8M model takes 25 s. Accessibility for 7 layers of the 8M model takes 58 s. One epoch of 150M LoRA on 64 samples takes about 4 s. The 150M model has 149,309,402 parameters, of which 924,161 are trainable with LoRA, for both pretrained and random.

Estimated (not measured): activations in fp16 take about 0.23 GB (8M), 0.66 GB (35M) and 2.1 GB (150M) per condition. The full plan with 4 models, before and after adaptation and 3 seeds takes about 20 GB in `data/processed/activations/`. Training 150M needs a few GB of GPU memory. Accessibility for 150M needs roughly 6 to 8 GB of RAM at a time because only one layer is loaded at once, and should take tens of minutes per condition on CPU.

## Status

| Component | Implemented | Tested | Run on the real experiment |
|---|---|---|---|
| Data download, validation, cubes, probe split | yes | yes (counts match ORBIT) | yes |
| WHT and additive baseline | yes | yes (24 unit tests pass) | yes |
| `models/train.py` | yes | smoke runs on 8M and 150M (64 samples) | no |
| `models/extract_activations.py` | yes | checked against an independent forward pass | no |
| `analysis/accessibility.py` | yes | synthetic cubes and an 8M smoke run | no |
| `analysis/visualize.py` | yes | rendered on smoke data, which was then deleted | no |
| 4 conditions, 3 seeds, before and after adaptation | n/a | n/a | not run |
| Mean pooling, random-split sensitivity, full fine-tuning | yes (flags) | random split on the 8M smoke run only | no |

No ESM-2 accessibility results are in the repository yet. `results/figures/` contains only the additive baseline figure.

## References

1. Rahimimovassagh, Garibay, Yousefi (2026). Beyond tokens: probing higher-order epistasis in learned protein representations. arXiv:2608.24953. Code: github.com/maryam-rahimi/ORBIT-GB1.
2. Wu et al. (2016). Adaptation in protein fitness landscapes is facilitated by indirect paths. eLife 5:e16965.
3. Dallago et al. (2021). FLIP: benchmark tasks in fitness landscape inference for proteins. NeurIPS Datasets and Benchmarks. Data: github.com/J-SNACKKB/FLIP (`splits/gb1`).
4. Simon, Zou (2024). InterPLM. arXiv:2412.12101.
5. Ahlmann-Eltze, Huber, Anders (2025). Nature Methods 22(8):1657-1661.
6. Wittmann, Yue, Arnold (2021). Cell Systems 12(11):1026-1045.
7. ORBIT-FMIB, arXiv:2610.00672 (ESM-2 analysis with mutated-site representations).
