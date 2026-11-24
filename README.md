# What does a protein language model learn about epistasis?

We study how pretraining and model size affect what ESM-2 represents about epistasis (interactions between mutations), following ORBIT (Rahimimovassagh et al., 2026, arXiv:2608.24953) on the GB1 four-site fitness landscape. Authors: Arya Bhanushali and Aryan Thakkar. See [proposal.md](proposal.md).

Research questions (our expected outcomes are hypotheses, not results):

1. Is pretrained ESM-2 150M more interaction-accessible than a randomly initialized one with the same architecture?
2. Does accessibility of higher-order epistasis grow with size (8M, 35M, 150M)?
3. At which layers does accessibility appear for each interaction order (1 to 4)?

## Method

- Data: 149,361 measured GB1 variants at positions 39, 40, 41, 54 (wild type VDGV; Wu et al., 2016).
- Cube: choose one alternative amino acid per position. The 16 variants using the wild-type or the chosen residue at each position form a cube. 109,235 cubes are complete.
- WHT: turns a cube's 16 values into 16 coefficients, one per subset of positions. Order 0 is the constant, order 1 single effects, orders 2 to 4 are epistasis. The same transform applied to a model's 16 hidden vectors gives representation coefficients.
- Accessibility: for each layer and order, a ridge probe predicts the measured fitness coefficients from the representation coefficients. Accessibility is its R squared on held-out cubes (negative means worse than the mean).
- Models: `facebook/esm2_t6_8M_UR50D`, `t12_35M_UR50D`, `t30_150M_UR50D` (pretrained) and a random 150M of the same architecture. All are adapted with LoRA and a linear head on the official FLIP split, and hidden states are cached before and after adaptation.

Worked example (real cube, alternatives I, Y, C, A). Bit i of the vertex number says whether position i (39, 40, 41, 54) is mutated.

| v | variant | fitness | v | variant | fitness |
|---|---|---|---|---|---|
| 0 | VDGV (wild type) | 1.000 | 8 | VDGA | 1.373 |
| 1 | IDGV | 1.446 | 9 | IDGA | 1.512 |
| 2 | VYGV | 3.891 | 10 | VYGA | 2.406 |
| 3 | IYGV | 4.388 | 11 | IYGA | 2.435 |
| 4 | VDCV | 0.061 | 12 | VDCA | 3.687 |
| 5 | IDCV | 0.111 | 13 | IDCA | 4.130 |
| 6 | VYCV | 0.143 | 14 | VYCA | 5.360 |
| 7 | IYCV | 0.189 | 15 | IYCA | 5.073 |

The coefficient for a subset S is (1/16) times the sum over vertices of f(v) times the product of x_i for i in S, with x_i = -1 for wild type and +1 for mutant. Here the constant is 2.325, order 1 is 0.085 / 0.660 / 0.019 / 0.922 (positions 39 / 40 / 41 / 54), the largest order 2 term is positions 41+54 at 1.297 (G41C alone gives 0.061, V54A alone 1.373, together 3.687), the largest order 3 term is 40+41+54 at 0.396, and order 4 is -0.035.

## Pipeline and files

```
data/prepare_data.py -> data/build_cubes.py -> methods/additive_baseline.py (check against ORBIT)
                                  |            methods/test_methods.py (run before model work)
                                  v
models/train.py -> models/extract_activations.py (pre / post) -> analysis/accessibility.py -> analysis/visualize.py
```

| Path | Purpose |
|---|---|
| `methods/gb1.py` | Constants, paths, indexing helpers |
| `methods/wht.py` | WHT, grouping by order, Walsh energy, pooled R squared |
| `methods/additive_baseline.py` | Additive ridge baseline and ORBIT verification |
| `methods/test_methods.py` | 24 pytest tests |
| `data/prepare_data.py` | Download and validate FLIP GB1 data and split |
| `data/build_cubes.py` | All complete cubes, probe split, shared-variant counts |
| `models/train.py` | LoRA (or `--full-finetune`) plus regression head, resumable |
| `models/extract_activations.py` | Cache per-layer representations of each unique variant, resumable |
| `analysis/accessibility.py` | Ridge probes and accessibility CSVs |
| `analysis/visualize.py` | Figures and summary table |

Weights go in `checkpoints/` and activations in `data/processed/activations/` (neither is tracked by git). Small CSV/JSON results and figures go in `results/`.

## Setup

Python 3.11 is required. On Apple Silicon use a native arm64 Python (`uv python install 3.11`); an x86_64 Python under Rosetta cannot install the pinned torch.

```bash
git clone https://github.com/aryabhanushali/esm2-epistasis.git && cd esm2-epistasis
python3.11 -m venv .venv && source .venv/bin/activate    # or: uv venv --python 3.11 .venv
pip install -r requirements.txt
```

On a CUDA server, if the default torch wheel does not match the driver, install torch first from pytorch.org (for example `--index-url https://download.pytorch.org/whl/cu121`), then the rest. The device is chosen automatically (CUDA, MPS, CPU) and bfloat16 is used on CUDA only. Run everything from the repository root.

## Running

Data and checks (CPU, minutes). Do this first.

```bash
python -m data.prepare_data
python -m data.build_cubes
python -m pytest methods/test_methods.py -q
python -m methods.additive_baseline
```

One condition (sizes `8M|35M|150M`, init `pretrained|random`):

```bash
python -m models.train --size 150M --init random --seed 42          # --resume continues a run
python -m models.extract_activations --size 150M --init random --stage pre  --seed 42
python -m models.extract_activations --size 150M --init random --stage post --seed 42
python -m analysis.accessibility --size 150M --init random --stage pre  --seed 42
python -m analysis.accessibility --size 150M --init random --stage post --seed 42
```

For pretrained models, `pre` is the unmodified Hugging Face model. For the random model, `pre` uses the initial weights saved before training.

Full experiment (GPU server): four conditions, three seeds.

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
python -m analysis.visualize          # add --stage pre for representations before adaptation
```

Optional: `--pooling mean` (extract, accessibility, visualize), `--protocol random` (accessibility, leaky sensitivity check), `--full-finetune --tag _fullft` (train and extract). Smoke test: `python -m models.train --size 8M --init pretrained --max-samples 64 --epochs 2 --tag _smoke`. Runs tagged `_smoke` are ignored by `visualize` unless `--include-smoke` is given.

## Data and protocol

- All 149,361 variants validated: no missing values or duplicates, wild type fitness 1.0, fitness 0 to 8.76 used without transformation.
- FLIP's `sequence` column is a 265-residue construct. We give ESM-2 only its first 56 residues (the GB1 domain), which is about 5 times cheaper; positions 39/40/41/54 are 1-indexed and the ESM token index of residue p is p (CLS is index 0). Both are tested.
- The official `two_vs_rest` split covers only 8,733 filtered variants: 381 train, 43 validation (FLIP's own flag, kept), 8,309 test. The validation set is used only to pick the best epoch.
- Probe split (ORBIT's amino-acid-identity-strict split, from the official code `maryam-rahimi/ORBIT-GB1`): per position, the 19 alternatives are shuffled with `default_rng(42)` and split 10/4/5. A cube is train, validation or test only if all four alternatives fall in the same group. Our counts match ORBIT: 7,352 / 245 / 613 cubes, 101,025 dropped. The splits share only the wild-type variant. About 11% of probe-test variants are FLIP test variants and 4% are FLIP train or validation variants.
- Probe: one ridge per (layer, order), pooled over cubes and over the subsets of that order; alpha grid 1e-6 to 1e3 chosen on probe-validation cubes; standardization and feature filtering use training data only; refit on train plus validation; pooled R squared on test cubes. Solved in closed form (a test checks it against sklearn).
- Representation: hidden states at the four mutated positions, concatenated (the head's input, as in ORBIT-FMIB). Mean pooling is optional. Layer 0 is the embeddings; ESM-2's final LayerNorm is applied to the last layer only. Models are compared by relative depth and absolute layer.
- The random-cube split (same 8,210 cubes, random assignment) shares many variants between probe train and test (81% in our 8M smoke run, where it gave accessibility of 0.7 to 0.95 against mostly negative values for the strict split). It is only a sensitivity check.

## Verification against ORBIT's additive baseline

Ridge on one-hot residues, alpha chosen on the 43 validation variants (alpha = 1). Paper values were copied from the arXiv HTML (Tables 1 and 3); please check them against the PDF.

| Quantity | Ours | Paper |
|---|---|---|
| Order 2, 3, 4 recovery (613 probe-test cubes) | -0.0012, -0.0147, -0.0218 | -0.0012, -0.0147, -0.0218 |
| Order 1 recovery | 0.309 | 0.4126 |
| FLIP test Spearman, R squared, RMSE | 0.594, -0.130, 1.308 | 0.5617, 0.1084, 1.1619 |
| Largest order 2 to 4 coefficient of the predictions | 3e-16 | should be 0 |

The match for orders 2 to 4 supports our cube, split and WHT conventions. The paper does not give its alpha or training set. As a diagnostic after the fact (it uses test metrics, so we do not use it), alpha near 3 fit on all 424 training variants gives R squared 0.100, Spearman 0.564 and order 1 recovery 0.409.

## Limitations

- LoRA on a frozen random backbone is not training a random network end to end: only rank-8 updates on attention query/key/value and the head change. A weak result for the random model may reflect this. `--full-finetune` is an optional control. Initial random weights are saved so before/after comparisons share the same start, and matched runs share LoRA/head initialization, data order and hyperparameters.
- Hyperparameters are untuned (lr 1e-3, 30 epochs, batch 16, LoRA r=8, alpha=16) because 43 validation variants are too few. The test set is evaluated once at the end.
- The official split trains only on the 424 FLIP-filtered variants with at most 2 mutations, not all of them.
- ORBIT studied small MLPs, so there are no published ESM-2 ORBIT numbers to compare with. ORBIT-FMIB (arXiv:2610.00672) reports its ESM-2 conclusions were unstable under replication, so effects may be small; we plan to report all seeds and keep unfavorable results.
- ESM-2 may have seen GB1 homologs in pretraining.

## Compute

Measured on a Mac (MPS): 8M extraction of 13,150 variants 25 s; 8M accessibility for 7 layers 58 s; 150M has 149,309,402 parameters with 924,161 trainable (same for pretrained and random). Estimated: fp16 activations about 0.23 / 0.66 / 2.1 GB per condition for 8M / 35M / 150M, about 20 GB for the full plan; accessibility for 150M needs roughly 6 to 8 GB of RAM and tens of minutes per condition.

## Status

| Component | Implemented | Tested | Run on the real experiment |
|---|---|---|---|
| Data, cubes, probe split | yes | yes (counts match ORBIT) | yes |
| WHT, additive baseline | yes | yes (24 tests) | yes |
| `train.py`, `extract_activations.py`, `accessibility.py`, `visualize.py` | yes | smoke runs on 8M (train also on 150M) | 8M and 35M |
| 8M pretrained, 8M random, 35M pretrained, seeds 42/43/44, before and after adaptation | yes | n/a | yes (run on a Mac) |
| 150M pretrained, 150M random | yes | n/a | not run (GPU server) |

First results (8M pretrained, strict probe split, last layer, after adaptation, seeds 42/43/44):

| Order | Accessibility R squared |
|---|---|
| 1 | 0.61, 0.72, 0.73 |
| 2 | 0.53, 0.38, 0.56 |
| 3 | 0.11, -0.06, -0.05 |
| 4 | -0.43, -1.05, -0.42 |

Same layer and seeds (42/43/44), for comparison:

| Condition | Order 1 | Order 2 | Order 3 | Order 4 |
|---|---|---|---|---|
| 8M random | -0.57, -0.58, -0.33 | 0.16, -0.29, 0.23 | 0.09, -0.30, 0.17 | -0.49, -0.92, -0.35 |
| 35M pretrained | 0.67, 0.48, 0.75 | 0.52, 0.46, 0.57 | 0.03, -0.21, 0.24 | -0.29, -0.18, -0.55 |

- Orders 1 and 2 are accessible in pretrained models; order 4 is negative in every condition (worse than predicting the mean).
- Order 3 is near zero and varies a lot between seeds. 8M random is not clearly worse than 8M pretrained on order 3, and 35M is not clearly better than 8M.
- Before adaptation, pretrained 8M has orders 2 to 4 at zero or below except a weak order 2 signal (0.13 to 0.22) in layers 4 to 6.
- Test Spearman on the FLIP test set (8,309 variants): 8M pretrained 0.685 / 0.695 / 0.711, 35M pretrained 0.694 / 0.678 / 0.707, 8M random 0.631 / 0.540 / 0.604. The random model also fits the task less well, so its accessibility gap is partly a training gap (see Limitations).
- Three seeds and 8M/35M only: nothing here answers question 1, which concerns 150M. Figures 3 to 7 and `summary_table.csv` are in `results/`.

## References

1. Rahimimovassagh, Garibay, Yousefi (2026). Beyond tokens. arXiv:2608.24953. Code: github.com/maryam-rahimi/ORBIT-GB1.
2. Wu et al. (2016). eLife 5:e16965.
3. Dallago et al. (2021). FLIP. NeurIPS Datasets and Benchmarks. github.com/J-SNACKKB/FLIP.
4. Simon, Zou (2024). InterPLM. arXiv:2412.12101.
5. Ahlmann-Eltze, Huber, Anders (2025). Nature Methods 22(8):1657-1661.
6. Wittmann, Yue, Arnold (2021). Cell Systems 12(11):1026-1045.
7. ORBIT-FMIB, arXiv:2610.00672.
