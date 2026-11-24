"""Cache hidden representations of every variant needed by the primary probe experiment.

Examples (from the repository root):
    python -m models.extract_activations --size 8M --init pretrained --stage pre     # before adaptation
    python -m models.extract_activations --size 8M --init pretrained --stage post    # after LoRA adaptation
    python -m models.extract_activations --size 150M --init random --stage pre       # saved random init weights
    python -m models.extract_activations --size 8M --init pretrained --stage pre --pooling mean   # secondary

Which variants: the unique variants of all probe-train / probe-val / probe-test cubes (the amino-acid-identity-
strict split, ~13k variants, wild type included once). Each variant is passed through the model ONCE and its
vectors are reused by every cube containing it.

Representation (--pooling):
  sites (primary): final-position hidden states at the four mutated residues (39,40,41,54), concatenated -> 4*d numbers.
                   Same representation the regression head reads, and the one used by ORBIT-FMIB's ESM-2 analysis.
  mean  (secondary): mean over the 56 residue tokens (<cls>/<eos>/padding excluded) -> d numbers.
Layers: index 0 = token embeddings, 1..L = outputs of transformer layers 1..L. NOTE: HF applies ESM-2's final
LayerNorm to layer L only, so layer L equals what the head sees; layers 1..L-1 are the raw residual stream.

Stages: pre = before fitness adaptation (pretrained weights, or the saved random initial weights);
        post = after LoRA adaptation (best-validation checkpoint from models/train.py, unmerged LoRA).

Output (data/processed/activations/<condition>/, git-ignored):
    layer_00.npy ... layer_LL.npy   float16 arrays (n_variants, dim), rows aligned with variant_indices.npy
    variant_indices.npy             row indices into data/processed/variants.csv
    metadata.json                   model, stage, checkpoint, pooling, dims, seed; progress.json allows resuming
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from methods.gb1 import CHECKPOINT_DIR, PROCESSED_DIR, variant_to_sequence
from models.train import MODEL_NAMES, SITE_TOKENS, build_backbone, get_device, load_trained_model, run_name

ACTIVATION_DIR = PROCESSED_DIR / "activations"


def condition_name(size, init, seed, stage, pooling, tag=""):
    """Directory name of one extraction condition. Pretrained 'pre' weights do not depend on the seed."""
    base = f"esm2_{size}_pretrained" if (init == "pretrained" and stage == "pre") else run_name(size, init, seed, tag)
    return f"{base}_{stage}_{pooling}"


def probe_variant_indices():
    """Sorted unique variant indices appearing in the probe-train/val/test cubes."""
    cubes = np.load(PROCESSED_DIR / "cubes.npz")
    in_probe = cubes["probe_split"] != "mixed"
    return np.unique(cubes["vertices"][in_probe].ravel())


def pool(hidden_states, pooling):
    """hidden_states: tuple of (B, L+2, d) per layer -> (B, n_layers+1, dim)."""
    stacked = torch.stack(hidden_states, dim=1)                          # (B, layers+1, L+2, d)
    if pooling == "sites":
        return stacked[:, :, SITE_TOKENS, :].flatten(2)                  # concatenate the 4 sites
    return stacked[:, :, 1:-1, :].mean(dim=2)                            # drop <cls> and <eos>; all variants have equal length


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=MODEL_NAMES, required=True)
    parser.add_argument("--init", choices=["pretrained", "random"], required=True)
    parser.add_argument("--stage", choices=["pre", "post"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pooling", choices=["sites", "mean"], default="sites")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tag", default="", help="run-name suffix used in models/train.py (post stage)")
    parser.add_argument("--full-finetune", action="store_true", help="the post checkpoint came from --full-finetune")
    parser.add_argument("--max-variants", type=int, default=None, help="only the first N variants (smoke tests)")
    args = parser.parse_args()

    device = get_device()
    name = condition_name(args.size, args.init, args.seed, args.stage, args.pooling, args.tag)
    out_dir = ACTIVATION_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- model ----
    if args.stage == "post":
        model = load_trained_model(args.size, args.init, args.seed, args.tag, args.full_finetune)
        backbone, checkpoint = model.backbone, str(CHECKPOINT_DIR / run_name(args.size, args.init, args.seed, args.tag) / "best.pt")
    else:
        run_dir = CHECKPOINT_DIR / run_name(args.size, args.init, args.seed, args.tag)
        backbone = build_backbone(args.size, args.init, args.seed, run_dir)
        checkpoint = "huggingface pretrained" if args.init == "pretrained" else str(run_dir / "initial_backbone.pt")
    backbone = backbone.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[args.size])
    n_layers = backbone.config.num_hidden_layers + 1
    dim = backbone.config.hidden_size * (4 if args.pooling == "sites" else 1)

    # ---- variants and (resumable) output arrays ----
    indices = probe_variant_indices()
    if args.max_variants:
        indices = indices[:args.max_variants]
    variants = pd.read_csv(PROCESSED_DIR / "variants.csv")["variant"].to_numpy()[indices]
    np.save(out_dir / "variant_indices.npy", indices)
    progress_path, metadata_path = out_dir / "progress.json", out_dir / "metadata.json"
    done = json.loads(progress_path.read_text())["done"] if progress_path.exists() else 0
    if done and (not metadata_path.exists() or json.loads(metadata_path.read_text())["n_variants"] != len(indices)):
        print("existing cache has a different number of variants (e.g. an earlier smoke test): restarting")
        done = 0
    arrays = [np.lib.format.open_memmap(out_dir / f"layer_{layer:02d}.npy", mode="r+" if done else "w+",
                                        dtype=np.float16, shape=(len(indices), dim)) for layer in range(n_layers)]
    metadata = {"condition": name, "size": args.size, "init": args.init, "stage": args.stage, "seed": args.seed,
                "pooling": args.pooling, "checkpoint": checkpoint, "n_layers_including_embedding": n_layers,
                "dim": dim, "n_variants": int(len(indices)), "dtype": "float16", "model_name": MODEL_NAMES[args.size]}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if done:
        print(f"resuming at variant {done}/{len(indices)}")

    # ---- extraction ----
    with torch.no_grad():
        for start in range(done, len(indices), args.batch_size):
            batch = [variant_to_sequence(v) for v in variants[start:start + args.batch_size]]
            tokens = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            output = backbone(**tokens, output_hidden_states=True)
            pooled = pool(output.hidden_states, args.pooling).float().cpu().numpy()      # (B, layers, dim)
            for layer in range(n_layers):
                arrays[layer][start:start + len(batch)] = pooled[:, layer].astype(np.float16)
            for array in arrays:
                array.flush()
            progress_path.write_text(json.dumps({"done": start + len(batch)}))
            print(f"\r{start + len(batch)}/{len(indices)}", end="", flush=True)
    print(f"\nsaved {n_layers} layers x {len(indices)} variants x {dim} dims to {out_dir}")


if __name__ == "__main__":
    main()
