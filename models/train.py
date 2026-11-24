"""Adapt an ESM-2 model to GB1 fitness prediction with LoRA + a linear regression head.

Examples (from the repository root):
    python -m models.train --size 8M   --init pretrained            # capacity comparison
    python -m models.train --size 150M --init random                 # pretraining comparison (same recipe)
    python -m models.train --size 8M --init pretrained --max-samples 64 --epochs 2 --tag _smoke   # quick test

Design (kept identical across all conditions so only size / initialization differ):
  * Data: official FLIP two_vs_rest split. train = 381, validation = 43 (used for checkpoint selection
    only), test = 8,309 (evaluated once, at the end, with the best-validation checkpoint).
  * Input: the 56-residue GB1 domain. Prediction head: Linear on the CONCATENATED final hidden states at the four
    mutated positions (4 x hidden_size inputs) -> fitness. Loss: MSE on untransformed fitness.
  * Adaptation: LoRA (rank/alpha/dropout below) on the attention query/key/value projections of a FROZEN
    backbone; only LoRA matrices and the head are trained. `--full-finetune` trains every weight instead (optional control).
  * LIMITATION: LoRA on a frozen random backbone is not the same as training a random network end-to-end:
    the random weights stay fixed and only low-rank updates + the head can adapt. Use --full-finetune to compare.
  * Random init: same architecture built from the config; its initial weights are saved to
    checkpoints/<run>/initial_backbone.pt so "before adaptation" can be analysed with the identical weights.
  * Checkpoints hold only trainable tensors (small). `last.pt` allows --resume; `best.pt` is the best-validation state.
"""
import argparse
import contextlib
import json
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr
from torch import nn
from transformers import AutoConfig, AutoTokenizer, EsmModel

from methods.gb1 import CHECKPOINT_DIR, METRICS_DIR, POSITIONS, PROCESSED_DIR, esm_token_index, variant_to_sequence

MODEL_NAMES = {"8M": "facebook/esm2_t6_8M_UR50D", "35M": "facebook/esm2_t12_35M_UR50D",
               "150M": "facebook/esm2_t30_150M_UR50D"}
SITE_TOKENS = [esm_token_index(p) for p in POSITIONS]       # token indices of the four mutated residues
LORA_TARGETS = ["query", "key", "value"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(use_amp):
    """bfloat16 autocast on CUDA; a no-op on CPU/MPS (full float32 there)."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()


def run_name(size, init, seed, tag=""):
    return f"esm2_{size}_{init}_seed{seed}{tag}"


def build_backbone(size, init, seed, run_dir=None):
    """ESM-2 backbone, either pretrained or randomly initialized (same architecture; seeded).
    For a random backbone the initial weights are saved (first call) and reloaded (later calls)
    so that every stage of the experiment uses exactly the same initialization."""
    name = MODEL_NAMES[size]
    if init == "pretrained":
        return EsmModel.from_pretrained(name, add_pooling_layer=False)
    torch.manual_seed(seed)
    backbone = EsmModel(AutoConfig.from_pretrained(name), add_pooling_layer=False)
    initial_path = run_dir / "initial_backbone.pt" if run_dir else None
    if initial_path is not None:
        if initial_path.exists():
            backbone.load_state_dict(torch.load(initial_path, map_location="cpu"))
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(backbone.state_dict(), initial_path)
    return backbone


class FitnessRegressor(nn.Module):
    """backbone -> hidden states at the 4 mutated positions (concatenated) -> linear -> fitness."""

    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(4 * hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(hidden[:, SITE_TOKENS, :].flatten(1)).squeeze(-1)


def build_model(size, init, seed, run_dir, full_finetune, lora_r=8, lora_alpha=16, lora_dropout=0.05):
    """Backbone + LoRA (or full fine-tuning) + head. The head is initialized after seeding."""
    backbone = build_backbone(size, init, seed, run_dir)
    hidden_size = backbone.config.hidden_size
    if not full_finetune:
        backbone = get_peft_model(backbone, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                                       target_modules=LORA_TARGETS))
    torch.manual_seed(seed + 1)                                # same LoRA/head initialization for matched runs
    model = FitnessRegressor(backbone, hidden_size)
    if not full_finetune:                                      # re-draw LoRA A matrices under the fixed seed
        for module in model.modules():
            if hasattr(module, "lora_A"):
                module.reset_lora_parameters("default", True)
    model.head.reset_parameters()
    return model


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


def load_trained_model(size, init, seed, tag="", full_finetune=False, which="best"):
    """Rebuild a model and load its adapted weights (used by extract_activations.py)."""
    run_dir = CHECKPOINT_DIR / run_name(size, init, seed, tag)
    model = build_model(size, init, seed, run_dir, full_finetune)
    state = torch.load(run_dir / f"{which}.pt", map_location="cpu")
    model.load_state_dict(state["trainable_state"], strict=False)
    return model.eval()


def encode(variants, tokenizer, device):
    sequences = [variant_to_sequence(v) for v in variants]
    batch = tokenizer(sequences, return_tensors="pt", padding=True)
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)


@torch.no_grad()
def predict(model, variants, tokenizer, device, batch_size, use_amp):
    model.eval()
    outputs = []
    for start in range(0, len(variants), batch_size):
        ids, mask = encode(variants[start:start + batch_size], tokenizer, device)
        with autocast(use_amp):
            outputs.append(model(ids, mask).float().cpu())
    return torch.cat(outputs).numpy()


def regression_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    return {"mse": float(np.mean((truth - prediction) ** 2)),
            "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
            "r2": float(1 - np.sum((truth - prediction) ** 2) / np.sum((truth - truth.mean()) ** 2)),
            "spearman": float(spearmanr(truth, prediction)[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=MODEL_NAMES, required=True)
    parser.add_argument("--init", choices=["pretrained", "random"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="fixed (not tuned): validation set has only 43 variants")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--full-finetune", action="store_true", help="optional control: train all weights, no LoRA")
    parser.add_argument("--max-samples", type=int, default=None, help="cap each split's size (smoke tests only)")
    parser.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast on CUDA")
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>/last.pt")
    parser.add_argument("--tag", default="", help="suffix for the run name, e.g. _fullft or _smoke")
    args = parser.parse_args()

    device = get_device()
    use_amp = device.type == "cuda" and not args.no_amp
    name = run_name(args.size, args.init, args.seed, args.tag)
    run_dir = CHECKPOINT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {name} on {device} (amp={use_amp})")

    # ---- data: official FLIP split ----
    variants = pd.read_csv(PROCESSED_DIR / "variants.csv")
    splits = {}
    for split in ["train", "val", "test"]:
        part = variants[variants["split"] == split]
        if args.max_samples:
            part = part.sample(min(args.max_samples, len(part)), random_state=0)
        splits[split] = (part["variant"].tolist(), part["fitness"].to_numpy(dtype=np.float32))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[args.size])

    # ---- model ----
    model = build_model(args.size, args.init, args.seed, run_dir, args.full_finetune,
                        args.lora_r, args.lora_alpha, args.lora_dropout).to(device)
    if args.full_finetune:
        for p in model.parameters():
            p.requires_grad = True
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: total {total_params:,}, trainable {trainable_params:,}")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)

    history, best_val, start_epoch, elapsed_before = [], float("inf"), 0, 0.0
    if args.resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", map_location="cpu")
        model.load_state_dict(state["trainable_state"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        history, best_val, start_epoch, elapsed_before = state["history"], state["best_val"], state["epoch"], state["elapsed"]
        print(f"resumed from epoch {start_epoch}")

    # ---- training ----
    train_variants, train_targets = splits["train"]
    shuffler = np.random.default_rng(args.seed)
    for _ in range(start_epoch):                                  # keep the data order identical after resuming
        shuffler.permutation(len(train_variants))
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = shuffler.permutation(len(train_variants))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            ids, mask = encode([train_variants[i] for i in batch], tokenizer, device)
            target = torch.tensor(train_targets[batch], device=device)
            with autocast(use_amp):
                prediction = model(ids, mask)
            loss = nn.functional.mse_loss(prediction.float(), target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val_prediction = predict(model, splits["val"][0], tokenizer, device, args.eval_batch_size, use_amp)
        val_loss = float(np.mean((val_prediction - splits["val"][1]) ** 2))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        print(f"epoch {epoch + 1:>3}  train {history[-1]['train_loss']:.4f}  val {val_loss:.4f}")
        checkpoint = {"trainable_state": trainable_state(model), "history": history, "epoch": epoch + 1,
                      "best_val": min(best_val, val_loss), "elapsed": elapsed_before + time.time() - started}
        if val_loss < best_val:                                  # checkpoint selection uses validation only
            best_val = val_loss
            torch.save({**checkpoint, "best_epoch": epoch + 1}, run_dir / "best.pt")
        torch.save({**checkpoint, "optimizer": optimizer.state_dict()}, run_dir / "last.pt")
    training_time = elapsed_before + time.time() - started

    # ---- final evaluation with the best-validation checkpoint (test touched once) ----
    best = torch.load(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best["trainable_state"], strict=False)
    result = {"run": name, "size": args.size, "init": args.init, "seed": args.seed, "full_finetune": args.full_finetune,
              "best_epoch": best["best_epoch"], "total_params": total_params, "trainable_params": trainable_params,
              "training_time_seconds": training_time, "device": str(device), "args": vars(args), "history": history,
              "n_samples": {k: len(v[0]) for k, v in splits.items()}}
    for split in ["val", "test"]:
        prediction = predict(model, splits[split][0], tokenizer, device, args.eval_batch_size, use_amp)
        result[split] = regression_metrics(splits[split][1], prediction)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"train_{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["best_epoch", "val", "test", "training_time_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
