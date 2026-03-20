"""
train.py — Training loop for ChessTransformer.
"""

import sys
import math
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import ChessTransformer
from pgn_data import load_data, ChessTokenizer
from attention import (
    MultiHeadAttention,
    RoPEMultiHeadAttention,
    GroupedQueryAttention,
    SlidingWindowAttention,
)


# Attention factory


def make_attention_factory(config: dict):
    """
    Returns a callable that creates one attention module per transformer layer.
    """
    variant     = config["variant"]
    d_model     = config["d_model"]
    n_heads     = config["n_heads"]
    dropout     = config["dropout"]
    kv_heads    = config.get("kv_heads", n_heads // 2)
    window_size = config.get("window_size", 32)
    max_seq_len = config["seq_len"]

    if variant == "vanilla":
        return lambda: MultiHeadAttention(d_model, n_heads, dropout)
    elif variant == "rope":
        return lambda: RoPEMultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
    elif variant == "gqa":
        return lambda: GroupedQueryAttention(d_model, n_heads, kv_heads, dropout)
    elif variant == "sparse":
        return lambda: SlidingWindowAttention(d_model, n_heads, window_size, dropout)
    else:
        raise ValueError(f"Unknown variant: {variant}")



### Learning rate schedule

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Linear warmup then cosine decay.

    Warmup prevents large gradient updates in early training when
    the model weights are random and loss is high.
    Cosine decay smoothly reduces LR to min_lr over training.
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay



### Evaluation

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Returns average validation loss (used to compute perplexity = exp(loss))."""
    model.eval()
    total_loss    = 0.0
    total_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        total_loss += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_move_legality(
    model: nn.Module,
    tokenizer: ChessTokenizer,
    device: torch.device,
    n_games: int = 50,
) -> float:
    """
    Generates n_games sequences and measures the fraction of moves that are
    legal according to chess rules (via python-chess).

    Each game is played out on a fresh board. The game stops at the first
    illegal or unparseable move — only moves up to that point count as legal.
    This is the standard legality metric used in chess language model papers.
    """

    import chess

    model.eval()
    legal_count = 0
    total_count = 0

    seed = torch.tensor([[tokenizer.bos_id]], device=device)

    for _ in range(n_games):
        board     = chess.Board()
        generated = model.generate(seed, max_new_tokens=40, temperature=1.0, top_k=40)
        moves     = tokenizer.decode(generated[0].tolist()[1:])  # skip BOS

        for move_str in moves:
            if move_str in ("<EOS>", "<PAD>"):
                break
            if move_str in ("<BOS>", "<UNK>"):
                total_count += 1
                break  # unparseable token — stop game
            total_count += 1
            try:
                move = board.parse_san(move_str)
                board.push(move)
                legal_count += 1
            except (chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError):
                break  # illegal or ambiguous move — stop game

    model.train()
    return legal_count / max(total_count, 1)


### Training loop

def train(config: dict):

    ### Reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ### Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    ### Weights & Biases
    wandb_run = None
    if config.get("wandb"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.get("wandb_project", "chess-transformer"),
                name=config["variant"],
                config=config,
            )
        except ImportError:
            print("wandb not installed — pip install wandb to enable logging.")

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ### Data
    train_ds, val_ds, tokenizer = load_data(
        pgn_path=config.get("pgn_path", "data/games.pgn"),
        seq_len=config["seq_len"],
        max_games=config.get("max_games"),
        train_split=config["train_split"],
    )
    tokenizer.save(str(out_dir / "tokenizer.json"))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,  num_workers=config["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

    ### Model
    model = ChessTransformer(
        vocab_size=tokenizer.vocab_size,
        attention_factory=make_attention_factory(config),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_seq_len=config["seq_len"],
        dropout=config["dropout"],
        use_sinusoidal_pe=config["variant"] != "rope",
    ).to(device)

    n_params = model.count_parameters()
    print(f"Variant: {config['variant']} | Parameters: {n_params:,}")

    # FLOPs per optimizer step (standard 6N approximation: 2 forward + 4 backward)
    tokens_per_step = config["batch_size"] * config["seq_len"] * config.get("gradient_accumulation_steps", 1)
    flops_per_step  = 6 * n_params * tokens_per_step

    if config.get("compile"):
        if not hasattr(torch, "compile"):
            print("torch.compile() not available (requires PyTorch 2.0+) — skipping")
        else:
            try:
                print("Compiling model with torch.compile()...")
                model = torch.compile(model)
                print("torch.compile() enabled.")
            except RuntimeError as e:
                print(f"torch.compile() failed ({e}) — falling back to eager mode")

    ### Optimizer
    decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": config["weight_decay"]},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=config["max_lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    ### Training loop
    step            = 0
    max_steps       = config["max_steps"]
    warmup_steps    = config.get("warmup_steps", max_steps // 10)
    best_val_loss   = float("inf")
    patience        = config.get("patience", 0)   # 0 = disabled
    patience_counter = 0
    metrics_log     = []

    print(f"\nStarting training: {max_steps} steps, warmup {warmup_steps} steps")
    print("-" * 60)

    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()

    accum_steps = config.get("gradient_accumulation_steps", 1)

    while step < max_steps:
        # Gradient accumulation: accumulate gradients over `accum_steps` micro-batches
        # before doing an optimizer step. Effective batch size = batch_size * accum_steps.
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, targets=y)
            # Divide loss before backward so gradients are averaged, not summed
            (loss / accum_steps).backward()
            accum_loss += loss.item()

        loss_for_log = accum_loss / accum_steps

        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, config["max_lr"], config["min_lr"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        optimizer.step()
        step += 1

        # Logging
        if step % config["log_interval"] == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = config["batch_size"] * accum_steps * config["seq_len"] * config["log_interval"] / dt
            print(
                f"step {step:5d} | loss {loss_for_log:.4f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.3f} | "
                f"{tokens_per_sec:.0f} tok/s"
            )
            if wandb_run:
                wandb_run.log({
                    "train/loss":           loss_for_log,
                    "train/lr":             lr,
                    "train/grad_norm":      grad_norm,
                    "train/tokens_per_sec": tokens_per_sec,
                }, step=step)

        # Validation
        if step % config["eval_interval"] == 0:
            val_loss = evaluate(model, val_loader, device)
            val_ppl  = math.exp(min(val_loss, 20))
            legality = evaluate_move_legality(model, tokenizer, device)

            print(f"\n{'='*60}")
            print(f"Step {step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Move Legality: {legality:.1%}")
            print(f"{'='*60}\n")

            metrics_log.append({
                "step":             step,
                "val_loss":         val_loss,
                "val_ppl":          val_ppl,
                "move_legality":    legality,
                "lr":               lr,
                "cumulative_flops": flops_per_step * step,
            })

            with open(out_dir / "metrics.json", "w") as f:
                json.dump(metrics_log, f, indent=2)

            if wandb_run:
                wandb_run.log({
                    "val/loss":             val_loss,
                    "val/perplexity":       val_ppl,
                    "val/move_legality":    legality,
                    "train/cumulative_flops": flops_per_step * step,
                }, step=step)

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                checkpoint = {
                    "step":                 step,
                    "model_state":          raw_model.state_dict(),
                    "optimizer_state":      optimizer.state_dict(),
                    "config":               config,
                    "val_loss":             val_loss,
                    "tokenizer_vocab_size": tokenizer.vocab_size,
                }
                torch.save(checkpoint, out_dir / "best_model.pt")
                print(f"Saved best model (val_loss={val_loss:.4f})")
            elif patience > 0:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"Early stopping at step {step} (patience={patience})")
                    break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    if wandb_run:
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.summary["best_val_ppl"]  = math.exp(min(best_val_loss, 20))
        wandb_run.finish()

    return metrics_log


#### CLI

def parse_args():
    parser = argparse.ArgumentParser(description="Train ChessTransformer")

    # Data
    parser.add_argument("--pgn_path",    type=str,   default="data/games.pgn")
    parser.add_argument("--max_games",   type=int,   default=None)
    parser.add_argument("--seq_len",     type=int,   default=128)
    parser.add_argument("--train_split", type=float, default=0.9)

    # Model
    parser.add_argument("--variant",     type=str,   default="vanilla",
                        choices=["vanilla", "rope", "gqa", "sparse"])
    parser.add_argument("--d_model",     type=int,   default=128)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=4)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--kv_heads",    type=int,   default=2)
    parser.add_argument("--window_size", type=int,   default=32)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Training
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--max_steps",    type=int,   default=5000)
    parser.add_argument("--max_lr",       type=float, default=3e-4)
    parser.add_argument("--min_lr",       type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip",                 type=float, default=1.0)
    parser.add_argument("--patience",                  type=int,   default=0,
                        help="Early stopping patience (evals without improvement). 0 = disabled")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients over N micro-batches before stepping (effective batch = batch_size * N)")
    parser.add_argument("--warmup_steps", type=int,   default=500)

    # Logging
    parser.add_argument("--log_interval",   type=int,  default=50)
    parser.add_argument("--eval_interval",  type=int,  default=500)
    parser.add_argument("--out_dir",        type=str,  default="checkpoints/vanilla")
    parser.add_argument("--wandb",          action="store_true", default=False,
                        help="Enable Weights & Biases experiment tracking")
    parser.add_argument("--wandb_project",  type=str,  default="chess-transformer",
                        help="W&B project name")
    parser.add_argument("--compile",        action="store_true", default=False,
                        help="Compile model with torch.compile() (PyTorch 2.0+, Linux/CUDA recommended)")

    return vars(parser.parse_args())


if __name__ == "__main__":
    config = parse_args()
    config["out_dir"] = f"checkpoints/{config['variant']}"
    train(config)


