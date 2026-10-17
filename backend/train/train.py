"""
Main training script for Polish language model - Optimized for Polish quality.

All model hyperparameters come from GPTConfig (model/gpt.py) - the single
source of truth. Only vocab_size is overridden here, from the trained
tokenizer, so the model always matches the corpus it is trained on.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional
from tokenizers import Tokenizer
import torch

from configs import GPTConfig, MODEL_PATH, TOKENIZER_PATH
from model.checkpoint import load_checkpoint, save_checkpoint
from model.gpt import GPT
from model.inference import generate
from train.dataloader import SmallLLMDataLoader

# Logs contain Polish diacritics; a non-UTF-8 console (e.g. cp1250/cp1252
# on Windows) cannot encode them and would crash the run mid-training.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

torch.set_float32_matmul_precision("high")

# Fixed prompts: per-epoch sample (comparable across epochs) and the
# post-training demo.
SAMPLE_PROMPT = "Polska to kraj, w którym pradawne lasy"
DEMO_PROMPT = (
    "Polska to kraina pełna wierzeń o demonach, w której pod każdym dębem mieszka"
)


@dataclass
class TrainingConfig:
    """Training-loop knobs; architecture stays in GPTConfig.

    Defaults fit the current mythology corpus (~1.7M tokens): roughly
    730 train steps and 72 val batches per epoch at B=8, T=256. For a
    fast smoke test set max_shards (e.g. 2) and model_path to a
    throwaway location so the real checkpoint is not overwritten.
    """

    batch_size: int = 8
    learning_rate: float = 5e-4
    max_epochs: int = 5
    patience: int = 2  # epochs without val improvement before early stop
    log_every: int = 25
    # Train/validate on only the first N shards of each split (fast
    # iteration). None = all shards.
    max_shards: Optional[int] = None
    # Context length override; None -> GPTConfig.block_size.
    context_length: Optional[int] = None
    # Checkpoint destination; None -> configs.MODEL_PATH. Smoke runs
    # should point this elsewhere to keep the real checkpoint intact.
    model_path: Optional[str] = None

    def __post_init__(self):
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert self.patience >= 1, "patience must be at least 1"
        assert self.log_every > 0, "log_every must be positive"
        assert (
            self.max_shards is None or self.max_shards > 0
        ), "max_shards must be positive"
        assert (
            self.context_length is None or self.context_length > 0
        ), "context_length must be positive"


def train_small_model(config: Optional[TrainingConfig] = None):
    """Train a small, affordable model on local laptop."""
    config = config or TrainingConfig()

    # Load the tokenizer FIRST: the model's vocab size must match it,
    # otherwise generation can sample never-trained embeddings.
    tokenizer_path = str(TOKENIZER_PATH)
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}\nPlease run: python -m cli --train_tokenizer"
        )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    print(f"Using tokenizer from: {tokenizer_path}")
    print(f"Tokenizer vocab size: {vocab_size}")

    # Architecture from GPTConfig defaults; only vocab_size is corpus-specific.
    cfg = GPTConfig(vocab_size=vocab_size)

    # Create small model (seeded so weight init is reproducible run-to-run)
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config=cfg).to(device)

    # Autocast dtype on GPU: bf16 on Ampere+, fp16 on older GPUs. CPU
    # stays fp32 (bf16 autocast measured ~35x slower without native support).
    autocast_dtype = torch.bfloat16
    if device == "cuda" and torch.cuda.get_device_capability() < (8, 0):
        autocast_dtype = torch.float16
        print(f"GPU capability {torch.cuda.get_device_capability()}: fp16 autocast")

    print("Starting small model training")
    print(f"Training config: {config}")
    print(
        f"Model size: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters"
    )
    print(f"Device: {device}")

    # Hyperparameters from TrainingConfig; context length defaults to
    # GPTConfig.block_size so the loader and the model can never disagree.
    batch_size = config.batch_size
    context_length = config.context_length or cfg.block_size
    learning_rate = config.learning_rate
    max_epochs = config.max_epochs
    patience = config.patience
    log_every = config.log_every

    # Loaders; max_shards limits each split to its first N shards for
    # fast smoke runs.
    train_loader = SmallLLMDataLoader(
        B=batch_size,
        T=context_length,
        process_rank=0,
        num_processes=1,
        split="train",
        max_shards=config.max_shards,
    )

    val_loader = SmallLLMDataLoader(
        B=batch_size,
        T=context_length,
        process_rank=0,
        num_processes=1,
        split="val",
        max_shards=config.max_shards,
    )

    if len(val_loader) == 0:
        raise RuntimeError(
            "Validation split is empty: no val shard holds enough tokens "
            f"for one B={batch_size}, T={context_length} batch"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model_path = config.model_path or str(MODEL_PATH)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    # Training loop: epochs over the train split, val loss after each epoch,
    # best-val checkpoint kept, early stop when val loss flattens.
    model.train()
    start_time = time.time()
    loss_history = []
    steps_per_epoch = max(1, len(train_loader))
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    steps_done = 0

    for epoch in range(1, max_epochs + 1):
        epoch_loss_start = len(loss_history)
        for _ in range(steps_per_epoch):
            x_batch, y_batch = train_loader[0]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Autocast only on CUDA (fp32 is the CPU fast path).
            with torch.autocast(
                device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")
            ):
                logits, loss = model(x_batch, y_batch)

            optimizer.zero_grad()
            loss.backward()
            # Clip so one bad batch cannot poison the step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_history.append(loss.item())
            steps_done += 1
            if steps_done % log_every == 0:
                print(f"  step {steps_done} | train loss: {loss.item():.4f}")

        # Full-split validation; the loader is reset each epoch so val
        # losses are comparable across epochs.
        model.eval()
        val_loader.reset()
        val_losses = []
        with torch.no_grad():
            for _ in range(len(val_loader)):
                x_batch, y_batch = val_loader[0]
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                with torch.autocast(
                    device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")
                ):
                    logits, val_loss = model(x_batch, y_batch)
                val_losses.append(val_loss.item())
        val_item = sum(val_losses) / len(val_losses)
        model.train()

        epoch_losses = loss_history[epoch_loss_start:]
        epoch_train_loss = sum(epoch_losses) / len(epoch_losses)
        print(
            f"Epoch {epoch}/{max_epochs} | train loss: {epoch_train_loss:.4f} | val loss: {val_item:.4f}"
        )

        # Fixed seed + prompt: samples comparable across epochs. The RNG
        # state is saved/restored so seeded sampling never disturbs the
        # training (dropout) stream.
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if device == "cuda" else None
        torch.manual_seed(1337)
        sample_text = generate(
            model,
            tokenizer,
            SAMPLE_PROMPT,
            max_new_tokens=20,
            temperature=0.8,
            device=device,
        )
        model.train()
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
        print(f"  sample: {sample_text[:100]}")

        if val_item < best_val_loss:
            best_val_loss = val_item
            epochs_without_improvement = 0
            # Atomic, versioned save; config round-trips via GPTConfig
            # (see model/checkpoint.py).
            save_checkpoint(
                model,
                model_path,
                losses=loss_history,
                optimizer_state_dict=optimizer.state_dict(),
                loss=loss.item(),
                val_loss=val_item,
                epoch=epoch,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stop: val loss flat for {patience} epochs")
                break

    # Generate with the best checkpoint, not the last state
    model, checkpoint = load_checkpoint(model_path, device=device)
    model.eval()

    print(
        f"\nBest val loss: {best_val_loss:.4f} (epoch {checkpoint.get('epoch', '?')})"
    )
    print(f"Model saved to {model_path}")

    print("\nText generation sample:")
    generated_text = generate(
        model, tokenizer, DEMO_PROMPT, max_new_tokens=20, temperature=0.7, device=device
    )
    print(f"Prompt: {DEMO_PROMPT}")
    print(f"Generated: {generated_text}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")
    print(f"Model ready for inference!")

    return model_path


if __name__ == "__main__":
    train_small_model()
