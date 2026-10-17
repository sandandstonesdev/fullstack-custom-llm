"""Inference for the Polish language model: load, generate, interactive loop."""
import os
from typing import Optional

import torch
from tokenizers import Tokenizer

from configs import (
    GEN_MAX_TOKENS,
    GEN_REPETITION_PENALTY,
    GEN_TEMPERATURE,
    SERVE_MODEL_PATH,
    TOKENIZER_PATH,
)
from model.checkpoint import load_checkpoint
from tokenizer.train_tokenizer import EOT_TOKEN


def load_trained_model(model_path: Optional[str] = None):
    """Load model + tokenizer from ``model_path`` (default: the serving
    path, env-overridable); returns (None, None) with a hint on failure."""
    model_path = str(SERVE_MODEL_PATH if model_path is None else model_path)
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}; run: python -m cli --train_model")
        return None, None

    try:
        model, checkpoint = load_checkpoint(model_path)
        model.eval()

        tokenizer_path = str(TOKENIZER_PATH)
        if not os.path.exists(tokenizer_path):
            print(f"Tokenizer not found at {tokenizer_path}; run: python -m cli --train_tokenizer")
            return None, None
        tokenizer = Tokenizer.from_file(tokenizer_path)

        print(f"Loaded model from {model_path} "
              f"(~{sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
              f"vocab {tokenizer.get_vocab_size()})")
        # Metadata is optional: older/evolved checkpoint formats may lack it.
        if 'loss' in checkpoint:
            print(f"Train loss: {checkpoint['loss']:.4f} | "
                  f"val loss: {checkpoint.get('val_loss', float('nan')):.4f}")
        return model, tokenizer

    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Try training a new model: python -m cli --train_model")
        return None, None


def generate(model, tokenizer, prompt, max_new_tokens=None, temperature=None,
             repetition_penalty=None, device='cpu'):
    """Generate text from a prompt; defaults come from configs (env-overridable)."""
    model.eval()
    model.to(device)
    max_new_tokens = GEN_MAX_TOKENS if max_new_tokens is None else max_new_tokens
    temperature = GEN_TEMPERATURE if temperature is None else temperature
    repetition_penalty = (GEN_REPETITION_PENALTY if repetition_penalty is None
                         else repetition_penalty)

    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            # stop at the document boundary instead of rambling past it
            stop_token=tokenizer.token_to_id(EOT_TOKEN),
        )

    # Drop replacement characters; collapse all whitespace runs, not just doubles.
    return " ".join(tokenizer.decode(generated[0].tolist()).replace('�', '').split())


def main():
    print("Polish Language Model Inference")
    print("=" * 50)

    model, tokenizer = load_trained_model()
    if model is None:
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}. Type Polish prompts ('quit' to exit).")

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            if not prompt:
                continue
            print(f"\n{generate(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8, device=device)}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
