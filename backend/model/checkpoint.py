"""
Checkpoint save/load - the single place that knows the on-disk format.

Resilience to code changes, in both directions:
- 'format' field lets future loaders recognize (and migrate) older files.
- The stored config round-trips through GPTConfig.from_dict, which
  ignores unknown keys: adding fields to GPTConfig never breaks loading
  existing checkpoints.
- state_dict keys are reconciled, not enforced: renamed/added/removed
  modules produce loud warnings instead of a crash. Weights the model no
  longer knows are dropped; weights missing from the file keep their
  fresh initialization, so the model serves degraded instead of dead.
  Shapes must still match exactly - a genuinely incompatible
  architecture raises, as it should.
- Writes are atomic (temp file + os.replace): the checkpoint is never
  left half-written on disk if the process dies mid-save.
"""

import os
from typing import Any, Tuple

import torch

from model.gpt import GPT, GPTConfig

# Bump when the on-disk payload layout changes; load_checkpoint keeps
# accepting older formats.
CHECKPOINT_FORMAT = 1


def save_checkpoint(model, path: str, **extra: Any) -> str:
    """Atomically save model + config + extra metadata to path."""
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        **extra,
    }
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    # Atomic on both POSIX and Windows: readers never observe a partial file.
    os.replace(tmp_path, path)
    return path


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[GPT, dict]:
    """Load a checkpoint into a freshly built model.

    Rebuilds the architecture from the stored config (falls back to
    GPTConfig defaults for pre-config checkpoints), then loads the
    weights with key reconciliation (see module docstring).

    Returns (model, checkpoint_dict).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a valid checkpoint: no 'state_dict'")

    # Config round-trip: unknown keys from newer GPTConfig versions are
    # ignored; checkpoints saved before config was stored use defaults.
    if "config" in checkpoint:
        cfg = GPTConfig.from_dict(checkpoint["config"])
    else:
        print(f"checkpoint: no stored config in {path}; using GPTConfig defaults")
        cfg = GPTConfig()

    model = GPT(config=cfg).to(device)

    # strict=False tolerates added/renamed/removed modules; shape
    # mismatches still raise inside load_state_dict.
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if unexpected:
        print(
            f"checkpoint: WARNING {len(unexpected)} weights in {path} are not used "
            "by the current model code (removed/renamed modules); they were "
            f"dropped, e.g. {sorted(unexpected)[:5]}"
        )
    if missing:
        print(
            f"checkpoint: WARNING {len(missing)} weights expected by the current "
            f"model code are missing from {path} (new/renamed modules); they "
            f"keep their fresh initialization, e.g. {sorted(missing)[:5]}"
        )

    checkpoint.setdefault("format", 0)
    return model, checkpoint
