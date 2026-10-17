"""Checkpoint format resilience: checkpoints must keep loading as the
model code evolves (see model/checkpoint.py)."""
import torch

from model.checkpoint import save_checkpoint, load_checkpoint
from model.gpt import GPT, GPTConfig


def small_config(**overrides):
    return GPTConfig(vocab_size=100, block_size=64, n_embd=64, n_head=1, n_layer=1, **overrides)


def rewrite(path, mutate):
    """Load a checkpoint raw, mutate it, save it back: simulates a file
    written by a different (older/newer) version of the code."""
    checkpoint = torch.load(path, weights_only=True)
    mutate(checkpoint)
    torch.save(checkpoint, path)


def test_round_trip(tmp_path):
    model = GPT(config=small_config())
    path = str(tmp_path / "model.pt")
    save_checkpoint(model, path, val_loss=1.25, epoch=3)

    loaded, checkpoint = load_checkpoint(path)
    assert checkpoint["val_loss"] == 1.25
    assert checkpoint["epoch"] == 3
    assert checkpoint["format"] == 1
    assert loaded.config == model.config
    for (name, param), (lname, lparam) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert name == lname
        assert torch.equal(param, lparam)
    # atomic write leaves no temp file behind
    assert not (tmp_path / "model.pt.tmp").exists()


def test_future_config_fields_are_ignored(tmp_path):
    model = GPT(config=small_config())
    path = str(tmp_path / "model.pt")
    save_checkpoint(model, path)
    # a newer GPTConfig gained fields after this checkpoint was written
    rewrite(path, lambda ck: ck["config"].update(future_field=1))
    loaded, _ = load_checkpoint(path)
    assert loaded.config == model.config


def test_renamed_state_keys_warn_but_load(tmp_path):
    model = GPT(config=small_config())
    path = str(tmp_path / "model.pt")
    save_checkpoint(model, path)
    # every weight renamed: loader must report and serve, not crash
    rewrite(path, lambda ck: ck.update(
        state_dict={f"{k}_old": v for k, v in ck["state_dict"].items()}
    ))
    loaded, _ = load_checkpoint(path)
    idx = torch.zeros(1, 8, dtype=torch.long)
    logits, _ = loaded.eval()(idx)  # forward pass still runs
    assert logits.shape == (1, 8, 100)


def test_pre_config_checkpoint_uses_defaults(tmp_path):
    model = GPT(config=GPTConfig())  # written before config was stored
    path = str(tmp_path / "model.pt")
    save_checkpoint(model, path)
    rewrite(path, lambda ck: ck.pop("config"))
    loaded, _ = load_checkpoint(path)
    assert loaded.config == GPTConfig()
    for (name, param), (lname, lparam) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert name == lname
        assert torch.equal(param, lparam)


def test_optional_metadata_may_be_absent(tmp_path):
    model = GPT(config=small_config())
    path = str(tmp_path / "model.pt")
    save_checkpoint(model, path)
    rewrite(path, lambda ck: [ck.pop(k) for k in ("loss", "val_loss") if k in ck])
    _, checkpoint = load_checkpoint(path)
    assert "loss" not in checkpoint and "val_loss" not in checkpoint


def test_invalid_checkpoint_rejected(tmp_path):
    path = str(tmp_path / "model.pt")
    torch.save({"not": "a checkpoint"}, path)
    try:
        load_checkpoint(path)
        assert False, "invalid checkpoint accepted"
    except ValueError as exc:
        assert "state_dict" in str(exc)
