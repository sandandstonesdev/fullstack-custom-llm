"""GPTConfig round-trip: the single source of truth for model hyperparameters."""
import pytest

from model.gpt import GPTConfig


def test_config_round_trip():
    cfg = GPTConfig(vocab_size=1000)
    assert GPTConfig.from_dict(cfg.to_dict()) == cfg
    # Unknown keys from future config versions are ignored, not fatal.
    assert GPTConfig.from_dict({**cfg.to_dict(), "future_field": 1}) == cfg


def test_config_validation():
    assert GPTConfig().head_dim == 32
    with pytest.raises(AssertionError):
        GPTConfig(block_size=32)          # too short
    with pytest.raises(AssertionError):
        GPTConfig(n_embd=100, n_head=6)   # not divisible
