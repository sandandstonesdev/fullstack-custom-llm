"""Generation controls: repetition penalty + stop token."""
import pytest
import torch

from model.gpt import GPT, GPTConfig, apply_repetition_penalty


def tiny_model():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=1000, block_size=64, n_embd=64, n_head=1, n_layer=1)
    return GPT(cfg).eval()


def test_penalty_formula():
    # positive logits get divided, negative multiplied, unseen untouched
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    idx = torch.tensor([[0, 1]])
    out = apply_repetition_penalty(logits, idx, 2.0)
    assert out[0, 0] == 1.0      # 2.0 / 2
    assert out[0, 1] == -4.0     # -2.0 * 2
    assert out[0, 2] == 1.0      # not in context: unchanged


def test_penalty_rejected_below_one():
    with pytest.raises(ValueError):
        tiny_model().generate(torch.zeros(1, 4, dtype=torch.long),
                              max_new_tokens=1, temperature=1.0, repetition_penalty=0.9)


def test_penalty_breaks_greedy_loops():
    # greedy (temperature -> 0) with a huge penalty must avoid repeating
    # a context token that would otherwise be the argmax
    model = tiny_model()
    x = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(x)
        argmax = logits[:, -1].argmax().item()
        x = torch.cat([x, torch.tensor([[argmax]])], dim=1)  # argmax now in context
        out_free = model.generate(x, max_new_tokens=5, temperature=1e-9)
        out_pen = model.generate(x, max_new_tokens=5, temperature=1e-9,
                                 repetition_penalty=1e6)
    assert (out_free[0, 8:] == argmax).all()          # without penalty: pure loop
    assert argmax not in out_pen[0, 9:].tolist()      # with penalty: loop broken


def test_stop_token_stops_and_is_not_emitted():
    model = tiny_model()
    x = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(x)
        forced_stop = logits[:, -1].argmax().item()
        out = model.generate(x, max_new_tokens=50, temperature=1e-9, stop_token=forced_stop)
    assert out.shape[1] == 8
    assert forced_stop not in out[0, 8:].tolist()

    with pytest.raises(ValueError):
        model.generate(torch.zeros(2, 4, dtype=torch.long), max_new_tokens=1,
                       temperature=1.0, stop_token=7)   # batch > 1 rejected
