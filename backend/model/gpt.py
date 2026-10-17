"""
GPT model implementation for Polish language modeling.

GPTConfig is the single source of truth for model hyperparameters:
train/train.py and model/inference.py both consume it, and
checkpoints store its dict form (to_dict/from_dict), so a retrain with
different sizes never breaks loading.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    # architecture - small by design, trains on a laptop CPU
    vocab_size: int = 4096  # BPE vocab (overridden with the trained tokenizer's size)
    block_size: int = 256  # context length; corpus samples average ~300 tokens
    n_embd: int = 128  # n_head * 32 -> head_dim = 32
    n_head: int = 4
    n_layer: int = 2
    # regularization
    dropout: float = 0.1
    attn_bias: bool = False  # no bias on attention projections (efficiency choice);
    # the MLP always has biases (GPT-2 convention)

    def __post_init__(self):
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.block_size >= 64, "block_size must be at least 64"
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        # Ignore unknown keys -> old checkpoints still load after new fields.
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.attn_bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
        # regularization: attention-weight dropout runs INSIDE the fused
        # kernel (dropout_p); the residual dropout stays a module.
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dim

        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=-1)  # (B,T,C) each

        # reshape for multi-head attention: split embedding dim into multiple heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # fused causal attention (flash/mem-efficient on CUDA, oneDNN on
        # CPU); numerically identical to the manual implementation it
        # replaced, and it never materializes the (B, nh, T, T) matrix.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    A simple linear layer followed by a non-linearity (GELU).
    """

    def __init__(self, config):
        super().__init__()
        # the MLP always keeps its biases (GPT-2 convention)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.gelu = nn.GELU(approximate="tanh")  # the GPT-2 paper's approximation
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """An unassuming Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=True)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def apply_repetition_penalty(logits, idx, penalty):
    """CTRL-style repetition penalty: positive logits of tokens already in
    the context get divided, negative ones multiplied by ``penalty`` (>= 1),
    making repeats less likely without banning them outright."""
    for b in range(logits.size(0)):
        seen = torch.unique(idx[b])
        vals = logits[b, seen]
        logits[b, seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


class GPT(nn.Module):
    """
    The full GPT language model, with a context size of block_size.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=True),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: the output head shares the token embedding matrix
        # (GPT-2 convention; saves vocab_size * n_embd parameters)
        self.lm_head.weight = self.transformer.wte.weight

        # init all weights, with the GPT-2 paper's scaled init on the
        # residual projections
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def num_params(self, non_embedding=True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"sequence length {t} should be <= {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        stop_token=None,
        repetition_penalty=1.0,
    ):
        """
        Generate text from the model.

        Call model.eval() first so dropout is off. Contexts longer than
        block_size are cropped to its last block_size tokens. When
        ``stop_token`` is drawn (e.g. the document-boundary EOT), sampling
        stops early without emitting it (single-sequence input only).
        ``repetition_penalty`` (> 1.0) penalizes tokens already in the
        context (see apply_repetition_penalty). No KV cache: each step
        recomputes the whole prefix (~2 ms/token for the 1M-param model).
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if repetition_penalty < 1.0:
            raise ValueError(
                f"repetition_penalty must be >= 1.0, got {repetition_penalty}"
            )
        if stop_token is not None and idx.size(0) > 1:
            raise ValueError(
                "stop_token is only supported for single-sequence input (B=1)"
            )
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens if it exceeds
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]  # logits at the final step
            # penalize repeats BEFORE temperature scaling (standard order)
            if repetition_penalty > 1.0:
                logits = apply_repetition_penalty(logits, idx_cond, repetition_penalty)
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # document boundary: stop without emitting it
            if stop_token is not None and idx_next.item() == stop_token:
                break
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
