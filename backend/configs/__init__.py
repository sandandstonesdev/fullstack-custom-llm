"""
Shared configuration: project paths and serving overrides.

Model hyperparameters live in ONE place: GPTConfig in model/gpt.py.
This module re-exports it lazily (module __getattr__), so importing
paths from here does NOT pull in torch - corpus building and other
stdlib-only commands stay fast and dependency-free.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "polish_corpus.txt"
PROCESSED_DIR = DATA_DIR / "processed"
# Single source of truth for the BPE vocab size: the tokenizer filename
# is derived from it, so the trainer and the saved path can never drift.
TOKENIZER_VOCAB_SIZE = 4096
TOKENIZER_PATH = PROCESSED_DIR / "tokenizers" / f"polish_bpe_{TOKENIZER_VOCAB_SIZE // 1000}k.json"
SHARDS_DIR = PROCESSED_DIR / "shards"
# Shard naming convention shared by the writer (dataset.prepare_dataset)
# and the reader (train.dataloader): f"{SHARD_NAME}_{split}_{index:06d}.npy".
# One constant, so the two sides can never drift apart. The corpus comes
# from Wolne Lektury, hence the name.
SHARD_NAME = "lektury"
MODEL_PATH = PROCESSED_DIR / "models" / "polish_small_model.pt"

# Serving overrides (env vars win over the defaults above).
SERVE_MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(MODEL_PATH)))
_redis_host = os.environ.get("REDIS_HOST", "127.0.0.1")
_redis_port = os.environ.get("REDIS_PORT", "6379")
VALKEY_URL = os.environ.get(
    "VALKEY_URL", f"redis://{_redis_host}:{_redis_port}/0"
)
CORS_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
]
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", "40"))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.5"))
# CTRL-style repetition penalty (1.0 = off): tames "ze nie, ze nie" loops
# in small undertrained models without banning repeats outright.
GEN_REPETITION_PENALTY = float(os.environ.get("GEN_REPETITION_PENALTY", "1.2"))


def __getattr__(name):
    # Lazy re-export: `from configs import GPTConfig` works, but only
    # pulls in torch when GPTConfig is actually requested.
    if name == "GPTConfig":
        from model.gpt import GPTConfig
        return GPTConfig
    raise AttributeError(f"module 'config' has no attribute {name!r}")
