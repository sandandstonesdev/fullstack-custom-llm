# Toolchain: uv with pyproject.toml + uv.lock (pinned, committed).
# `uv run` syncs the .venv to the lockfile automatically - the same lock
# the Dockerfile uses, so dev and prod install identical versions.
# Override locally with: make PY=python <target> (inside an active venv).
#
# Run all targets from backend/ (the uv project root).
PY ?= uv run python

.PHONY: corpus tokenizer dataset train serve test up deploy

# One-time corpus build (curated mythological-literary collection:
# Greek/Roman antiquity, Baltic legends, Egypt - see MYTHOLOGY_BOOKS).
# Standard-library only - runs even without the ML deps installed.
corpus:
	$(PY) -m cli --build_corpus_lektury

tokenizer:
	$(PY) -m cli --train_tokenizer

dataset:
	$(PY) -m cli --prepare_dataset

# Full local training pipeline: tokenizer -> shards -> checkpoint.
train:
	$(PY) -m cli --train_tokenizer --prepare_dataset --train_model

# Serving (dev). Valkey optional: podman run -p 6379:6379 -d valkey/valkey
serve:
	$(PY) -m uvicorn app.main:app --reload

test:
	$(PY) -m pytest tests/

# Full stack via podman compose (backend + frontend + valkey); repo-level.
up:
	podman compose -f ../deploy/compose/docker-compose.yml up --build

deploy:
	cd ../deploy/tofu && tofu apply
