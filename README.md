# Polish LLM Chat Application

Real-time chat with a small Polish language model: a FastAPI backend (WebSocket), an Astro+React frontend, and a local pipeline that turns public-domain literature into the served checkpoint.

The corpus is a curated mythological-literary collection from [Wolne Lektury](https://wolnelektury.pl) (HF: [ArielUW/wolnelektury](https://huggingface.co/datasets/ArielUW/wolnelektury)): Greek and Roman antiquity (Homer, Ajschylos, Sofokles, Eurypides, Arystofanes, Wergiliusz, Owidiusz, Horacy), mythographic studies (Zipper, Brückner), Baltic legends (Stryjkowski, Mickiewicz, Baranauskas, Asnyk, Słowacki), Eastern accents (Leśmian, Kasprowicz, Żmichowska) and ancient Egypt (Prus's *Faraon*) — 38 books ≈ 5,700 samples ≈ 1.5M tokens. Small enough to train on a laptop CPU, themed enough to evaluate by eye.

## Architecture

```
Frontend (Astro+React) ⇄ WebSocket /ws ⇄ Backend (FastAPI) ⇄ Valkey (event logging)
```

```
corpus → tokenizer → shards → train ─► polish_small_model.pt (baked into the serve image)
```

- Training runs locally only; serving is stateless — model + tokenizer are baked into the image.
- The two sides meet in exactly one file: the checkpoint `data/processed/models/polish_small_model.pt`.
- Corpus builder: `backend/corpus/` (the `MYTHOLOGY_BOOKS` list lives there); tokenizer/shards/model/train: `backend/tokenizer/`, `backend/dataset/`, `backend/model/`, `backend/train/`; serving: `backend/app/` + `frontend/`; all artifacts land in gitignored `data/`.

## Commands (all from `backend/`)

### 1. Install

```bash
uv sync
```

### 2. Full pipeline — corpus → train → check (~15 min on a laptop CPU)

```bash
uv run python -m cli --build_corpus_lektury                                # mythology corpus → data/raw/polish_corpus.txt
uv run python -m cli --train_tokenizer --prepare_dataset --train_model     # tokenizer + shards + checkpoint
uv run python -m cli --inference "Antygona"                                # quick generation check
```

### 3. Smoke test first (seconds instead of ~11 min of training)

```bash
uv run python -m cli --build_corpus_lektury --corpus_samples 500          # small scratch corpus (overwrites the real one — rebuild it afterwards)
uv run python -m cli --train_tokenizer --prepare_dataset
uv run python -m cli --train_model --max_shards 2 --max_epochs 1 --model_path ../data/smoke_model.pt
uv run python -m cli --inference "Antygona" --model_path ../data/smoke_model.pt
```

### 4. Serve

```bash
podman run -p 6379:6379 -d valkey/valkey        # optional — chat event logging only
uv run python -m uvicorn app.main:app --reload  # backend :8000
cd ../frontend && pnpm install && pnpm run dev  # frontend http://localhost:3000
```

### 5. Deploy

```bash
podman compose -f deploy/compose/docker-compose.yml up --build   # backend :8000, frontend :3000, valkey :6379
```

### Corpus options (`--build_corpus_lektury`)

| Flag | Effect |
|---|---|
| *(none)* | the curated mythology collection (default) |
| `--lektury_list` | list the available books |
| `--lektury_books "faraon,homer-iliada"` | a substring-matched subset instead (`faraon` → all three *Faraon* volumes) |
| `--lektury_all` | all ~2400 books in the dataset (combine with `--corpus_samples`) |
| `--corpus_samples N` | cap on the sample count (default: no cap) |

Useful training knobs: `--max_shards` (smaller corpus), `--batch_size`, `--max_epochs`, `--learning_rate`, `--patience`. Serving env knobs: `MODEL_PATH`, `VALKEY_URL`, `CORS_ORIGINS`, `API_HOST`/`API_PORT`, `GEN_MAX_TOKENS`, `GEN_TEMPERATURE`, `GEN_REPETITION_PENALTY`.

### Testing

```bash
uv run python -m pytest tests/
```

## License

MIT License

## Authors

- **Paweł Aksiutowicz** - Lead Developer

## Acknowledgments

- Code generation and optimization supported by **Mistral AI**
- Corpus texts courtesy of **Wolne Lektury** ([wolnelektury.pl](https://wolnelektury.pl)) and the [ArielUW/wolnelektury](https://huggingface.co/datasets/ArielUW/wolnelektury) dataset on Hugging Face
- Special thanks for invaluable resources, code and strong support in learning:
  - Arkadiusz Słota [@Maggio333](https://github.com/Maggio333/)
  - Kacper Wikieł [@kwikiel](https://github.com/kwikiel/)
  - **Slayer AI Labs** [@slayerlabs](https://github.com/slayerlabs)
