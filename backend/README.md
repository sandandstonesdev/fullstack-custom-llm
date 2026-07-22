# Polish Language Model Training Pipeline

A complete pipeline for training Polish language models using PyTorch.

## Quick Start

```bash
# Install dependencies
uv pip install -r requirements.txt

# Train tokenizer and prepare dataset
uv run python main.py --train_tokenizer --prepare_dataset

# Train model
uv run python main.py --train_model

# Run inference
uv run python main.py --inference "Your prompt here"

#Run Valkey Container
podman run --name valkey -p 6379:6379 -d valkey/valkey

uv run python api/main.py

Runs on: `http://localhost:8000`
```

## Project Structure

```
data/
├── raw/polish_corpus.txt          # Polish text corpus
└── processed/models/tokenizers/polish_bpe_32k.json  # Tokenizer

src/
├── tokenizer/train_tokenizer.py  # Tokenizer training
├── dataset/prepare_dataset.py     # Dataset preparation
├── model/inference.py             # Inference logic
└── main.py                        # Main pipeline

configs/tokenizer_config.json      # Configuration
```

## Features

- **Tokenizer Training**: Polish BPE tokenizer (32K vocabulary)
- **Dataset Preparation**: Automatic corpus processing
- **Model Training**: GPT-style transformer architecture
- **Inference**: Interactive text generation interface
- **Testing**: Comprehensive test suite
- WebSocket API for real-time chat
- Polish language model integration
- Valkey (Redis) message storage
- CORS enabled for frontend

## Docker

```bash
docker build -t polish-llm-backend:latest .
docker run -p 8000:8000 polish-llm-backend:latest
```

## Kubernetes

```bash
kubectl apply -f kubernetes/
```

## Configuration

Edit `configs/tokenizer_config.json`:

```json
{
    "vocab_size": 32000,
    "min_frequency": 2,
    "special_tokens": ["<|endoftext|>", "[UNK]", "[PAD]"]
}
```

Create `.env` file:
```
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Testing

```bash
uv run pytest -v
```

## Requirements

- Python 3.10+
- PyTorch (with CUDA for GPU)
- HuggingFace tokenizers
- NumPy

## License

MIT License