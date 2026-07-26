# Polish LLM Chat Application

A real-time chat application with a Polish language model, built with FastAPI backend and Astro+React frontend.

## Quick Start

### 1. Start Redis
```bash
redis-server

```Redis Server in Docker/Podman
docker run --name valkey -p 6379:6379 -d valkey/valkey
```

### 2. Run Backend
```bash
cd backend
uv pip install -r requirements.txt
uv run python api/main.py
```

### 3. Run Frontend (npm/pnpm)
```bash
cd frontend
npm install
npm run build
npm run dev
```

### 4. Open Browser
Visit: `http://localhost:3000`

## Features

- Real-time chat with WebSocket
- Polish language model integration
- Clean chat history (🗑️ button)
- Message storage in Valkey (Redis)
- Responsive design with Tailwind CSS

## Architecture

```
Frontend (React+Astro) → WebSocket → Backend (FastAPI) → Valkey (Redis)
```

### Model Training & Inference

## Tokenizer Training
# Input Corpus file 'polish_corpus.txt'
# Tokenizer Output polish_bpe_32k.json

```bash
cd backend
python main.py --train_tokenizer
```

## Dataset Preparation
# polish_wiki_val_000000.npy
# polish_wiki_val_000000.npy

```bash
python main.py --prepare_dataset
```

## Model Training
# polish_small_model.pt

```bash
python main.py --train_model
```

### Inference
```bash
python main.py --inference "Your prompt here"
```

## Data Structure

```
data/
├── raw/
│   └── polish_corpus.txt      # Raw Polish text data
└── processed/
    ├── models/
    │   ├── tokenizers/       # Trained tokenizers
    │   │   └── polish_bpe_32k.json
    │   └── polish_small_model.pt  # Trained model
    └── shards/               # Tokenized dataset shards
```

## Configuration

Edit `backend/configs/tokenizer_config.json`:

```json
{
    "vocab_size": 32000,
    "min_frequency": 2,
    "special_tokens": ["<|endoftext|>", "[UNK]", "[PAD]"]
}
```

## Requirements

### Backend
- Python 3.10+
- PyTorch (with CUDA for GPU acceleration)
- HuggingFace tokenizers
- NumPy
- FastAPI
- Redis

### Frontend
- Node.js 16+
- npm or pnpm

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests (TODO)
cd frontend
npm test
```

### Building for Production

```bash
# Frontend build
cd frontend
pnpm run build

# Backend (use production server)
uv run --host 0.0.0.0 --port 8000 api.main:app
```

## Deployment

### Docker

```bash
# Build images
docker build -t polish-llm-frontend:latest frontend/
docker build -t polish-llm-backend:latest backend/

# Run containers
docker run -p 3000:80 polish-llm-frontend:latest
docker run -p 8000:8000 polish-llm-backend:latest
```

### Kubernetes (TODO)
```bash
kubectl apply -f frontend/kubernetes/
kubectl apply -f backend/kubernetes/
```

### OpenTofu (TODO)
```bash
tofu apply -f infra/
```


## Troubleshooting

### Model Training
1. **CUDA Out of Memory**: Reduce batch size
2. **Tokenizer Errors**: Verify UTF-8 encoding
3. **Slow Training**: Use GPU acceleration

### Chat Application
1. **WebSocket Failed**: Verify backend running
2. **No Response**: Check backend logs
3. **Redis Errors**: Ensure Redis is running

## License

MIT License

## Authors

- **Paweł Aksiutowicz** - Lead Developer

## Acknowledgments

- Code generation and optimization supported by **Mistral AI**
- Special thanks for invaluable resources, code and strong support in learning:
  - Arkadiusz Słota [@Maggio333](https://github.com/Maggio333/)
  - Kacper Wikieł [@kwikiel](https://github.com/kwikiel/)
  - **Slayer AI Labs** [@slayerlabs](https://github.com/slayerlabs)