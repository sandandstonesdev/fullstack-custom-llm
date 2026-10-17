# Serving image for the Polish LLM API - CPU-only, rootless-friendly.
# Build from the REPO ROOT (needs backend/, the trained artifacts,
# and the pinned lockfile):
#   podman build -f backend/Dockerfile -t polish-llm-api .
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv for locked, reproducible dependency installs (same lock as dev).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install only the pinned runtime dependencies from uv.lock.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# The whole backend project: /app mirrors backend/ (the project root),
# so modules import top-level (app, configs, model, ...).
COPY backend/ ./

# Bake in the trained artifacts (small model: baking in beats PVC/object store).
COPY data/processed/models/polish_small_model.pt ./data/processed/models/
COPY data/processed/tokenizers/polish_bpe_4k.json ./data/processed/tokenizers/

# Non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
