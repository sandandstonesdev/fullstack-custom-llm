
# Dockerfile for Polish LLM Backend API - CPU-only, Rootless-Friendly
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=off

# Install system dependencies (minimal for CPU-only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY tokenizer/ ./tokenizer/
COPY dataset/ ./dataset/
COPY model/ ./model/
COPY train/ ./train/
COPY data/ ./data/
COPY configs/ ./configs/
COPY main.py ./
COPY api/ ./api/

# Create directory for model files
RUN mkdir -p /app/data/processed/models/tokenizers

# Copy model files
COPY data/processed/models/tokenizers/polish_bpe_32k.json /app/data/processed/models/tokenizers/
COPY data/processed/models/polish_small_model.pt /app/data/processed/models/

# Add non-root user for security (rootless-friendly)
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden in docker-compose.yml)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]