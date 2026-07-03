# syntax=docker/dockerfile:1.7-labs

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    ENCODE_BATCH_SIZE=4 \
    MAX_SEQ_LENGTH=512 \
    TORCH_NUM_THREADS=2

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps with cache mounts for speed
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /app/requirements.txt

# Copy app code
COPY app /app/app

EXPOSE 8000

# Note: HF gated model requires HF_TOKEN env at runtime
ENV MODEL_ID=google/embeddinggemma-300m

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


