# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PostgreSQL and building ML packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching layers
COPY requirements.txt .

# Install PyTorch CPU-only explicitly first (saves ~2GB vs default GPU build)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install hdbscan separately with a GCC 14 compatibility flag.
# GCC 14 (Debian Trixie) treats incompatible pointer types as hard errors.
# This flag downgrades them to warnings so hdbscan compiles correctly.
RUN CFLAGS="-Wno-incompatible-pointer-types" pip install --no-cache-dir hdbscan

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy language model (small, ~15MB — safe to bake in)
RUN python -m spacy download en_core_web_sm

# NOTE: Large ML models (BART ~1.6GB, GLiNER ~400MB, BGE ~1.3GB) are NOT baked
# into the image to keep image size small. They are downloaded on first container
# startup and cached in the 'huggingface_cache' Docker volume defined in
# docker-compose.yml, so subsequent restarts are instant.

# Copy the rest of the application codebase
COPY . .

# Set environment variables to optimize Python performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Point HuggingFace cache to the mounted volume path
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

# By default, running the container will execute the main pipeline
CMD ["python", "pipeline/main.py"]
