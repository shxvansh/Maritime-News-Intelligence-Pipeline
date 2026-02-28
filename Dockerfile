# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PostgreSQL, building ML packages, and spaCy
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching layers
COPY requirements.txt .

# Install Python dependencies
# Adding PyTorch explicitly for CPU before other requirements to save space/time and ensure GLiNER finds it
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy language model required for preprocessing
RUN python -m spacy download en_core_web_sm

# Pre-download the HuggingFace zero-shot classifier model and GLiNER down into the image
# This ensures a faster startup time on the first run rather than downloading 2GB of weights every boot
RUN python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"
RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_medium-v2.1')"

# Copy the rest of the application codebase
COPY . .

# Set environment variables to optimize Python performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# By default, running the container will execute the main pipeline
CMD ["python", "pipeline/main.py"]
