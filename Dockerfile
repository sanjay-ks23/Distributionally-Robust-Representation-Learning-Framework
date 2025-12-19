# DRRL Framework Docker Image
# Multi-stage build for smaller image size

# Base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories
RUN mkdir -p /app/data /app/logs /app/outputs /app/plots

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "scripts/train.py", "--help"]

# Entrypoints for common tasks
# Training: docker run drrl python scripts/train.py --method erm
# Evaluation: docker run drrl python scripts/evaluate.py --checkpoint outputs/best_model.pt
# Tests: docker run drrl pytest tests/ -v
