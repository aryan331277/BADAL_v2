# Use Python slim image to reduce base size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations and longer timeout
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Copy application files
COPY . .

# Create directory for model cache
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models

# Pre-download the model during build to avoid runtime delays
RUN python startup.py

# Expose port
EXPOSE 8080

# Use gunicorn for production with health check
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "--keep-alive", "2", "--max-requests", "1000", "--preload", "app:app"]
