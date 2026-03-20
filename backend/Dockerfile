# Backend Dockerfile - BiLSTM-CRF NER API
# Multi-stage build để giảm image size

# Stage 1: Build
FROM python:3.10-slim AS builder

WORKDIR /tmp

# Install Java runtime (cần cho VnCoreNLP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy unified requirements from root
COPY requirements.txt .

# Create wheels for faster installation
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /tmp/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install Java runtime and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /tmp/wheels /wheels
COPY --from=builder /tmp/requirements.txt .

# Install from wheels
RUN pip install --upgrade pip && \
    pip install --no-cache /wheels/*

# Copy application code from backend directory
COPY backend/app.py .
COPY backend/predictor.py .

# Copy artifacts folder
COPY backend/artifacts ./artifacts

# Copy VnCoreNLP
COPY backend/vncorenlp ./vncorenlp

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000')" || exit 1

# Run Flask app
CMD ["python", "app.py"]
