# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base

# Install OS-level dependencies required by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source
COPY . .

# Default runtime environment
ENV PYTHONUNBUFFERED=1 \
    MODEL_FILES_DIR=/data/models \
    FORECAST_PLOTS_DIR=/data/plots

# Create data directories (they can be mounted as persistent volumes later)
RUN mkdir -p ${MODEL_FILES_DIR} ${FORECAST_PLOTS_DIR}

# Forecast mode is the recommended runtime configuration
CMD ["python3", "metrics.py", "--forecast", "--interval", "15"]


