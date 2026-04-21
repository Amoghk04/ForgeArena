FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification first for layer caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install core package (no training extras — too large for the Space)
RUN pip install --no-cache-dir -e "."

# Copy remaining project files
COPY openenv.yaml ./
COPY tasks/ ./tasks/

# HuggingFace Spaces expects the app to listen on port 7860
ENV PORT=7860
EXPOSE 7860

# Single worker for demo — swap for gunicorn in production
CMD ["uvicorn", "forge_arena.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
