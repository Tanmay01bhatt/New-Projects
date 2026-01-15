# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10.15
FROM python:${PYTHON_VERSION}-bullseye as base

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Unbuffered output for logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Optional: Install system dependencies if needed
# e.g., gcc, libpq-dev, curl for backend tasks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies efficiently
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER appuser

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run backend app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
