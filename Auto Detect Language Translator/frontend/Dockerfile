# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10.15
FROM python:${PYTHON_VERSION}-bullseye as base

# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures logs appear immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies: curl and ping
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl iputils-ping && \
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

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies efficiently
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER appuser

# Copy frontend source
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "front.py", "--server.address=0.0.0.0", "--server.port=8501"]

