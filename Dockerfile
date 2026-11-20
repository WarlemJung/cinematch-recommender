FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Create non-privileged application user
RUN groupadd --gid 10001 cinematch && \
    useradd --uid 10001 --gid cinematch --shell /bin/bash --create-home cinematch

# Install minimal runtime dependencies
COPY pyproject.toml ./
COPY src ./src
COPY frontend ./frontend
COPY app.py recommender.py ./

RUN pip install --no-cache-dir . && \
    mkdir -p /app/data /app/models && \
    chown -R cinematch:cinematch /app

USER cinematch

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; res = urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=4); sys.exit(0 if res.status in (200, 503) else 1)"

CMD ["python", "-m", "cinematch.main"]
