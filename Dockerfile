FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    CINEMATCH_ENV=production \
    CINEMATCH_HOST=0.0.0.0 \
    CINEMATCH_PORT=5000 \
    CINEMATCH_DEBUG=false \
    CINEMATCH_DATASET_PATH=/app/data/movies_metadata.csv \
    GUNICORN_WORKERS=2 \
    GUNICORN_TIMEOUT=120 \
    GUNICORN_GRACEFUL_TIMEOUT=30

WORKDIR /app

# Create non-privileged application user
RUN groupadd --gid 10001 cinematch && \
    useradd --uid 10001 --gid cinematch --shell /bin/bash --create-home cinematch

# Install minimal runtime dependencies including Gunicorn WSGI server
COPY pyproject.toml ./
COPY src ./src
COPY frontend ./frontend
COPY app.py recommender.py ./

RUN pip install --no-cache-dir . && \
    mkdir -p /app/data /app/models && \
    chown -R cinematch:cinematch /app

USER cinematch

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; res = urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=4); sys.exit(0 if res.status == 200 else 1)"

CMD ["sh", "-c", "exec gunicorn --bind ${CINEMATCH_HOST:-0.0.0.0}:${CINEMATCH_PORT:-5000} --workers ${GUNICORN_WORKERS:-2} --timeout ${GUNICORN_TIMEOUT:-120} --graceful-timeout ${GUNICORN_GRACEFUL_TIMEOUT:-30} --preload --access-logfile - --error-logfile - cinematch.wsgi:app"]
