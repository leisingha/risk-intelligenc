# syntax=docker/dockerfile:1
# ---- builder: resolve and install dependencies into a venv ------------------------
FROM python:3.11-slim AS builder
# CPU-only torch wheels (~200MB) instead of the CUDA build (~2.5GB). Set to "" to use PyPI.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt .
RUN if [ -n "$TORCH_INDEX" ]; then \
        pip install --extra-index-url "$TORCH_INDEX" -r requirements.txt; \
    else \
        pip install -r requirements.txt; \
    fi

# ---- runtime: slim image, non-root user, models/index mounted at run time --------
FROM python:3.11-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/hf-cache \
    ANONYMIZED_TELEMETRY=False
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src ./src
COPY --chown=app:app data/eval ./data/eval
COPY --chown=app:app data/labels ./data/labels
RUN mkdir -p models .chroma data/processed && chown -R app:app /app
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=4)"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
