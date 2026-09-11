# Serves the API only - the web client (web/) is a separate static SPA with
# its own Dockerfile (web/Dockerfile), built with Node/nginx, not Python.
FROM python:3.12-slim

WORKDIR /app

# libgomp1: required by torch/faiss-cpu wheels (OpenMP) at import time, not
# just build time - must be in the final image, not just a build stage.
# curl: used by the Compose healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
# The evaluation dataset is small and git-committed (unlike data/corpus,
# data/processed, indexes/, which are regenerated artifacts and gitignored),
# so `docker compose run --rm api python scripts/evaluate_*.py` works
# without a volume mount. A mount over /app/data at runtime (see
# docker-compose.yml) still takes precedence when present.
COPY data/evaluation/ data/evaluation/

# All extras needed to (a) serve the API and (b) build the index via
# scripts/*.py as a one-off command if needed - not `eval`/`dev`, which
# this image never needs to run.
RUN pip install --no-cache-dir ".[ingestion,tokenization,ml,api,llm]"

EXPOSE 8000

CMD ["uvicorn", "cardiorag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
