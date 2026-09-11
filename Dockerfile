# One image serves both the API and the UI (different `command:` per
# docker-compose service) rather than two separate Dockerfiles: simpler to
# build and maintain for a project this size, at the cost of the UI image
# also carrying the ML stack (torch/faiss/transformers) it doesn't actually
# use at runtime - the UI is a pure HTTP client of the API. A reasonable
# tradeoff to revisit if image size ever becomes a real constraint.
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
COPY app/ app/
COPY scripts/ scripts/
# The evaluation dataset is small and git-committed (unlike data/corpus,
# data/processed, indexes/, which are regenerated artifacts and gitignored)
# so the UI's "try a real example question" feature works even without a
# volume mount. A mount over /app/data at runtime (see docker-compose.yml)
# still takes precedence when present.
COPY data/evaluation/ data/evaluation/

# All extras needed to (a) serve the API, (b) run the Streamlit UI, and
# (c) build the index via scripts/*.py as a one-off command if needed -
# not `eval`/`dev`, which this image never needs to run.
RUN pip install --no-cache-dir ".[ingestion,tokenization,ml,api,ui,llm]"

EXPOSE 8000 8501

CMD ["uvicorn", "cardiorag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
