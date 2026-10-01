# syntax=docker/dockerfile:1.10
# Serving image: charade + runtime deps only (no dev tools, no training frameworks).

FROM ghcr.io/astral-sh/uv:0.12.9-python3.12-trixie-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY tools/mlcheck/pyproject.toml tools/mlcheck/README.md tools/mlcheck/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.12-slim-trixie AS runtime
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app
WORKDIR /app
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app pyproject.toml ./
# One CPU thread per worker for every library: request-level parallelism comes from uvicorn workers,
# and per-library thread pools (polars defaults to one thread per core) only add contention.
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=2s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["uvicorn", "charade.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
