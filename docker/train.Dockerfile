# syntax=docker/dockerfile:1.10
# Training image: CPU torch + the offline stack + AWS CLI. Runs scripts/retrain.sh daily.

FROM ghcr.io/astral-sh/uv:0.12.21-python3.12-trixie-slim
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY tools/mlcheck/pyproject.toml tools/mlcheck/README.md tools/mlcheck/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-packages --extra train --extra cpu --no-install-project --no-install-workspace
COPY src ./src
COPY tools/mlcheck/src ./tools/mlcheck/src
COPY data/derived ./data/derived
COPY scripts/retrain.sh ./scripts/retrain.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-packages --extra train --extra cpu \
    && uv pip install --no-cache "awscli==1.46.1"
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app \
    && mkdir -p /work /app/reports && chown app:app /work /app/reports
# The run manifest records the commit; images have no .git, so CI bakes it in (--build-arg GIT_SHA).
ARG GIT_SHA
# poe delegates to `uv run`; keep uv offline and away from the read-only home directory.
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 CHARADE_GIT_SHA=$GIT_SHA \
    UV_CACHE_DIR=/tmp/uv-cache UV_NO_SYNC=1 UV_FROZEN=1
USER 10001
CMD ["./scripts/retrain.sh"]
