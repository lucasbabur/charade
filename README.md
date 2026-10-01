# 🎭 Charade

Charade (**chara**cter + **ad**) ranks contextual ads inside AI companion chats. For each ad opportunity (character, conversation moment, publisher, device, hour) and a set of candidate ads, it predicts the click probability, filters out ads that don't fit (brand safety, frequency caps, budget), and returns a ranked list in under 50 ms.

Design and results live in [docs/](docs/index.md).

## 🚀 Run it

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it if missing).

```bash
uv sync --all-packages                         # install the workspace
cp /path/to/{impressions,characters}.csv .     # raw data, gitignored
uv run mlcheck . --stage data --stage static   # check the data and the code
```

> 🚧 Training, evaluation and the API are being built. Their commands will appear here as they land.

## ⚙️ Settings

Every setting lives in `pyproject.toml`: tool settings under `[tool.<name>]`, project settings under `[tool.charade]`. Environment variables override them. API keys come only from the environment.

## 🤝 Contribute

1. Read [AGENTS.md](AGENTS.md). It holds the rules, invariants and definition of done for humans and AI agents alike.
2. Find the doc for your area in [docs/index.md](docs/index.md).
3. Before committing, run lint, types, tests and the ML gates:
   ```bash
   cd tools/mlcheck && uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
   uv run mlcheck . --stage static
   ```
4. Use atomic [conventional commits](https://www.conventionalcommits.org/) (`feat(ranking): …`). Every commit must pass its own tests.
