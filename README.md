# 🎭 Charade

Charade (**chara**cter + **ad**) ranks contextual ads inside AI companion chats. For each ad opportunity (character, conversation moment, publisher, device, hour) and a set of candidate ads, it predicts the click probability, filters out ads that don't fit (brand safety, frequency caps, budget), and returns a ranked list in under 50 ms.

Design and results live in [docs/](docs/index.md).

## 🚀 Run it

Requirements: [uv](https://docs.astral.sh/uv/) (installs Python 3.12 if missing). Optional: Docker for `docker compose up` (API + Redis), Terraform ≥ 1.13 for the infrastructure checks.

```bash
uv sync --all-packages                         # install the workspace
cp .env.sample .env                            # optional: overrides and API keys
cp /path/to/{impressions,characters}.csv .     # raw data, gitignored
uv run poe mlcheck-data                        # check the data contract
uv run poe serve                               # API on http://127.0.0.1:8000 (docs at /docs)
docker compose up --build                      # or: API + Redis in containers
uv run poe                                     # list every task
```

The API contract is committed at [docs/api/openapi.json](docs/api/openapi.json) and regenerated with `uv run poe openapi`. CI fails if it is stale.

> 🚧 Training, evaluation and ranking are being built. Their tasks will appear in `uv run poe` as they land.

## ⚙️ Settings

Every setting lives in `pyproject.toml`: tool settings under `[tool.<name>]`, project settings under `[tool.charade]`, tasks under `[tool.poe.tasks]`. `CHARADE_*` environment variables (or `.env`) override them. API keys come only from the environment; see [.env.sample](.env.sample).

## 🤝 Contribute

1. Read [AGENTS.md](AGENTS.md). It holds the rules, invariants and definition of done for humans and AI agents alike.
2. Find the doc for your area in [docs/index.md](docs/index.md).
3. Before committing, run what CI runs:
   ```bash
   uv run poe check   # lint, types, tests (charade + mlcheck), ML static gates
   uv run poe tf      # terraform fmt + validate
   ```
   If you changed the API, run `uv run poe openapi` and commit the contract.
4. Use atomic [conventional commits](https://www.conventionalcommits.org/) (`feat(ranking): …`). Every commit must pass its own tests.
