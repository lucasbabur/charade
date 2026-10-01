# simula-ctr

CTR prediction and candidate ranking for contextual ads in AI companion chats: calibrated P(click), gated and explained rankings, under 50 ms p99.

**Status:** design and ML release gates are done; model code is in progress. See [docs/index.md](docs/index.md).

```bash
uv sync --all-packages
cp /path/to/{impressions,characters}.csv .            # raw data, gitignored
uv run mlcheck . --stage data --stage static          # data contract + code gates
```

- Design: [docs/PLAN.md](docs/PLAN.md)
- ML gates: [docs/mlcheck.md](docs/mlcheck.md)
- Agent and contributor rules: [AGENTS.md](AGENTS.md)
