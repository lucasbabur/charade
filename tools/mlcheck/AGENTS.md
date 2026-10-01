# AGENTS.md — tools/mlcheck

Scoped rules for the ML gate runner. The root [AGENTS.md](../../AGENTS.md) still applies. Reference: [docs/mlcheck.md](../../docs/mlcheck.md).

## Commands (run from this directory)

`uv run pytest` · `uv run ruff check . && uv run ruff format --check .` · `uv run pyright` · `uv run mlcheck --list`

## Adding or changing a check

1. Register it with `@check(code, name, stage, rationale, severity)` in the module for its stage:

   | Stage | Module |
   |---|---|
   | static | `checks/static.py` |
   | data | `checks/data.py` |
   | repro, leakage | `checks/provenance.py` |
   | model | `checks/model.py` |
   | serving, policy, drift | `checks/runtime.py` |

   The rationale is one sentence on the failure it prevents.
2. Codes are stable: prefix by stage (MLS, MLD, MLR, MLL, MLM, MLV, MLP, MLX), never reuse or renumber, and remove only with an entry in the docs.
3. Put numeric limits in `config.Thresholds` with a defensible default. Never hardcode them in the check.
4. Add **one breakage** to `BREAKAGES` in `tests/test_checks.py`: the minimal mutation of the golden project that the check must catch. `test_every_check_has_a_breakage` enforces this.
5. Keep the golden project (`tests/conftest.py`) passing every check. If a new check needs a new artifact, add it to `mlcheck.contract`, the golden builder and the artifact table in docs.
6. Add the row to the catalogue table in `docs/mlcheck.md`; `test_docs_catalogue_lists_every_check` enforces this.

## Rules

- **Fail closed.** A missing artifact, missing source or crash is FAIL. SKIP only when the config section is absent.
- **Recompute, don't trust.** Model metrics come from `predictions.parquet` via `stats.py`, never from numbers a project reports about itself.
- Checks are pure reads: no network, no writes, no imports of the checked project (static analysis via `ast`/grimp only).
- WARNING severity is only for conditions that need a human judgement (drift, volume, dominant values). Correctness gates are ERROR.
- Dependencies stay minimal (grimp, numpy, polars, pydantic, typer). Justify any addition.
