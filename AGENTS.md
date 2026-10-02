# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, …) working in this repo. Humans: start at [README](README.md) and [docs/index.md](docs/index.md).

## Project

**Charade** is a CTR prediction and candidate-ranking system for contextual ads inside AI companion chats (Simula take-home). Given an impression (character, conversation turn, publisher, device, hour) and N candidate ads, it predicts calibrated P(click) and returns a gated, explained ranking in under 50 ms p99. The stack is Python 3.12, uv workspace, polars, PyTorch (DCN-v2), LightGBM (challenger), ONNX Runtime (serving), FastAPI, Redis, Terraform (AWS), Docker.

Design decisions live in [docs/adr/](docs/adr/); gates in [docs/mlcheck.md](docs/mlcheck.md). If code and an ADR disagree, stop and ask; do not silently pick one.

## Commands

Only commands that work today are listed. When you add a command, add it here in the same commit.

| Purpose | Command |
|---|---|
| Install everything | `uv sync --all-packages --extra train --extra gpu` (`--extra cpu` without CUDA) |
| List tasks | `uv run poe` |
| Everything CI runs on Python | `uv run poe check` (lint, types, import layers, dead code, tests, mlcheck tests, ML static gates) |
| Terraform fmt + validate | `uv run poe tf` (CI also runs tflint and checkov; locally via their Docker images, see docs/07-serving-operations.md) |
| Regenerate OpenAPI contract | `uv run poe openapi` (commit `docs/api/openapi.json`) |
| Data contract on raw CSVs | `uv run poe mlcheck-data` |
| Train + evaluate + export (GPU if available) | `uv run poe train` (then `uv run mlcheck .`) |
| Ablations / tuning / EDA / text bake-off / refit study | `uv run poe ablate` / `tune` / `eda` / `text` / `refit-study` |
| Off-policy evaluation of ranking policies (after `train`) | `uv run poe ope` |
| Train/serve parity, sample rankings | `uv run poe parity`, `uv run poe samples` |
| Experiments: smoke-run all / rerun on full data | `uv run poe test-notebooks` / `uv run poe experiments` |
| Cold start, drift + staleness, adaptation replay | `uv run poe coldstart`, `drift`, `adapt` |
| Load test (needs `docker compose up -d --build`) | `uv run poe loadtest` |
| Run the API | `uv run poe serve` |
| One ML gate with evidence | `uv run mlcheck . --only MLM004 -v` |
| Lint workflows | `actionlint` |
| Build and run API + Redis | `docker compose up --build` |

Tasks live in `[tool.poe.tasks]`. CI calls the same tasks, so a green `poe check` locally means a green Python job.

The raw data (`impressions.csv`, `characters.csv`, 170 MB) sits in the repo root and is gitignored. Never commit it. Tests use `tests/fixtures/` samples.

## Layout

The code is organised by ML concern, not by clean-architecture layers ([ADR 0008](docs/adr/0008-ml-layout-and-mlcheck.md)).

```
src/charade/{data,features,text,models,evaluation,ranking,scoring,serving,analysis}/  config.py
tools/mlcheck/            ML release gates (own tests, own AGENTS.md)
artifacts/current/        run output in the mlcheck artifact contract (gitignored)
reports/                  generated figures/tables, committed; never hand-edited
docs/                     everything explained; index at docs/index.md
infra/terraform/, docker/, docker-compose.yml, .github/ (workflows; dependabot.yml is the only non-pyproject config because GitHub requires its location)
```

## Dependency direction

`analysis → serving → models → evaluation → {text | ranking | scoring} → features → data → config`. A package imports only packages to its right; same-layer packages never import each other, so cycles are impossible. Serving-path packages (`serving`, `scoring`, `ranking`, `features`) never reach `models`, `evaluation`, `analysis`, `text` or training frameworks. Enforced by import-linter (`[tool.importlinter]`, `uv run poe imports`) on every PR.

## Settings

**All settings live in `pyproject.toml`** — no YAML, `.ini`, `setup.cfg`, `.ruff.toml` or stray config files. Tool settings go under `[tool.<name>]`. Project settings go under `[tool.charade]`: paths, seeds, split dates, `model` (feature groups, DCN and LightGBM hyperparameters) and `policy` (brand-safety matrix, caps, exploration). `charade.config` loads them with pydantic-settings; `CHARADE_*` environment variables override them, and secrets come only from the environment.

## Invariants

Each one is enforced by an mlcheck gate. Breaking one fails CI.

1. **Time is the only split.** Train 10-21→10-27, val 10-28, test 10-29→10-30. No `train_test_split`, KFold or shuffling (MLL001–002 check the split artifacts).
2. **Nothing is fitted on the future.** Vocabularies, encoders, priors, PCA, scalers and models are fitted on train; the calibrator is fitted on val. Record every fitted object in the manifest with `fit_split`/`fit_end` (MLL003). A legitimate fit on eval data (the calibrator on val) is named in the manifest with its `fit_split`.
3. **Counters are strictly causal.** Use only hours before the impression hour, because order within an hour is unknown.
4. **One feature transform.** Training and serving both call `charade.features`. Never re-implement a feature in `serving/` (MLS002, MLV001).
5. **Serving stays light.** `serving/` must not reach torch, lightgbm, sklearn, optuna or mlflow, even transitively. It loads ONNX (MLS001).
6. **Explicit randomness.** Pass `np.random.Generator`/seeds; never use global `np.random.*` or `random.*` (code review; no gate). The primary model runs with ≥ 3 seeds (MLR004).
7. **No pickle.** Artifacts are ONNX, JSON, parquet or safetensors. `torch.load` needs `weights_only=True` (MLS005).
8. **Choose on validation, confirm on test.** Tune, ablate and select on val. If a choice needs the test days, it is a new experiment on a fresh window, and the docs say so. No gate can prove this; it is a discipline.
9. **Calibration matters as much as ranking.** Log loss and normalized entropy are the primary metrics; AUC is secondary. A model that improves AUC but breaks calibration (MLM005–007) does not ship.
10. **Claims need intervals.** A "model A beats B" statement needs a paired hour-block bootstrap CI (MLM004). Use one seed and one split only for debugging.
11. **Notebooks only in `experiments/`.** Each experiment is a folder with a frontmatter README (id, hypotheses, status, conclusion) and a notebook paired with a `.py` (edit the `.py`, then `jupytext --sync`). Logic lives in `src/charade/analysis`; CI executes every notebook on the fixture. Notebooks anywhere else fail MLS006.
12. **Gated ads are never served**, and every candidate logs its exact propensity; logged propensities must form the policy's distribution (MLP003–005, recovery test in `tests/ranking`).

## Data facts you will otherwise get wrong

Read [docs/01-data.md](docs/01-data.md) before touching features. The traps: `device_id='a99f214a'` is a placeholder (82 % of rows); C14 → C17 → C21 is creative → campaign → advertiser; genre is the `character_name` prefix; same-hour user counts leak; 2014-10-30 is a partial day.

## External APIs (paid)

Embedding providers (`charade.text.embed`: TF-IDF and Qwen3, both local) run **offline only**, cached by provider and a hash of the texts.

- Never call them from tests or CI; tests use TF-IDF.
- Never call them from `serving/`.
- Keys come from the environment ([.env.sample](.env.sample)). Never write keys to files, logs or artifacts.
- Ask before running a full re-embed: the cost is trivial, but the cache is the reproducibility record.

## Workflow

1. Read [docs/index.md](docs/index.md) → the doc for the area you touch → the code.
2. Make the smallest change that fully meets the task. Do not add backward-compatibility shims; change the callers.
3. Prefer established libraries. Before writing custom code, check what exists and say why it doesn't fit.
4. Write or update tests in the same change. Each test proves one behaviour, and its name states it.
5. Run lint, types, tests and the relevant `mlcheck` stages. Paste failures verbatim; never weaken a threshold or add an ignore just to go green.
6. Update the docs whose numbers or behaviour changed. Numbers in docs come from `reports/`, never typed by hand.
7. Commit atomically with conventional commits (`feat(features): …`, `fix(ranking): …`, `test(…)`, `docs(…)`, `chore`, `ci`, `build`). Every commit must build and pass its own tests. The git identity is the personal one (automatic under `~/Developer/personal`).

## Definition of done

- [ ] `uv run poe check` passes (and `uv run poe tf` if you touched `infra/`). Coverage on touched packages ≥ 85 % branch.
- [ ] `uv run mlcheck . --stage static` passes. If you touched training or evaluation, the full run gates pass with `--strict`, or every WARN is explained in the docs.
- [ ] If the API changed, `docs/api/openapi.json` is regenerated and committed.
- [ ] Every doc you changed has its `updated-at` bumped (CI checks it), links resolve and `uv run poe` commands exist (`tests/docs`).
- [ ] No secrets, raw data, notebooks or artifacts are staged.

## Writing style for docs and code comments

Maximum density. Lead with the conclusion, then the evidence (a number with its interval and source). No filler, no marketing. Comments explain *why*, never *what*. Match the surrounding code's naming and idiom.

Docs in `docs/` start with frontmatter (`title`, `created-at`, `updated-at`, ISO dates). Each fact has one home: state it once and link to it elsewhere. Delete superseded docs; git keeps history.
