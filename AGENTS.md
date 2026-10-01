# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, …) working in this repo. Humans: start at [README](README.md) and [docs/index.md](docs/index.md).

## Project

**Charade** is a CTR prediction and candidate-ranking system for contextual ads inside AI companion chats (Simula take-home). Given an impression (character, conversation turn, publisher, device, hour) and N candidate ads, it predicts calibrated P(click) and returns a gated, explained ranking in under 50 ms p99. The stack is Python 3.12, uv workspace, polars, PyTorch (DCN-v2), LightGBM (challenger), ONNX Runtime (serving), FastAPI, Redis, Terraform (AWS), Docker.

The source of truth for design is [docs/PLAN.md](docs/PLAN.md); for gates it is [docs/mlcheck.md](docs/mlcheck.md). If code and PLAN disagree, stop and ask; do not silently pick one.

## Commands

Only commands that work today are listed. When you add a command, add it here in the same commit.

| Purpose | Command |
|---|---|
| Install everything | `uv sync --all-packages` |
| ML gates, whole project | `uv run mlcheck .` |
| ML gates before a run exists | `uv run mlcheck . --stage data --stage static` |
| One gate with evidence | `uv run mlcheck . --only MLM004 -v` |
| List gates | `uv run mlcheck --list` |
| mlcheck tests | `cd tools/mlcheck && uv run pytest` |
| Lint, format, types (mlcheck) | `cd tools/mlcheck && uv run ruff check . && uv run ruff format --check . && uv run pyright` |

The raw data (`impressions.csv`, `characters.csv`, 170 MB) sits in the repo root and is gitignored. Never commit it. Tests use `tests/fixtures/` samples.

## Layout

The code is organised by ML concern, not by clean-architecture layers ([ADR in PLAN §2](docs/PLAN.md#2-repository-layout)).

```
src/charade/{data,features,text,models,evaluation,ranking,coldstart,drift,serving}/  cli.py, config.py
tools/mlcheck/            ML release gates (own tests, own AGENTS.md)
artifacts/current/        run output in the mlcheck artifact contract (gitignored)
reports/                  generated figures/tables, committed; never hand-edited
docs/                     everything explained; index at docs/index.md
infra/terraform/, docker/, .github/workflows/
```

## Settings

**All settings live in `pyproject.toml`** — no YAML, `.ini`, `setup.cfg`, `.ruff.toml` or stray config files. Tool settings go under `[tool.<name>]`. Project settings go under `[tool.charade]`: paths, seeds, split dates, `experiments.<name>` and `policy` (safety matrix, fatigue, pacing, exploration). `charade.config` loads them with pydantic-settings (`PyprojectTomlConfigSettingsSource`); environment variables override them, and secrets come only from the environment. An experiment is a new `[tool.charade.experiments.<name>]` table, not a code edit.

## Invariants

Each one is enforced by an mlcheck gate. Breaking one fails CI.

1. **Time is the only split.** Train 10-21→10-27, val 10-28, test 10-29→10-30. No `train_test_split`, KFold or shuffling (MLS003, MLL001–002).
2. **Nothing is fitted on the future.** Vocabularies, encoders, priors, PCA, scalers and models are fitted on train; the calibrator is fitted on val. Record every fitted object in the manifest with `fit_split`/`fit_end` (MLL003). A legitimate fit on eval data needs `# mlcheck: ignore[MLS007]` and a reason.
3. **Counters are strictly causal.** Use only hours before the impression hour, because order within an hour is unknown.
4. **One feature transform.** Training and serving both call `charade.features`. Never re-implement a feature in `serving/` (MLS002, MLV001).
5. **Serving stays light.** `serving/` must not reach torch, lightgbm, sklearn, optuna or mlflow, even transitively. It loads ONNX (MLS001).
6. **Explicit randomness.** Pass `np.random.Generator`/seeds; never use global `np.random.*` or `random.*` (MLS004). The primary model runs with ≥ 3 seeds (MLR004).
7. **No pickle.** Artifacts are ONNX, JSON, parquet or safetensors. `torch.load` needs `weights_only=True` (MLS005).
8. **The test set is touched once per model version** and logged in `evaluation_ledger.jsonl` (MLL006). Tune on val. If you want to look at test, you are doing it wrong.
9. **Calibration matters as much as ranking.** Log loss and normalized entropy are the primary metrics; AUC is secondary. A model that improves AUC but breaks calibration (MLM005–007) does not ship.
10. **Claims need intervals.** A "model A beats B" statement needs a paired hour-block bootstrap CI (MLM004). Use one seed and one split only for debugging.
11. **No notebooks.** Analysis is a CLI command that writes to `reports/` (MLS006).
12. **Gated ads are never served**, and every served ad logs its propensity (MLP003–004).

## Data facts you will otherwise get wrong

- `device_id='a99f214a'` is a placeholder on 82 % of rows. The user proxy is `device_id` when real, otherwise `hash(device_ip, device_model)`.
- The C-features are an ad hierarchy: C14 = creative → C17 = campaign → C21 = advertiser; C15×C16 = size. C1 and C20 vary within a creative, so they are context. A candidate = `banner_pos` + C14 (+ derived fields).
- The genre is the `character_name` prefix (`romance_…`). Romance and horror run ~+4 pp CTR, mentor −4 pp; mature tier +2–3 pp.
- Descriptions are templated, and words add ≤ 0.44 pp within genre. Do not expect text to rescue a model.
- `conversation_turn`/`session_msg_count` show flat CTR. `session_msg_count` and `num_interactions` may leak future information; keep them behind ablation flags.
- 2014-10-30 is a partial day (23k rows). Weight it accordingly; don't read its daily metrics alone.
- Daily CTR swings 16.5–19.6 % and the test days are low. Check per-day calibration.

## External APIs (paid)

Embeddings (Gemini, Voyage, OpenAI) and Claude Haiku 4.5 attribute extraction run **offline only**, through `charade.text`, cached by `sha256(text)+model+prompt_version`.

- Never call them from tests or CI. Use recorded responses.
- Never call them from `serving/`.
- Keys come from the environment (`GEMINI_API_KEY`, `VOYAGE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). Never write keys to files, logs or artifacts.
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

- [ ] Lint, format, pyright strict and tests pass. Coverage on touched packages ≥ 85 % branch.
- [ ] `uv run mlcheck . --stage static` passes. If you touched training or evaluation, the full run gates pass with `--strict`, or every WARN is explained in the docs.
- [ ] The docs and `docs/index.md` status column are updated.
- [ ] No secrets, raw data, notebooks or artifacts are staged.

## Writing style for docs and code comments

Maximum density. Lead with the conclusion, then the evidence (a number with its interval and source). No filler, no marketing. Comments explain *why*, never *what*. Match the surrounding code's naming and idiom.
