# 03 — Character text

**Bottom line:** descriptions are templates: embeddings recover genre perfectly but predict no CTR beyond genre × tier (ρ ≈ 0 ± 0.1), and adding them hurts validation log loss ([04](04-models-evaluation.md)). The pipeline stays for real free-text personas.

## Bake-off ([reports/text_bakeoff.csv](../reports/text_bakeoff.csv), `uv run poe text`)

| Provider | Dims | Genre 5-NN acc | Tier 5-NN acc | Residual ρ (95 % CI), 363 val characters |
|---|---|---|---|---|
| TF-IDF 1–2-grams → SVD | 128 | 1.00 | 0.50 | 0.017 (−0.086, 0.120) |
| Qwen3-Embedding-0.6B (local, GTX 1660 SUPER) | 1024 | 1.00 | 0.50 | 0.001 (−0.102, 0.104) |

The residual is each character's training CTR (≥ 100 impressions) minus its genre × tier rate. A ridge model on the embedding is trained on training characters and scored by Spearman ρ on validation characters.

## Providers not run

| Provider | Status |
|---|---|
| OpenAI `text-embedding-3-large` | Implemented (`Provider.OPENAI`); the available key returned `insufficient_quota` on 2026-10-01. Run `OPENAI_API_KEY=… uv run poe text` to add it |
| Gemini `gemini-embedding-001`, Voyage `voyage-4-large` | No keys available. They would plug in as one function each in `charade.text.embed.EMBEDDERS` |

## Pipeline rules

- Raw embeddings are cached in `artifacts/cache/text/` (gitignored), keyed by provider and a hash of the texts.
- Only 16-dim PCA reductions are committed, in `data/derived/text_<provider>.parquet` (~0.3 MB each), so results reproduce without keys or a GPU.
- PCA is fitted only on characters created by the end of training. New characters are projected with the stored transform when they are published.
- Embedding never runs in serving, tests or CI. Serving reads the character table with precomputed vectors.
