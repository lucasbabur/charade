---
id: E002
title: "Character description embeddings"
hypotheses: [H3]
status: concluded
conclusion: "Embeddings recover genre perfectly but predict no CTR beyond genre x tier (Spearman rho ~0 +/- 0.1); text is not a model input."
created-at: 2026-10-01
updated-at: 2026-10-01
---

# E002 — Character description embeddings

**Question:** Does a character's description carry CTR signal beyond its genre and safety tier?

**Method:** Embed every description (TF-IDF->SVD, Qwen3-Embedding-0.6B on the GPU; OpenAI when a key with quota exists), check 5-NN recovery of genre and tier, and probe the character CTR residual over genre x tier with ridge regression trained on training characters and scored on validation characters.

**Decision:** Do not use text features (the extrinsic ablation in E005 agrees); keep the pipeline for free-text personas.

The notebook ([notebook.ipynb](notebook.ipynb), paired with [notebook.py](notebook.py)) holds the code, outputs and reading. Rerun with `uv run poe experiments experiments/E002-text-embedding-bakeoff/notebook.py`.
