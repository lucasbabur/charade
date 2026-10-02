# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # E002 — Character description embeddings
#
# **Hypotheses:** H3 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Question:** Does a character's description carry CTR signal beyond its genre and safety tier?
#
# **Method:** Embed every description (TF-IDF->SVD, Qwen3-Embedding-0.6B on the GPU), check 5-NN recovery of genre and tier, and probe the character CTR residual over genre x tier with ridge regression trained on training characters and scored on validation characters.

# %%
from charade.analysis.experiment import setup

ctx = setup()
f"mode: {'smoke (fixture, tiny model)' if ctx.smoke else 'full data'}"

# %%
from charade.text import bakeoff
from charade.text.embed import DERIVED_DIR, Provider

providers = [Provider.TFIDF] if ctx.smoke else list(Provider)
derived = ctx.reports / "derived" if ctx.smoke else DERIVED_DIR
table = bakeoff.run(
    providers, ctx.data_dir, derived, ctx.reports / "text_bakeoff.csv", min_impressions=5 if ctx.smoke else 100
)
table

# %% [markdown]
# ## Reading
# - Genre 5-NN accuracy is 1.00 for every embedder: the descriptions are templates keyed by genre.
# - Tier accuracy is at or below the majority rate: the text says nothing about safety tier.
# - The residual correlation is zero within its CI for two very different embedders, so the absence
#   of signal is a property of the data, not of the embedder.

# %% [markdown]
# ## Conclusion
# Embeddings recover genre perfectly but predict no CTR beyond genre x tier (Spearman rho ~0 +/- 0.1); text is not a model input.
#
# **Decision:** Do not use text features (the extrinsic ablation in E005 agrees); keep the pipeline for free-text personas.
