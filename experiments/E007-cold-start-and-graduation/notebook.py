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
# # E007 — Cold start and graduation
#
# **Hypotheses:** H10, H8 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Question:** What signal exists before an entity's first click, and when should a character get its own parameters?
#
# **Method:** Permutation importance on cold slices of the test days; method-of-moments Beta priors per genre x tier with binomial noise removed; a causal per-character logit correction from strictly earlier impressions, evaluated by earlier-impression bucket.

# %%
from IPython.display import Markdown

from charade.analysis.experiment import setup

ctx = setup()
f"mode: {'smoke (fixture, tiny model)' if ctx.smoke else 'full data'}"

# %%
from charade.analysis import coldstart
from charade.analysis.experiment import ensure_bundle

ensure_bundle(ctx)
coldstart.run(ctx.settings, ctx.data_dir, ctx.reports / "coldstart")
Markdown((ctx.reports / "coldstart" / "coldstart.md").read_text())

# %% [markdown]
# ## Reading
# - Publisher surface and genre carry the pre-click signal for both cold characters and new users.
# - The pooled prior strength is enormous (true sd ~0.0001): a character's own clicks never outweigh it.
# - A weak prior (tau = 100) makes predictions slightly worse in the high-evidence bucket.

# %% [markdown]
# ## Conclusion
# This synthetic dataset shows no reliable gain from character-specific parameters (the pooled spread beyond genre x tier hits the estimator's floor); unseen characters score NE 0.908 vs 0.885 for warm ones (n = 1,478, no CI), and unseen creatives (43 % of test) are under-predicted by about 5 %.
#
# **Decision:** No character ID in the model; graduation is re-measured on fresh data instead of fixed as a constant.
