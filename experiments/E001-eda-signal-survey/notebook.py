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
# ---

# %% [markdown]
# # E001 — Signal survey on the training split
#
# **Hypotheses:** H2, H5, H6, H7, H8 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Question:** Which raw signals move CTR before any model, and which of them would leak?
#
# **Method:** Group-by CTR tables on the training split only (validation and test unseen), plus the genre/tier x campaign interaction residuals against their binomial noise.

# %%
from IPython.display import Markdown

from charade.analysis.experiment import setup

ctx = setup()
f"mode: {'smoke (fixture, tiny model)' if ctx.smoke else 'full data'}"

# %% [markdown]
# ## Run
# Every table is computed on the training split (10-21..10-27) except the daily CTR series.

# %%
from charade.analysis import eda

eda.run(ctx.data_dir, ctx.reports / "eda")
Markdown((ctx.reports / "eda" / "eda.md").read_text())

# %% [markdown]
# ## Reading
# - Genre spans mentor ~14.7 % to romance ~23.2 %; mature tier sits ~2.5 pp above sfw.
# - Conversation-turn buckets all sit at 18.4-18.7 %: no visible signal (H7).
# - Users seen in earlier hours click less (15-17 % vs 19.1 %); repeat campaign exposure drops CTR from
#   ~19.5 % to 13-15 % (H6).
# - Same-hour user counts separate CTR sharply, but the hour's total is only known after the hour (H5).
# - Genre x campaign residual sd 2.7 pp vs 1.1 pp binomial noise: interactions are real (feeds H1).

# %% [markdown]
# ## Conclusion
# Genre, safety tier, prior user activity and campaign repetition carry signal; conversation turn and character age do not; same-hour user counts are a leak.
#
# **Decision:** Keep genre, tier and causal user counters; treat turn and age as ablation candidates; exclude same-hour counts.
