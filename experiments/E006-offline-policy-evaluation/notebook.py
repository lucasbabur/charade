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
# # E006 — Offline evaluation of ranking policies
#
# **Hypotheses:** H11, H6 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Question:** Would ranking by the model's pCTR serve more clicked ads than the policy that produced the logs, and what do safety gates and exploration cost?
#
# **Method:** Reconstruct candidate sets per publisher x hour on the test days, take each creative's share of the cell as its logging propensity, re-score every (impression, candidate) pair with the shipped bundle, and estimate each policy with IPS, SNIPS and doubly robust estimators plus paired hour-block bootstrap lifts.

# %%
from IPython.display import Markdown

from charade.analysis.experiment import setup

ctx = setup()
f"mode: {'smoke (fixture, tiny model)' if ctx.smoke else 'full data'}"

# %%
from charade.analysis import policy_eval
from charade.analysis.experiment import ensure_bundle

ensure_bundle(ctx)
policy_eval.run(ctx.settings, ctx.data_dir, report=ctx.reports / "policy" / "ope.md")
Markdown((ctx.reports / "policy" / "ope.md").read_text())

# %% [markdown]
# ## Reading
# - Uniform random lands near the logging CTR, as it should when logging is frequency-proportional.
# - Greedy pCTR's DR lift excludes zero; the shipped policy pays ~0.15 pp for brand safety, caps and
#   exploration, which pushes its interval across zero.
# - H6 shows up as the frequency-cap gate and as `log_user_campaign_imps` in the model.

# %% [markdown]
# ## Conclusion
# Greedy pCTR ranking beats the logging policy by +1.28 pp CTR (DR, [+0.07, +2.50]); the shipped policy with gates and 5 % exploration is +1.13 pp with a CI touching zero; ESS ~4 % of rows.
#
# **Decision:** Ship greedy + gates + 5 % Thompson exploration and log real propensities and candidate sets so the next evaluation does not need reconstruction.
