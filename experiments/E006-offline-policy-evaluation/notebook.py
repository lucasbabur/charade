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
# - Both the ungated greedy and the shipped policy have DR lifts whose intervals exclude zero; the
#   shipped policy pays ~0.16 pp for brand safety, caps and exploration.
# - The intervals cover sampling noise under the reconstruction assumptions, not bias from them.
# - H6 shows up as the frequency-cap gate and as `log_user_campaign_imps` in the model.

# %% [markdown]
# ## Conclusion
# Under reconstructed candidate sets and frequency-share propensities, with DR using an independent LightGBM reward model, the shipped policy's estimated lift over logging is +1.23 pp CTR [+0.20, +2.20] (ungated greedy +1.41 pp; validation day +2.08 pp); ESS ~4 % of rows; a demonstration of the evaluation, not a measured production lift.
#
# **Decision:** Ship greedy + gates + 5 % exploration from a closed-form distribution, and log every candidate's exact propensity so the next evaluation does not need reconstruction.
