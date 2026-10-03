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
# # E011 — How much of the model's skill ranks ads?
#
# **Hypotheses:** H1, H11 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Result:** Only 3.4 % of the model's skill ranks ads: replacing each candidate's pCTR with its opportunity's mean costs 0.0018 log loss [0.0010, 0.0030] (NE 0.8842 -> 0.8882); the rest predicts whether a moment is clickable at all, which the ranker cannot use.
#
# **Question:** NE rewards predicting *whether* a moment gets a click. The ranker only uses *differences between the candidates* of one moment. How much of the model's skill is the second kind?
#
# **Method:** An opportunity is a test impression with its reconstructed candidate set (the off-policy evaluation's publisher x hour cell; every candidate scored in that impression's context). On the logged ad, compare the shipped pCTR with the mean pCTR of the opportunity's candidates, which keeps the context signal and removes every difference between ads. Paired hour-block bootstrap of the log-loss gap. Suggested by the external reviewer's flattening diagnostic (NE 0.8896 -> 0.8940 on their rows).

# %%
from charade.analysis.experiment import setup

ctx = setup()
f"mode: {'smoke (fixture, tiny model)' if ctx.smoke else 'full data'}"

# %%
import json

from charade.analysis import opportunity
from charade.analysis.experiment import ensure_bundle

ensure_bundle(ctx)
report = opportunity.run(ctx.settings, ctx.data_dir, ctx.reports / "models" / "opportunity.json")
print(json.dumps(report, indent=2))

# %% [markdown]
# ## Reading
# - The headline NE (0.8855) is mostly *context* skill: which publisher, device and user are likely to click. E009 agrees: `app_id`, `site_id` and `site_domain` are the three most needed features. Within one request that signal is identical for every candidate and cancels out.
# - The part that changes which ad is served is small but real (CI excludes zero), consistent with the modest offline policy lift (+1.46 pp, [04](../../docs/04-ranking-policy.md)).
# - Avazu ads are anonymous ids, so there is little to tell them apart. Simula's value is exactly this within-opportunity part (which ad fits this conversation), which needs ad content and conversation features ([08](../../docs/08-next-steps.md)).
# - Evaluation consequence: report a within-opportunity metric next to NE, because that is what the ranker consumes.

# %% [markdown]
# ## Conclusion
# Only 3.4 % of the model's skill ranks ads: replacing each candidate's pCTR with its opportunity's mean costs 0.0018 log loss [0.0010, 0.0030] (NE 0.8842 -> 0.8882); the rest predicts whether a moment is clickable at all, which the ranker cannot use.
#
# **Decision:** No model change. The flattening gap is reported next to NE in the summary, and next steps prioritise ad-content and conversation features, which act within opportunities.
