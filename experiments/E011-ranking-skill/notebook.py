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
# # E011 — Predictive cost of flattening candidate scores
#
# **Hypotheses:** H1, H11 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Result:** Replacing candidate scores with their unweighted opportunity mean costs 0.0018 logged-impression log loss [0.0010, 0.0030] (NE 0.8842 -> 0.8882). This is a predictive score ablation, not a percentage of ranking skill or an estimate of CTR lift.
#
# **Question:** How much does logged-impression predictive loss change when every candidate in an opportunity receives the same score?
#
# **Method:** On 102,874 test opportunities, compare the shipped scorer on the reconstructed logged candidate with the unweighted candidate mean. Logged ads outside the top-K are excluded; ad attributes are reconstructed by their mode. Paired hour-block bootstrap of the logged-impression log-loss gap.

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
# - Flattening removes score differences but also changes calibration and candidate weighting. The loss gap is not a decomposition of ranking skill.
# - With nonuniform logging, the unweighted mean is not the context-conditional CTR. Duplicating a lower-scored candidate changes the gap while leaving the winner unchanged.
# - Aggregate NE and AUC remain good with identical scores within each request. This shows why those metrics alone cannot establish good ad selection; it does not quantify ranking value.
# - The former skill-share interpretation is withdrawn. Policy value needs real candidate sets and propensities, or an online test ([04](../../docs/04-ranking-policy.md)).

# %% [markdown]
# ## Conclusion
# Replacing candidate scores with their unweighted opportunity mean costs 0.0018 logged-impression log loss [0.0010, 0.0030] (NE 0.8842 -> 0.8882). This is a predictive score ablation, not a percentage of ranking skill or an estimate of CTR lift.
#
# **Decision:** No model change. Report the score-ablation gap with its scope; evaluate ranking decisions with real candidate sets, propensities and outcomes.
