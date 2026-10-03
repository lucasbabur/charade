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
# # E008 — Drift, staleness and exposure re-balancing
#
# **Hypotheses:** H9 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Question:** What drifts across the ten days, what does a stale model cost, and can an adaptation layer spread exposure without losing CTR?
#
# **Method:** Daily PSI against the training window, churn statistics, a staleness backtest of models frozen at different days, a causal per-genre hourly recalibration, and a replay of a cohort-exposure penalty with doubly robust CTR estimates.

# %%
from charade.analysis.experiment import setup

ctx = setup()
f"mode: {'smoke (fixture, tiny model)' if ctx.smoke else 'full data'}"

# %%
import matplotlib.pyplot as plt
from IPython.display import Markdown

from charade.analysis import adaptation, drift
from charade.analysis.experiment import ensure_bundle

ensure_bundle(ctx)
tables = drift.run(ctx.settings, ctx.data_dir, ctx.reports / "drift")
tables["Daily mix and churn"]

# %%
stale = tables[drift.STALE_TITLE]
stale

# %%
_, ax = plt.subplots(figsize=(6, 4))
for (day,), part in stale.group_by("scored_day", maintain_order=True):
    ax.plot(part["age_days"], part["ne"], "o-", label=str(day))
ax.set(xlabel="model age (days since last training day)", ylabel="NE", title="Staleness")
ax.legend(title="scored day", fontsize=7)
plt.show()

# %%
adaptation.run(ctx.settings, ctx.data_dir, ctx.reports / "drift" / "adaptation.md")
Markdown((ctx.reports / "drift" / "adaptation.md").read_text())

# %% [markdown]
# ## Reading
# - The CTR level moves for every genre together; character preferences are stable.
# - Three-day windows fix duration, not training volume or traffic mix. The table reports train/validation counts and NE, AUC and calibration. Slopes are descriptive point estimates without intervals; they establish neither a causal ageing rate nor stable ranking.
# - lambda is chosen on the validation day by the pre-registered rule; the test days only confirm it.
# - The replay uses logged exposure history, so it can show "no CTR loss" but never a fatigue benefit.

# %% [markdown]
# ## Conclusion
# The rolling-window backtest reports NE, AUC and calibration with training counts; varying volume and traffic mix prevent isolating model age, and no ranking-stability claim is established. Online recalibration slightly worsens validation NE. The selected exposure penalty (lambda = 2) cuts cohort HHI 24 % on test at +0.06 pp CTR [-0.60, +0.74], which does not confirm non-inferiority at the 0.2 pp margin.
#
# **Decision:** Daily retraining with gated promotion; no online recalibration; exposure penalty at lambda = 2 (chosen on validation) is a candidate for an online A/B test, not shipped: test does not confirm non-inferiority.
