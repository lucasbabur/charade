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
import polars as pl

from charade.analysis import adaptation, drift
from charade.analysis.experiment import ensure_bundle

ensure_bundle(ctx)
tables = drift.run(ctx.settings, ctx.data_dir, ctx.reports / "drift")
tables["Daily mix and churn"]

# %%
stale = tables["Staleness: frozen models scored on later days (1 seed each)"]
fig, ax = plt.subplots(figsize=(6, 4))
for (day,), part in stale.group_by("scored_day", maintain_order=True):
    ax.plot(part["age_days"], part["ne"], "o-", label=str(day))
ax.set(xlabel="model age (days since last training day)", ylabel="NE", title="Staleness")
ax.legend(title="scored day", fontsize=7)
plt.show()

# %%
adapt = adaptation.run(ctx.settings, ctx.data_dir, ctx.reports / "drift" / "adaptation.md")
adapt.with_columns(pl.col(pl.Float64).round(4))

# %% [markdown]
# ## Reading
# - The CTR level moves for every genre together; character preferences are stable.
# - NE worsens with model age on every scored day, hence daily retraining.
# - Higher lambda trades concentration for estimated CTR; at lambda = 4 the CTR change is within noise.

# %% [markdown]
# ## Conclusion
# Ads rotate fast (13-37 % new creatives a day) and a frozen model loses ~0.003 NE per day, so retrain daily; online recalibration adds nothing; an exposure penalty (lambda = 4) cuts cohort concentration 30 % with no detectable CTR loss.
#
# **Decision:** Daily retraining with gated promotion; no online recalibration; exposure penalty recommended at lambda = 4 pending an online A/B test.
