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
# # E009 — Per-feature importance (LightGBM SHAP and permutation)
#
# **Hypotheses:** H2, H7, H8 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Result:** the publisher (app, site) and genre carry most of the signal; character ID is *used* but replaceable by genre (E005), conversation features and character age are within noise.
#
# **Question:** Within the groups the ablations kept, which individual features does a model rely on, and do the dropped ones look different feature by feature?
#
# **Method:** One LightGBM (tuned config) on train with every non-text group, including character ID and conversation. On the validation day: mean |SHAP| (TreeSHAP, logit units, 50k rows) and the log-loss increase when each feature is shuffled, with a 95 % hour-block bootstrap CI. Selection stays with the group ablations (E005): a used feature can still be replaceable.

# %%
from charade.analysis.experiment import setup

ctx = setup()
f"mode: {'smoke (fixture, tiny model)' if ctx.smoke else 'full data'}"

# %%
import matplotlib.pyplot as plt
import polars as pl

from charade.analysis import importance

table = importance.run(ctx.settings, ctx.data_dir, ctx.reports / "models")
table.with_columns(pl.col(pl.Float64).round(5))

# %%
table.group_by("group").agg(pl.col("permutation_delta").sum()).sort("permutation_delta", descending=True)

# %%
top = table.head(20).reverse()
_, ax = plt.subplots(figsize=(7, 6))
ax.errorbar(
    top["permutation_delta"],
    range(top.height),
    xerr=[top["permutation_delta"] - top["ci_low"], top["ci_high"] - top["permutation_delta"]],
    fmt="o",
)
ax.axvline(0, color="grey", lw=1)
ax.set_yticks(range(top.height), top["feature"].to_list())
ax.set_xlabel("validation log-loss increase when shuffled")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Reading
# - Context dominates: `app_id`, `site_id` and `site_domain` are the three most needed features. Where the ad runs matters more than anything about the ad or the user.
# - H2: `genre` is the fourth most needed feature and the top character feature; `safety_tier` adds a little. `creator_type` is unused.
# - Character ID is used (shuffling it costs about 0.001) yet removing the whole ID group costs nothing (E005): its signal is duplicated by genre and tier, the textbook case of "used but replaceable".
# - H7: every conversation feature is within noise.
# - H8: `log_character_age_days` is within noise on its own, which closes the gap left by ablating age only inside `character_meta`.
# - User-history features are individually small; together they are the user-history group E005 keeps.

# %% [markdown]
# ## Conclusion
# The publisher (app, site) and genre carry most of the signal; character ID is used but replaceable by genre (E005), conversation features and character age are within noise.
#
# **Decision:** No change to the shipped groups. Per-feature results support the group decisions and settle H8 for age.
