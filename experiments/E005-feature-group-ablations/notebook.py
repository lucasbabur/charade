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
# # E005 — Feature-group ablations
#
# **Hypotheses:** H2, H3, H4, H7 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Question:** Which feature groups earn their place in the model?
#
# **Method:** 3-seed DCN-v2 ensembles on the validation day, each identical to the all-features reference except for one change; paired hour-block bootstrap of the log-loss difference (raw logits, because the calibrator is fitted on validation).

# %%
from charade.analysis.experiment import setup

ctx = setup()
f"mode: {'smoke (fixture, tiny model)' if ctx.smoke else 'full data'}"

# %%
import matplotlib.pyplot as plt
import polars as pl

from charade.models import ablate

table = ablate.run(out=ctx.reports / "models", data_dir=ctx.data_dir)
table.with_columns(pl.col(pl.Float64).round(5))

# %%
rows = table.filter(pl.col("variant") != "all features")
_, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(
    rows["delta_logloss"],
    range(rows.height),
    xerr=[rows["delta_logloss"] - rows["ci_low"], rows["ci_high"] - rows["delta_logloss"]],
    fmt="o",
)
ax.axvline(0, color="grey", lw=1)
ax.set_yticks(range(rows.height), rows["variant"].to_list())
ax.set_xlabel("validation log loss vs all features (positive = worse)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Reading
# - H2: removing metadata costs far more than removing the character ID, which costs nothing.
# - H3: text features make the model worse, consistent with E002.
# - H4 partly refuted: user history helps, but less than device and character metadata.
# - H7: conversation features are within noise.
# - Deltas of +/-0.0003 (e.g. ID dropout, the shipped variant) are within seed noise, which this
#   bootstrap over hours does not include.

# %% [markdown]
# ## Conclusion
# Character metadata is worth +0.0035 log loss and user history +0.0012; character ID and conversation features are within noise and were dropped; text features hurt (+0.0005 to +0.0007).
#
# **Decision:** Ship without character ID and conversation groups; reject text; keep metadata, device and user history.
