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
# # E003 — Hyperparameter search for DCN-v2 and LightGBM
#
# **Hypotheses:** none (supporting experiment) ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Question:** Which configurations to carry into the model comparison, chosen without touching the validation day?
#
# **Method:** Optuna TPE, 60 DCN-v2 and 40 LightGBM trials, trained on 10-21..10-26 and selected on 10-27 NE. The validation day (10-28) stays clean for early stopping, calibration and comparisons.

# %%
from charade.analysis.experiment import setup

ctx = setup()
f"mode: {'smoke (fixture, tiny model)' if ctx.smoke else 'full data'}"

# %% [markdown]
# The full search takes ~16 minutes on the GTX 1660 SUPER and GPU training is not bit-reproducible, so in
# full mode this notebook shows the committed search results; set `CHARADE_RERUN_TUNING=1` to rerun it.

# %%
import os

import polars as pl

from charade.models import tune

out = ctx.reports / "tuning"
if ctx.smoke:
    tune.run(trials_dcn=2, trials_gbdt=2, data_dir=ctx.data_dir, out=out)
elif os.environ.get("CHARADE_RERUN_TUNING") == "1":
    tune.run(out=out)
dcn, gbdt = pl.read_csv(out / "dcn.csv"), pl.read_csv(out / "gbdt.csv")
dcn.head(5)

# %%
gbdt.head(5)

# %%
shipped = ctx.settings.model
{"dcn": shipped.dcn.model_dump(), "gbdt": shipped.gbdt.model_dump()}

# %% [markdown]
# ## Reading
# - The top DCN trials agree on 24-dim embeddings, batch 2048 and lr ~2.5-2.9e-3; architecture choices
#   beyond that move inner NE by < 0.0003.
# - The best LightGBM trial is 0.004 NE behind the best DCN trial on the same inner split.

# %% [markdown]
# ## Conclusion
# Best inner-split NE: DCN-v2 0.8746 (24-dim embeddings, 2 cross layers, 256-128 MLP, dropout 0.3, lr 2.9e-3, batch 2048) vs LightGBM 0.8787; both configs ship in [tool.charade.model].
#
# **Decision:** Copy the best trials into `[tool.charade.model]` (done by hand, citing the CSVs).
