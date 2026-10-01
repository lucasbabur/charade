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
# # E004 — CTR model comparison on the test days
#
# **Hypotheses:** H1, H9 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Question:** Does a cross network beat additive and tree baselines on the same inputs, and does validation-day calibration transfer to the test days?
#
# **Method:** Train logistic, tuned LightGBM and a 3-seed DCN-v2 ensemble on 10-21..10-27 with early stopping on 10-28; choose identity/Platt/isotonic calibration by hour-fold CV on 10-28; score the test days once; compare with a paired hour-block bootstrap. Writes the serving bundle and the mlcheck evidence.

# %%
from charade.analysis.experiment import setup

ctx = setup()
f"mode: {'smoke (fixture, tiny model)' if ctx.smoke else 'full data'}"

# %%
import numpy as np
import polars as pl

from charade.models import pipeline

metrics = pipeline.run(ctx.settings, reports=ctx.reports / "models")
pl.DataFrame(
    [{"model": m, **{k: round(v, 4) for k, v in s["test"].items() if k != "n"}} for m, s in metrics["models"].items()]
)

# %%
pl.DataFrame(
    [{"comparison": k, **{a: round(b, 5) for a, b in v.items()}} for k, v in metrics["comparisons_test"].items()]
)

# %%
{m: c["kind"] for m, c in metrics["calibration_choice"].items()}

# %% [markdown]
# ## Calibration on the test days (equal-mass bins, shipped model)

# %%
import matplotlib.pyplot as plt

preds = pl.read_parquet(ctx.settings.artifacts_dir / "predictions.parquet").filter(
    (pl.col("model") == "dcn_v2") & (pl.col("split") == "test")
)
order = np.argsort(preds["pred"].to_numpy())
bins = np.array_split(order, 15)
p, y = preds["pred"].to_numpy(), preds["label"].to_numpy()
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot([0, 0.6], [0, 0.6], color="grey", lw=1)
ax.plot([p[b].mean() for b in bins], [y[b].mean() for b in bins], "o-")
ax.set(xlabel="predicted CTR", ylabel="observed CTR", title="Reliability, test days")
plt.show()

# %%
pl.DataFrame(metrics["slices_test"]).filter(pl.col("slice").is_in(["character_support", "day", "user_seen"])).select(
    "slice", "value", "n", pl.col("ne").round(4), pl.col("calibration_ratio").round(3)
)

# %% [markdown]
# ## Reading
# - H1: the cross network beats both baselines with CIs that exclude zero.
# - H9 refuted as stated: early stopping on the low-CTR validation day anchors the level, no
#   calibration map beats identity by more than noise, and per-day ratios stay in [0.9, 1.1].
# - The single-seed comparison halves the DCN-LightGBM gap and its interval includes zero.

# %% [markdown]
# ## Conclusion
# DCN-v2 3-seed ensemble reaches test NE 0.8855 (pred/obs 1.002, no calibration map needed) and beats an equally tuned 3-seed LightGBM ensemble by -0.0040 [-0.0053, -0.0027] and logistic by -0.0117; single model vs single model DCN-LightGBM is -0.0020 [-0.0041, +0.0004], not established.
#
# **Decision:** Ship the DCN-v2 ensemble (ensemble result + single ONNX graph); keep LightGBM and logistic as yardsticks with equal tuning and seeds; do not claim an architecture advantage over trees.
