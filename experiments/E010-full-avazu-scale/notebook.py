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
# # E010 — The same model family at 40x the data
#
# **Hypotheses:** H1 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Result:** 9x the data (every impression of 1 in 4 users of the full Kaggle file) improves NE from 0.914 to 0.857 and AUC from 0.706 to 0.766; DCN-v2 and LightGBM stay within about 0.003 NE of each other at both scales (one seed, no CI), so data volume, not architecture, is the lever.
#
# **Question:** Does the DCN-v2 vs LightGBM ranking, and the level of NE, hold up when the 1M-row sample is replaced by the full Kaggle Avazu train file (~40M rows, the same ten days)?
#
# **Method:** The full file has no character layer and its 40M rows need ~150 GB in this in-memory pipeline, so the large run keeps every impression of 1 in 4 users (9.2M rows, 9x the sample; sampling users keeps each kept user's history and counters exact). The character layer is absent, so both runs drop the character and conversation groups and keep context, device, ad and user history. Same causal counters, same temporal split and test window (through 2014-10-30 05:00), same tuned hyperparameters, one seed each. This is not a leaderboard comparison: published Avazu numbers use random splits.
#
# Rerun on full data with `uv run python -m charade.analysis.fullscale data/raw/avazu-full/train` (needs the Kaggle file, ~6 GB, and ~25 GB of RAM; about 6 minutes on the GTX 1660 SUPER); the notebook only reads the saved result.

# %%
import json
from pathlib import Path

import polars as pl

report = json.loads(Path("../../reports/models/fullscale.json").read_text())
pl.DataFrame(
    [
        {
            "data": r["data"],
            "train rows": r["rows"]["train"],
            "test rows": r["rows"]["test"],
            "model": model,
            "NE": r[key]["ne"],
            "log loss": r[key]["logloss"],
            "AUC": r[key]["auc"],
            "pred/obs": r[key]["calibration_ratio"],
        }
        for r in report["results"]
        for model, key in (("DCN-v2", "dcn_v2"), ("LightGBM", "lightgbm"))
    ]
)

# %% [markdown]
# ## Reading
# - **Volume is worth more than any architecture choice here:** 9x the rows lowers NE by 0.057 (0.914 -> 0.857), five times the DCN-v2 vs LightGBM ensemble gap on the shipped data. User-history counters also get denser with more of each user's impressions.
# - **The model ranking does not hold, and that is consistent with E004:** DCN-v2 leads by 0.003 NE on 1M rows and trails by 0.002 on 9M, both single seeds within seed noise (single-seed spread on the shipped data is ~0.002). E004 already found the architecture-alone gap not established.
# - **Without the character layer, the 1M model scores NE 0.914 against the shipped 0.8855:** the synthetic character features carry ~0.03 NE on this data (they were generated with genre effects, so this is not evidence for real personas).
# - **Compare NE, not log loss, across the two rows:** the sample's test CTR (17.1 %) differs from the full file's (15.6 %), so raw log loss is not comparable; NE normalises by each test set's base rate.
# - Published Avazu numbers (log loss ~0.37-0.38) use random splits on all 40M rows; the 9M temporal result (0.371) lands in that range without same-hour leakage, but the setups differ too much to rank against them.

# %% [markdown]
# ## Conclusion
# 9x the data (every impression of 1 in 4 users of the full Kaggle file) improves NE from 0.914 to 0.857 and AUC from 0.706 to 0.766; DCN-v2 and LightGBM stay within about 0.003 NE of each other at both scales (one seed, no CI), so data volume, not architecture, is the lever.
#
# **Decision:** Keep DCN-v2 for the shipped model (ONNX export, character interactions on the real task) but do not claim it beats boosted trees; with production volume, re-run the DCN vs LightGBM comparison with seeds and CIs before choosing. More data is the first scaling lever.
