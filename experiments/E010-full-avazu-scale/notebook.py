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
# # E010 — The same model family at 9x the data
#
# **Hypotheses:** H1 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Result:** On one shared test set, single-seed DCN-v2 NE is 0.873 with 0.74M training rows and 0.857 with 6.7M; LightGBM is 0.875 and 0.855. These are descriptive results with changing training and validation volume; seeds and paired intervals are needed before comparative claims.
#
# **Question:** How much does training volume move NE, compared with the DCN-v2 vs LightGBM choice?
#
# **Method:** The full Kaggle Avazu file (same ten days) has no character layer, so the character and conversation groups are removed everywhere. From the file, every impression of 1 in 4 users is kept (40M rows need ~150 GB in this in-memory pipeline; sampling users keeps histories and counters exact). Two models per family are trained on that population's train/validation days: on 1 in 36 users (0.74M rows, the sample's size) and on all of it (6.7M rows). **Both are scored on the same 1.2M test rows.** One seed each. Separately, the 1M sample is trained without the character layer and scored on its own test set, which is comparable to the shipped model. Run after every decision was made; nothing shipped was trained, tuned or selected on the full file.
#
# Rerun on full data with `uv run python -m charade.analysis.fullscale data/raw/avazu-full/train` (needs the Kaggle file, ~6 GB, and ~25 GB of RAM); the notebook reads the saved result.

# %%
import json
from pathlib import Path

import polars as pl

report = json.loads(Path("../../reports/models/fullscale.json").read_text())
pl.DataFrame(
    [
        {
            "run": r["data"],
            "train rows": r["rows"]["train"],
            "test rows": r["rows"]["test"],
            "model": model,
            "NE": r[key]["ne"],
            "AUC": r[key]["auc"],
            "pred/obs": r[key]["calibration_ratio"],
        }
        for r in [report["sample_baseline"], *report["shared_test"]]
        for model, key in (("DCN-v2", "dcn_v2"), ("LightGBM", "lightgbm"))
    ]
)

# %% [markdown]
# ## Reading
# - **Volume:** the observed NE differences are 0.016 for DCN-v2 and 0.021 for LightGBM. Training and validation volume both change; no paired intervals or repeated seeds are available. The earlier fourfold comparison mixed these NE differences with E004 log loss and is withdrawn.
# - **Architecture:** DCN-v2 leads by 0.003 NE at 0.74M rows and trails by 0.002 at 6.7M (single seeds; seed variability was not measured). Consistent with E004: the architecture-alone advantage is not established.
# - **Character layer:** the no-character sample run is a single model, whereas the shipped result is an ensemble. Use the controlled E005 ablation for the character-feature effect, not this cross-configuration comparison.
# - **Correction:** an earlier version compared the sample's test set with the full file's (test CTR 17.1 % vs 15.6 %) and reported a 0.057 NE gain; that comparison did not isolate data volume.

# %% [markdown]
# ## Conclusion
# On one shared test set, single-seed DCN-v2 NE is 0.873 with 0.74M training rows and 0.857 with 6.7M; LightGBM is 0.875 and 0.855. These are descriptive results with changing training and validation volume; seeds and paired intervals are needed before comparative claims.
#
# **Decision:** Keep DCN-v2 for the shipped model (single ONNX graph; character interactions on the real task) without claiming it beats boosted trees; with production volume, re-run the comparison with seeds and CIs before choosing. More data is the first scaling lever.
