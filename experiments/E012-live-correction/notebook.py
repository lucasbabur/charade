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
# # E012 — Live (campaign, genre) correction: graduation and adaptation from outcomes
#
# **Hypotheses:** H12 ([docs/hypotheses.md](../../docs/hypotheses.md))
#
# **Question:** Can one online component answer the two open halves of the brief, a rule for when a cold entity has graduated and an adaptation layer that follows outcomes, without hurting CTR?
#
# **Method:** A Gamma-Poisson posterior per (campaign, genre) on a multiplier of the model's pCTR, learned from logged outcomes in strictly earlier hours and replayed over the reconstructed candidate sets of E006. Selection on the validation day by the E008 non-inferiority rule (95 % lower bound of the DR CTR change vs greedy clears -0.2 pp, best point estimate among deterministic configurations), one read of test.

# %%
from charade.analysis.experiment import setup

ctx = setup()
f"mode: {'smoke (fixture, tiny model)' if ctx.smoke else 'full data'}"

# %%
import matplotlib.pyplot as plt
import polars as pl
from IPython.display import Markdown

from charade.analysis import correction
from charade.analysis.experiment import ensure_bundle

ensure_bundle(ctx)
tables = correction.run(
    ctx.settings,
    ctx.data_dir,
    ctx.reports / "policy" / "correction.md",
    ctx.reports / "coldstart" / "graduation.csv",
    priors=(20.0,) if ctx.smoke else correction.PRIORS,
)
Markdown((ctx.reports / "policy" / "correction.md").read_text())

# %% [markdown]
# ## Deterministic configurations, validation then test
#
# Positive `dr_delta_vs_greedy_pp` is a CTR gain; negative `logloss_adj_minus_model` means the corrected pCTR predicts the logged ad's click better than the raw one.

# %%
columns = [
    "prior",
    "half_life_h",
    "dr_delta_vs_greedy_pp",
    "ci_low_pp",
    "ci_high_pp",
    "logloss_adj_minus_model",
    "ll_ci_low",
    "ll_ci_high",
    "pred_obs_model_cold",
    "pred_obs_adj_cold",
    "graduated_share_last_hour",
    "cohort_hhi",
]
tables["val"].filter(pl.col("z") == 0.0).select(columns)

# %%
tables["test"].filter(pl.col("z") == 0.0).select(columns)

# %%
curve = tables["graduation"]
if curve.height:
    _, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(curve.height), curve["graduated"], "o-")
    ax.set(xlabel="hour of the test window", ylabel="share of impressions on graduated pairs", ylim=(0, 1))
    ax.set_title("Graduation: logged evidence outweighs the prior (empty start)")
    plt.show()

# %% [markdown]
# ## Reading
# - The prior strength sets the trade-off. A weak prior (5 expected clicks) corrects cold campaigns almost fully but hurts the logged-ad log loss on validation; a strong one (320) is safe and nearly inert. Between 20 and 80 both readings improve.
# - The log-loss reading is causal and online: it compares two predictions for the same shown ad using only earlier hours. It does not rest on reconstructed candidate sets or frequency-share propensities.
# - Half-life cannot be distinguished on a 30-hour window; the decay exists for drift at week scale.
# - Each split starts from empty sums, so graduation shares understate production, where the state carries over.
# - The replay learns from the logging traffic, not from the corrected policy's own choices; an online test remains the arbiter, as for the exposure penalty.

# %% [markdown]
# ## Conclusion
# Selected on validation (prior 80 expected clicks, no decay), the corrected greedy policy is non-inferior to greedy pCTR on test at the 0.2 pp margin with a positive point estimate, lowers the log loss of the shown ad with a CI excluding zero, and narrows the cold-campaign under-prediction. Graduation is defined: a (campaign, genre) pair has graduated once its logged expected clicks reach the prior. Half-life is indistinguishable on 30 hours. Numbers: the test table above and [reports/policy/correction.md](../../reports/policy/correction.md).
#
# **Decision:** Ship the correction in the policy with the selected prior and no decay, replacing the frozen training-count evidence table; graduation = logged expected clicks ≥ prior. The CTR point estimate awaits the same online test as the exposure penalty.
