"""How much of the model's skill ranks ads? (`python -m charade.analysis.opportunity`).

An opportunity is one evaluated test impression with its reconstructed candidate set (the off-policy
evaluation's publisher x hour cell, every candidate scored in that impression's own context;
`charade.analysis.policy_eval`). Two scores are compared on the logged ad of each opportunity:
    model       the shipped pCTR of the ad that was shown
    flattened   the mean pCTR over the opportunity's candidates, i.e. the same context signal with
                every ad scored alike
The ranker only sees differences between candidates of one opportunity, so the gap between the two
is the part of the model's skill that can change which ad is served; the rest predicts whether the
moment is clickable at all. Rows whose logged ad is outside the candidate set are excluded.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from charade.analysis.policy_eval import load_candidates
from charade.config import Settings, get_settings
from charade.evaluation.metrics import logloss_rows, normalized_entropy, paired_bootstrap

OUT = Path("reports/models/opportunity.json")


def run(settings: Settings | None = None, data_dir: Path | None = None, out: Path = OUT) -> dict[str, object]:
    """Write the model vs flattened comparison on the test split and return it."""
    settings = settings or get_settings()
    c = load_candidates(settings, data_dir, "test")
    inside = c.logged >= 0
    rows = np.arange(len(c.y))[inside]
    y, blocks = c.y[inside], c.blocks[inside]
    p_model = c.pctr[rows, c.logged[inside]]
    p_flat = (c.pctr * c.mask).sum(axis=1)[inside] / c.mask.sum(axis=1)[inside]
    delta, low, high = paired_bootstrap(logloss_rows(y, p_flat) - logloss_rows(y, p_model), blocks)
    ne_model, ne_flat = normalized_entropy(y, p_model), normalized_entropy(y, p_flat)
    report: dict[str, object] = {
        "opportunities": int(inside.sum()),
        "mean_candidates": float(c.mask[inside].sum(axis=1).mean()),
        "model": {"ne": ne_model, "auc": float(roc_auc_score(y, p_model))},
        "flattened": {"ne": ne_flat, "auc": float(roc_auc_score(y, p_flat))},
        "logloss_cost_of_flattening": {"delta": delta, "ci_low": low, "ci_high": high},
        "share_of_skill_that_ranks_ads": (ne_flat - ne_model) / (1 - ne_model),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
