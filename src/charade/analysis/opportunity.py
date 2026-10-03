"""Predictive cost of flattening candidate scores (`python -m charade.analysis.opportunity`).

An opportunity is one evaluated test impression with its reconstructed candidate set (the off-policy
evaluation's publisher x hour cell, every candidate scored in that impression's own context;
`charade.analysis.policy_eval`). Two scores are compared on the logged ad of each opportunity:
    model       the shipped scorer's pCTR for the reconstructed logged candidate
    flattened   the unweighted mean pCTR over the opportunity's candidates
Flattening changes calibration and candidate weighting as well as discrimination. Its loss cost is
not a decomposition of ranking skill or a policy-value estimate. With nonuniform logging, the
unweighted mean is not the context-conditional CTR. Rows outside the candidate set are excluded.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from charade.analysis.policy_eval import CandidateMatrices, load_candidates
from charade.config import Settings, get_settings
from charade.evaluation.metrics import logloss_rows, normalized_entropy, paired_bootstrap

OUT = Path("reports/models/opportunity.json")


def compare(c: CandidateMatrices) -> dict[str, object]:
    """Logged-candidate predictive loss under an unweighted, within-opportunity score ablation."""
    inside = c.logged >= 0
    rows = np.arange(len(c.y))[inside]
    y, blocks = c.y[inside], c.blocks[inside]
    p_model = c.pctr[rows, c.logged[inside]]
    p_flat = (c.pctr * c.mask).sum(axis=1)[inside] / c.mask.sum(axis=1)[inside]
    delta, low, high = paired_bootstrap(logloss_rows(y, p_flat) - logloss_rows(y, p_model), blocks)
    ne_model, ne_flat = normalized_entropy(y, p_model), normalized_entropy(y, p_flat)
    return {
        "opportunities": int(inside.sum()),
        "excluded_outside_candidate_set": int((~inside).sum()),
        "mean_candidates": float(c.mask[inside].sum(axis=1).mean()),
        "model": {"ne": ne_model, "auc": float(roc_auc_score(y, p_model))},
        "flattened": {"ne": ne_flat, "auc": float(roc_auc_score(y, p_flat))},
        "logloss_cost_of_flattening": {"delta": delta, "ci_low": low, "ci_high": high},
        "interpretation": "Predictive score ablation on reconstructed candidates; not ranking-skill share or CTR lift.",
    }


def run(settings: Settings | None = None, data_dir: Path | None = None, out: Path = OUT) -> dict[str, object]:
    """Write the model vs flattened comparison on the test split and return it."""
    settings = settings or get_settings()
    report = compare(load_candidates(settings, data_dir, "test"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
