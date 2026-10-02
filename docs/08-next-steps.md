---
title: "Next steps"
created-at: 2026-10-01
updated-at: 2026-10-02
---

# 08 — Next steps

Three investments, in order. Each answers a limit the evidence exposed, not a technology wish list.

1. **Log the real candidate sets and propensities, then evaluate on them.** The API already logs every candidate's exact selection probability. A few weeks of those logs replace the reconstructed candidate sets and frequency-share propensities behind today's OPE (ESS ≈ 4 % of rows), and make the exposure-penalty question (λ = 2 passed validation, not test) answerable by an online A/B test with a CTR guardrail.
2. **Content features for ads, which are the real cold-start problem.** 43 % of test impressions show a creative never seen in training; they rank as well as seen ones but are under-predicted by ~5 % ([05](05-cold-start.md#cold-ads)). Text or image embeddings of the creative, plus a campaign → advertiser backoff, target exactly that miscalibration. Characters, by contrast, showed no signal beyond genre × tier.
3. **Conversation context at ad time.** Turn number and session length are flat here; what the user is talking about is the missing input for a contextual ad. A privacy-filtered embedding of the last turns, as one more DCN input, then re-run the same ablation gate. With real free-text personas, re-run the text bake-off too ([03](03-models-evaluation.md#tested-and-rejected-character-text-e002)).

Not before these: new architectures, bandits or a feature store. The model is ~2 ms of a 25 ms p99 and the ranking already logs what counterfactual training would need; the binding constraint is data, not machinery.
