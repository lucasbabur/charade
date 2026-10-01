---
title: "Next steps"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# 10 — Next steps

Ordered by expected value per week of work.

## Data to gather

| Data | Why | Unlocks |
|---|---|---|
| **Logged propensities and full candidate sets** (the API logs both; start collecting) | Today's OPE infers μ from impression shares, and its ESS is ~4 % | Trustworthy policy comparison; unbiased counterfactual training |
| **Conversation content at ad time** (topic, sentiment, intent embeddings of the last turns, privacy-filtered) | Turn and session length are flat; what the user is talking about is the obvious missing context | Context × ad semantic matching, the core promise of "contextual" ads |
| **Creative content** (ad text and image embeddings, landing category) | C14–C21 are opaque ids; 45 % of test creatives are new | Cold start for ads, which matters more here than cold start for characters |
| **Post-click outcomes** (dwell, conversion, hide or report) | Clicks reward curiosity and accidental taps; companion apps care about churn | Multi-objective ranking (CTR × CVR, minus annoyance) |
| **Real, consented user ids** | 82 % of devices are placeholders | User-level fatigue and personalisation |
| **Advertiser brand-safety preferences and bids** | Currently illustrative | Real gating and auction pricing |
| **Free-text character personas** | Current ones are templated | Re-run the text bake-off: it is where embeddings should pay |

## Models to try

1. **Sequence model over the session.** A small transformer over the last k turns (content embeddings) and previous ad interactions, as an extra DCN tower.
2. **Semantic affinity distilled from an LLM.** Score (persona, conversation snippet, ad creative) with a strong LLM offline, then distil it into a feature or a two-tower model for the hot path.
3. **Two-tower retrieval** for the candidate-generation stage the brief assumes exists; the ranking model becomes its second stage.
4. **Multi-task CTR + CVR** (ESMM-style) once conversions are logged.
5. **Counterfactual learning** (IPS or DR-weighted training) on the logged propensities, closing the exploration loop.
6. **Contextual bandit for exploration**, replacing the fixed 5 % with LinTS/NeuralTS on the DCN's last layer, with exploration budgeted per advertiser.

## Scaling and productionisation

- **Feature store:** move user counters to a streaming job (Kinesis/Flink) writing Redis, so the API never does read-modify-write. Add Feast if more online features arrive.
- **Serving:** Graviton tasks (ONNX Runtime on ARM64) and int8 quantisation if candidate counts grow. Shadow and canary through the existing bundle pointer: `bundles/current` becomes `bundles/{control,candidate}` with a hashed traffic split.
- **Online experimentation:** an A/B framework with CUPED and guardrail metrics (latency, no-fill, brand-safety incidents). The adaptation λ and the exploration rate are the first experiments.
- **Monitoring:** daily PSI and calibration-ratio jobs feeding the existing alarms; a slice dashboard (genre × surface × new or returning).
- **Governance:** model cards generated from `metrics.json` and the ADRs; a data-retention policy for decision logs (privacy: no raw IPs downstream of the proxy hash).
