---
title: "Models and evaluation"
created-at: 2026-10-01
updated-at: 2026-10-03
---

# 03 — Models and evaluation

**Bottom line:** a 3-seed DCN-v2 ensemble reaches **test NE 0.8855** (AUC 0.737, pred/obs 1.002, ECE 0.005), beating an equally tuned 3-seed LightGBM ensemble by −0.0040 [−0.0053, −0.0027] and logistic by −0.0117 [−0.0136, −0.0098] log loss. Single model against single model, the gap halves to −0.0020 [−0.0041, +0.0004] and is not established. Ablations repeated with two independent seed sets dropped character ID and conversation features and rejected text ([metrics.json](../reports/models/metrics.json)).

## Protocol

| Step | Data | Rationale |
|---|---|---|
| Hyperparameter search (Optuna TPE, 60 DCN + 60 LightGBM trials) | Train 10-21..26, select on 10-27 | Keeps 10-28 clean for early stopping, calibration and comparisons |
| Training + early stopping (4 validation passes per epoch, patience 4) | Train 10-21..27; stop on 10-28 | Production retrains on everything up to yesterday |
| Calibration choice (identity / Platt / isotonic) | 10-28, 2-fold over even/odd hours | Fitting on the most recent day absorbs the daily CTR level shift (H9) |
| Ablations | 10-28, raw logits | Group decisions never look at test |
| **Test** | 10-29 + 10-30 05h | Read only to confirm choices made on validation; exceptions are listed under Honesty notes |

**Primary metric: normalized entropy (NE)**, the log loss divided by the entropy of the evaluation set's base rate. The auction ranks by pCTR × bid, so the probabilities must be right, not just well ordered. AUC, calibration ratio and ECE are reported alongside. Any claim that one model beats another needs a paired bootstrap that resamples whole hours, because rows within an hour share traffic mix.

## Models

| Model | Test NE | Test log loss | Test AUC | Pred/obs | ECE | Calibrator |
|---|---|---|---|---|---|---|
| Logistic regression (same inputs, one weight per value) | 0.9110 | 0.4174 | 0.709 | 0.985 | 0.0076 | identity |
| LightGBM, 3-seed ensemble (60 Optuna trials) | 0.8942 | 0.4098 | 0.729 | 0.979 | 0.0111 | Platt |
| **DCN-v2, 3-seed ensemble (60 Optuna trials)** | **0.8855** | **0.4056** | **0.737** | 1.002 | 0.0053 | identity |

| Comparison (test, paired hour-block bootstrap) | Δ log loss [95 % CI] |
|---|---|
| DCN-v2 − logistic | −0.0117 [−0.0136, −0.0098] |
| DCN-v2 − LightGBM, 3-seed ensembles, calibrated | −0.0040 [−0.0053, −0.0027] |
| DCN-v2 − LightGBM, one seed each, uncalibrated | −0.0020 [−0.0041, +0.0004] |

DCN-v2 configuration (tuned): 24-dim field embeddings → 2 low-rank cross layers (rank 64) → MLP 256-128 (dropout 0.3) → logit. AdamW, lr 2.9e-3, batch 2048, weight decay 1.3e-5. Single-seed validation NE (uncalibrated) is 0.8761, 0.8758 and 0.8776. Training everything takes about 76 s on the GTX 1660 SUPER.

**Fairness of the comparison.** Both models get the same tuning effort (60 trials each on the inner split) and the same 3 seeds. Ensembling helps DCN-v2 more than LightGBM: the single-model gap is about half the ensemble gap, and its interval includes zero. What is established: the shipped DCN-v2 ensemble beats the LightGBM ensemble. What is not: that the architecture alone beats boosted trees on this data. LightGBM also gets DCN's inputs (the same encoded ids, as native categoricals); with feature engineering of its own (target encodings, count features, explicit crosses) it might narrow the gap. E010 reports single-seed NE on one shared test set without the character layer: DCN-v2 0.873 → 0.857 and LightGBM 0.875 → 0.855 as training rows increase from 0.74M to 6.7M. Validation volume also changes. These are descriptive results without paired intervals or seed-variability estimates, not a quantified ratio of volume and architecture effects. The logistic baseline is not tuned (fixed learning rate): it is a floor, not a tuned rival. H1 (interactions are real) rests on the EDA residuals and on both models beating the additive logistic baseline by a wide margin. DCN-v2 ships for the ensemble result and because it exports to one ONNX graph (|Δ logit| ≤ 1.9e-6 against PyTorch).

**Calibration.** Candidates are scored by 2-fold CV over validation hours. A more flexible map must beat a simpler one by more than 1e-4 log loss per row, or the simpler one wins. For DCN-v2 the margins are about 4e-5 (isotonic) and 1e-5 (Platt), both noise, so it ships uncalibrated (identity). Test pred/obs is 1.002. An earlier version took isotonic for its 0.00003 win; that bought nothing on test and created calibration plateaus that tied candidates' pCTRs.

**Refit on the validation day: tested, not shipped** ([refit_study.json](../reports/models/refit_study.json), `uv run poe refit-study`). The validation day is the most recent data the shipped model never trains on, so I asked whether to retrain on train + validation for the step count early stopping chose. I decided it one day earlier, never on test: early-stop and calibrate on 10-27, then refit through 10-27, both scored on 10-28. The refit changes log loss by +0.00001 [−0.0008, +0.0009] and worsens calibration (pred/obs 1.08 against 1.02). The early-stopped model ships; daily retraining already brings the newest day in a day later.

## Ablations (validation, 3-seed ensembles, two independent seed sets; Δ log loss against all non-text groups; positive = worse)

The bootstrap over hours measures data noise only, so every comparison is run twice with different seeds (reference and variant both retrained). An effect is real only when both seed sets agree on its sign; the interval shown is the union of the two.

| Variant | Val NE | Δ log loss [union of 95 % CIs] | Seed set A / B | Verdict → decision |
|---|---|---|---|---|
| All features | 0.8718 | — | — | reference |
| − character metadata (genre, tier, creator, popularity, age) | 0.8795 | +0.0035 [+0.0027, +0.0043] | +0.0032 / +0.0037 | worse → keep |
| − character ID | 0.8716 | −0.0001 [−0.0006, +0.0002] | −0.0000 / −0.0002 | noise → **drop** |
| − all character features | 0.8944 | +0.0101 [+0.0089, +0.0114] | +0.0104 / +0.0099 | worse → keep the character layer |
| − user history counters | 0.8744 | +0.0012 [+0.0004, +0.0018] | +0.0009 / +0.0015 | worse → keep |
| − conversation (turn, session length) | 0.8722 | +0.0002 [−0.0000, +0.0004] | +0.0001 / +0.0002 | noise → **drop** |
| − device | 0.8762 | +0.0020 [+0.0015, +0.0027] | +0.0020 / +0.0020 | worse → keep |
| + description text (Qwen3) | 0.8733 | +0.0007 [+0.0003, +0.0011] | +0.0008 / +0.0005 | worse → **reject** |
| + description text (TF-IDF) | 0.8729 | +0.0005 [+0.0000, +0.0010] | +0.0008 / +0.0003 | worse (barely) → reject |
| No character-ID dropout | 0.8716 | −0.0001 [−0.0007, +0.0003] | −0.0003 / +0.0002 | noise |
| **Shipped: − character ID − conversation** | **0.8714** | **−0.0002 [−0.0007, +0.0004]** | −0.0003 / −0.0000 | noise → ship the simpler model |

Reading it: the character layer is worth about 0.010 log loss, more than device and user history combined, and metadata carries all of it. Character ID, conversation features and ID dropout are noise. The earlier single-seed-set table showed the shipped variant as a small improvement (−0.0003); with seed variance included it is indistinguishable from all features. The decision stands for a different reason: equal performance with fewer inputs and no identity parameter. Text is worse under both seed sets.

## Slices (test, shipped model; full table in `metrics.json`)

| Slice | Rows | NE | Pred/obs |
|---|---|---|---|
| Character unseen in training | 1,478 | 0.908 | 1.07 |
| Character with 1–20 training impressions | 2,549 | 0.889 | 1.03 |
| 21–200 | 30,308 | 0.885 | 1.00 |
| > 200 | 93,071 | 0.885 | 1.00 |
| New user (no earlier impressions) | 97,843 | 0.895 | 0.99 |
| Returning user | 29,563 | 0.856 | 1.03 |
| App surface / site surface | 52,360 / 75,046 | 0.875 / 0.899 | 0.95 / 1.03 |
| `banner_pos` 0 / 1 | 87,995 / 39,110 | 0.874 / 0.914 | 0.99 / 1.02 |
| 2014-10-29 / 2014-10-30 (partial) | 104,450 / 22,956 | 0.885 / 0.886 | 1.00 / 1.01 |

- **Cold characters:** NE is 2.5 % worse than for warm characters, on a small slice (n = 1,478, no interval) that is over-predicted (1.07). See [05-cold-start.md](05-cold-start.md).
- **Romance and horror:** the highest-CTR genres have the weakest NE (0.901 and 0.907). Their CTR is closer to 0.5, so there is less entropy to remove relative to the base rate. No slice is worse than its own base rate (MLM009).

## Tested and rejected: character text (E002)

**Result:** descriptions are templates: embeddings recover genre perfectly but predict no CTR beyond genre × tier (ρ ≈ 0 ± 0.1), and adding them hurts validation log loss ([03](03-models-evaluation.md)). The pipeline stays for real free-text personas.

### Bake-off ([reports/text_bakeoff.csv](../reports/text_bakeoff.csv), `uv run poe text`)

| Provider | Dims | Genre 5-NN acc | Tier 5-NN acc | Residual ρ (95 % CI), 363 val characters |
|---|---|---|---|---|
| TF-IDF 1–2-grams → SVD | 128 | 1.00 | 0.50 | 0.017 (−0.086, 0.120) |
| Qwen3-Embedding-0.6B (local, GTX 1660 SUPER) | 1024 | 1.00 | 0.50 | 0.001 (−0.102, 0.104) |

The residual is each character's training CTR (≥ 100 impressions) minus its genre × tier rate. A ridge model on the embedding is trained on training characters and scored by Spearman ρ on validation characters.


### Pipeline rules (if text ships later)

- Raw embeddings are cached in `artifacts/cache/text/` (gitignored), keyed by provider and a hash of the texts.
- Only 16-dim PCA reductions are committed, in `data/derived/text_<provider>.parquet` (~0.3 MB each), so results reproduce without keys or a GPU.
- PCA is fitted only on characters created by the end of training. New characters are projected with the stored transform when they are published.
- Embedding never runs in serving, tests or CI. Serving reads the character table with precomputed vectors.

Paid embedding APIs (OpenAI, Gemini, Voyage) were not run (no usable keys); with ρ ≈ 0 for two very different embedders, a third would not change the decision.

## Leakage and shift probes (`artifacts/current/leakage.json`)

- **Shuffled-label retrain:** clicks are shuffled on the raw rows *before* the label-derived counters and features are rebuilt, validation labels with them, and the baseline is retrained; mean AUC over 3 seeds is **0.4999**. With a same-hour counter leak injected (the Avazu trap, H5) the same probe scores **0.983**, so it catches leaks inside feature construction. An earlier version shuffled after the features were built: it could not see such a leak and drifted between 0.48 and 0.55 across seeds because early stopping used real labels. The gate (MLL004) is one-sided, failing only above 0.53, because a leak can only raise the AUC. It still does not certify serve-time availability of non-label features; that rests on the causal counters (property-tested) and the train-only fits in [02](02-features.md).
- **Single features:** the strongest feature on its own is `site_id` at AUC 0.671. No near-perfect single feature exists.
- **Adversarial validation: train vs test AUC 0.959 (WARN).** The separating features are C14, C17 and C19 (creatives and campaigns rotate: 43 % of test rows show a creative absent from training), then the character ID and the cumulative counters, which grow with time by construction. This is real ad rotation, not leakage. It is why the ad hierarchy (creative → campaign → advertiser) and daily retraining matter; see [06-drift-adaptation.md](06-drift-adaptation.md).

## Honesty notes

- The first end-to-end run used all feature groups and also scored test, before the ablations existed. The group decision was made from validation ablations only. That run's test NE (0.8855) equals the shipped run's (0.8855), so the holdout did not steer the choice.
- **Choices made on test, then redone.** The adaptation penalty λ and the decision not to ship online recalibration were first assessed on the test days. λ is now selected on the validation day by a pre-registered rule, and test only confirms ([06](06-drift-adaptation.md)). The recalibration decision is supported by the validation day alone (NE 0.8712 frozen vs 0.8714 recalibrated).
- Tuning picked configurations on 10-27, a high-CTR day (19.6 %). Validation and test are low-CTR days. Early stopping on 10-28 anchors the level (test pred/obs 1.002 uncalibrated); observed NE is 0.8746 on the inner split and 0.8712 on validation; these different-population probability metrics do not establish within-opportunity ranking transfer.
