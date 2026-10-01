---
title: "Models and evaluation"
created-at: 2026-10-01
updated-at: 2026-10-01
---

# 04 — Models and evaluation

**Bottom line:** a 3-seed DCN-v2 ensemble, isotonic-calibrated, reaches **test NE 0.8848** (AUC 0.737, pred/obs 1.012), beating LightGBM by −0.0040 [−0.0052, −0.0027] and logistic by −0.0120 [−0.0138, −0.0102] log loss. Validation ablations dropped character ID and conversation features and rejected text ([metrics.json](../reports/models/metrics.json)).

## Protocol

| Step | Data | Rationale |
|---|---|---|
| Hyperparameter search (Optuna TPE, 60 DCN + 40 LightGBM trials) | Train 10-21..26, select on 10-27 | Keeps 10-28 clean for early stopping, calibration and comparisons |
| Training + early stopping (4 validation passes per epoch, patience 4) | Train 10-21..27; stop on 10-28 | Production retrains on everything up to yesterday |
| Calibration choice (identity / Platt / isotonic) | 10-28, 2-fold over even/odd hours | Fitting on the most recent day absorbs the daily CTR level shift (H9) |
| Ablations | 10-28, raw logits | Group decisions never look at test |
| **Test** | 10-29 + 10-30 05h | Every scoring logged with a configuration hash; at most 3 configurations per model family may look at this window (MLL006) |

**Primary metric: normalized entropy (NE)**, the log loss divided by the entropy of the evaluation set's base rate. The auction ranks by pCTR × bid, so the probabilities must be right, not just well ordered. AUC, calibration ratio and ECE are reported alongside. Any claim that one model beats another needs a paired bootstrap that resamples whole hours, because rows within an hour share traffic mix.

## Models

| Model | Test NE | Test log loss | Test AUC | Pred/obs | ECE | Calibrator |
|---|---|---|---|---|---|---|
| Logistic regression (same inputs, one weight per value) | 0.9110 | 0.4174 | 0.709 | 0.985 | 0.0076 | identity |
| LightGBM (tuned, native categoricals) | 0.8934 | 0.4093 | 0.729 | 0.980 | 0.0102 | Platt |
| **DCN-v2, 3-seed ensemble** | **0.8848** | **0.4053** | **0.737** | 1.012 | 0.0048 | isotonic |

DCN-v2 configuration (tuned): 24-dim field embeddings → 2 low-rank cross layers (rank 64) → MLP 256-128 (dropout 0.3) → logit. AdamW, lr 2.9e-3, batch 2048, weight decay 1.3e-5. Single-seed validation NE (uncalibrated) is 0.8761, 0.8758 and 0.8776. The ensemble adds about 0.004 on top. Training all three models takes **53 s** on the GTX 1660 SUPER.

Why DCN-v2 ships: H1 holds. Genre × campaign interactions are real (residual sd 2.7 pp against 1.1 pp noise), and a network that learns explicit crosses beats both an additive model and trees on the same inputs. It also exports to a single ONNX graph with max |Δ logit| = 1.9e-6 against PyTorch.

Calibration: isotonic wins the hour-fold CV narrowly (log loss 0.39065 against 0.39070 for identity). On test it improves NE from 0.8855 to 0.8848. The predicted/observed ratio moves from 1.002 to 1.012; both are inside the [0.9, 1.1] gate.

## Ablations (validation, 3-seed ensembles, Δ log loss against all non-text groups; positive = worse)

| Variant | Val NE | Δ log loss [95 % CI] | Decision |
|---|---|---|---|
| All features | 0.8720 | — | reference |
| − character metadata (genre, tier, creator, popularity, age) | 0.8792 | +0.0032 [+0.0027, +0.0037] | keep |
| − character ID | 0.8720 | −0.0000 [−0.0002, +0.0002] | **drop** (CI includes 0) |
| − all character features | 0.8951 | +0.0104 [+0.0095, +0.0114] | keep the character layer |
| − user history counters | 0.8739 | +0.0009 [+0.0004, +0.0013] | keep |
| − conversation (turn, session length) | 0.8722 | +0.0001 [−0.0000, +0.0003] | **drop** (CI includes 0) |
| − device | 0.8763 | +0.0019 [+0.0015, +0.0026] | keep |
| + description text (Qwen3) | 0.8738 | +0.0008 [+0.0006, +0.0011] | **reject** (hurts) |
| + description text (TF-IDF) | 0.8737 | +0.0008 [+0.0006, +0.0010] | reject |
| No character-ID dropout | 0.8712 | −0.0003 [−0.0007, −0.0001] | moot once ID is dropped |
| **Shipped: − character ID − conversation** | **0.8712** | **−0.0003 [−0.0007, −0.0000]** | ship |

Reading the table: the character layer is worth 0.0104 log loss, more than device and user history combined, and genre, tier and the other metadata carry all of it. The per-character ID adds nothing once metadata is present. That fits the EDA, where each character adds only about 1 pp of real variation beyond genre, and it is good news for cold start (§ Slices). Text hurts: 16 extra dense inputs with no signal (see [03](03-text-enrichment.md)) cost about 0.0008.

## Slices (test, shipped model; full table in `metrics.json`)

| Slice | Rows | NE | Pred/obs |
|---|---|---|---|
| Character unseen in training | 1,478 | 0.906 | 1.08 |
| Character with 1–20 training impressions | 2,549 | 0.887 | 1.04 |
| 21–200 | 30,308 | 0.884 | 1.01 |
| > 200 | 93,071 | 0.885 | 1.01 |
| New user (no earlier impressions) | 97,843 | 0.894 | 1.00 |
| Returning user | 29,563 | 0.856 | 1.04 |
| App surface / site surface | 52,360 / 75,046 | 0.875 / 0.898 | 0.97 / 1.03 |
| `banner_pos` 0 / 1 | 87,995 / 39,110 | 0.873 / 0.914 | 1.00 / 1.03 |
| 2014-10-29 / 2014-10-30 (partial) | 104,450 / 22,956 | 0.885 / 0.885 | 1.01 / 1.02 |

- **Cold characters:** NE is 2.4 % worse than for warm characters. That is inside the 3 % target (H10), but the slice is small and slightly over-predicted. See [06-cold-start.md](06-cold-start.md).
- **Romance and horror:** the highest-CTR genres have the weakest NE (0.900 and 0.905). Their CTR is closer to 0.5, so there is less entropy to remove relative to the base rate. No slice is worse than its own base rate (MLM009).

## Leakage and shift probes (`artifacts/current/leakage.json`)

- **Shuffled-label retrain:** validation AUC 0.472, inside the chance band. Nothing in the pipeline leaks the label.
- **Single features:** the strongest feature on its own is `site_id` at AUC 0.671. No near-perfect single feature exists.
- **Adversarial validation: train vs test AUC 0.959 (WARN).** The separating features are C14, C17 and C19 (creatives and campaigns rotate: 45 % of test rows show a creative absent from training), then the character ID and the cumulative counters, which grow with time by construction. This is real ad rotation, not leakage. It is why the ad hierarchy (creative → campaign → advertiser) and daily retraining matter; see [07-drift-adaptation.md](07-drift-adaptation.md).

## Honesty notes

- The first end-to-end run used all feature groups and also scored test, before the ablations existed (ledger run `20261001T050637Z-02e0ea61`). The group decision was made from validation ablations only. That run's test NE (0.8855) is within noise of the shipped run's (0.8848), so the holdout did not steer the choice.
- Tuning picked configurations on 10-27, a high-CTR day (19.6 %). Validation and test are low-CTR days. Calibration on 10-28 absorbs the level difference; the ranking quality transferred (inner NE 0.8746 against validation 0.8693).
