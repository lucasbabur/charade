# 🎬 Cameo

**Ads that know their scene.** Cameo picks which ad gets a moment in an AI companion chat. It predicts calibrated click probability for every candidate, removes the ones that don't belong, and returns an explained ranking in under 50 ms.

> The right ad makes a cameo. It never takes over the scene.

A user flirting with an AI girlfriend mid-roleplay and a user two messages into a fresh chat with AI Superman are different audiences. Cameo scores each candidate against the character (genre, safety tier, persona), the moment (turn, hour), the surface, and the user's recent exposure. It then decides what to show, and how confident it is.

---

## ✨ What it does

| | |
|---|---|
| 🎯 **CTR prediction** | DCN-v2 in PyTorch, judged against a LightGBM yardstick. Calibrated so `pCTR × bid` means something |
| 🏆 **Candidate ranking** | Brand-safety gates → batched scoring → pacing and fatigue → Thompson exploration on 5 % of traffic, with every propensity logged |
| 🧊 **Cold start** | New characters scored from genre, tier and persona text; beta-binomial priors; an evidence-based graduation rule |
| 🌊 **Drift** | PSI, character churn, novelty decay, and an adaptation layer that holds CTR while spreading exposure |
| ⚡ **Serving** | FastAPI + ONNX Runtime + Redis, under 50 ms p99, observable end to end |
| 🛡️ **mlcheck** | 46 ML release gates: leakage, reproducibility, calibration with CIs, train/serve parity, latency, policy safety |

## 🔍 What the data told us

- 🎭 **Genre matters most.** Romance and horror characters run ~+4 pp CTR, mentors −4 pp. Mature tier adds +2–3 pp every day.
- 👻 **Most "users" are ghosts.** One placeholder `device_id` covers 82 % of rows, and 81 % of IPs appear once, so cold start is the default case.
- 🧩 **The anonymized C-columns hide an ad hierarchy:** creative (C14) → campaign (C17) → advertiser (C21). That hierarchy defines what a candidate is.
- 📉 **Conversation turn barely moves CTR**, and the character descriptions are templated. Text helps cold start; it is not a big lift on its own.

Full evidence: [docs/PLAN.md §0](docs/PLAN.md#0-data-findings-that-decide-the-design).

## 🚀 Quickstart

```bash
uv sync --all-packages                                # 📦 install the workspace
cp /path/to/{impressions,characters}.csv .            # 🗂️ raw data (gitignored)
uv run mlcheck . --stage data --stage static          # 🛡️ data contract + code gates
uv run mlcheck --list                                 # 📋 all 46 gates and why each exists
```

> 🚧 **Status:** design ✅ · ML gates ✅ · model, ranking and API in progress. Live status per doc: [docs/index.md](docs/index.md).

## 🗺️ Repository map

```
src/cameo/        🧠 data · features · text · models · evaluation · ranking · coldstart · drift · serving
tools/mlcheck/    🛡️ ML release gates
configs/          ⚙️ experiments and policy as YAML
reports/          📊 generated figures and tables
infra/terraform/  ☁️ AWS: ECS Fargate, ElastiCache, S3, Firehose
docs/             📚 everything explained
```

## 📚 Read next

| | |
|---|---|
| 🧭 [docs/index.md](docs/index.md) | Every doc, its status, and the reading path for each task |
| 🏗️ [docs/PLAN.md](docs/PLAN.md) | Full design: split, features, models, ranking, cold start, drift, serving, infrastructure |
| 🛡️ [docs/mlcheck.md](docs/mlcheck.md) | The release gates, artifact contract and statistics |
| 🤖 [AGENTS.md](AGENTS.md) | Rules and invariants for AI agents and contributors |

---

<sub>Built as the Simula ML Engineer take-home · Python 3.12 · uv · PyTorch · ONNX · FastAPI · Terraform</sub>
