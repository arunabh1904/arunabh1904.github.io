---
title: 'MotionLM: Multi-Agent Motion Forecasting as Language Modeling'
date: '2023-09-28T04:00:00.000Z'
section: paper-shorts
postSlug: motionlm-multi-agent-motion-forecasting-as-language-modeling
legacyPath: /paper shorts/2023/09/28/motionlm-multi-agent-motion-forecasting-as-language-modeling.html
tags:
  - Other
field: BEV
summary: MotionLM discretizes continuous trajectories into motion tokens and forecasts interacting road agents with an autoregressive language-model objective.
---
## 2023 - MotionLM

**arXiv:** [2309.16534](https://arxiv.org/abs/2309.16534)

**Summary:** MotionLM asks a simple question: what if multi-agent motion forecasting is a language modeling problem? It tokenizes future trajectories and trains an autoregressive Transformer to predict sequences of motion tokens.

That reframing removes several pieces of hand-built forecasting machinery. The model does not need anchors or explicit latent-variable optimization for multimodality, and it can generate joint futures for interacting agents in one decoding process.

## Paper Insights

The problem is forecasting plausible, interactive futures for multiple road agents. MotionLM turns continuous trajectories into discrete motion tokens and maximizes the average log probability of those tokens with a standard language-model objective. Its sequential factorization gives the model temporally causal conditional rollouts, which matters when one agent's future should react to another's predicted motion.

The main evidence is performance on the Waymo Open Motion Dataset. The paper reports state-of-the-art multi-agent motion prediction and a first-place rank on the interactive challenge leaderboard. The caveat is shared with most tokenized forecasting systems: discretization simplifies the objective, but token design and decoding strategy become part of the modeling assumptions.

![Figure 2 from MotionLM showing scene encoding, autoregressive motion-token decoding, and rollout aggregation](/assets/images/motionlm-multi-agent-motion-forecasting-as-language-modeling-paper-figure.png)
_Figure 2 shows the language-model analogy concretely: scene features condition an autoregressive decoder that rolls out discrete motion tokens. From the [MotionLM paper](https://arxiv.org/abs/2309.16534), via ar5iv._

**What to look at:**
- Motion tokens replace anchors and manually specified multimodal heads.
- Joint autoregressive decoding models interactions directly.
- Temporally causal rollouts make conditional prediction natural.

**Evals / Benchmarks / Artifacts:**

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Representation | Discrete motion tokens | Brings trajectory forecasting into the next-token framework. |
| Objective | Standard language-model likelihood | Simplifies multimodal forecasting training. |
| Benchmark | Waymo Open Motion Dataset interactive challenge | Tests joint futures, not only independent agent predictions. |

**Compact result slice:**

| Model | Interactive minADE | Interactive miss rate | Interactive mAP |
| ----- | ------------------ | --------------------- | --------------- |
| MTR | 0.9181 | 0.4411 | 0.2037 |
| JFP | 0.8817 | 0.4233 | 0.2050 |
| MotionLM | 0.8911 | 0.4115 | 0.2178 |

**Context:** MotionLM made the language-model analogy concrete for autonomous-driving behavior prediction.

**Takeaway:** A good tokenization can turn motion forecasting into sequence modeling, but the planner still has to care about calibration, coverage, and interaction quality.
