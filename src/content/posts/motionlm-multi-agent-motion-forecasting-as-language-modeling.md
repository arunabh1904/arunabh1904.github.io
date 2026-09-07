---
title: 'MotionLM: Multi-Agent Motion Forecasting as Language Modeling'
date: '2023-09-28T00:00:00.000Z'
section: paper-shorts
postSlug: motionlm-multi-agent-motion-forecasting-as-language-modeling
legacyPath: /paper shorts/2023/09/28/motionlm-multi-agent-motion-forecasting-as-language-modeling.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2023 – MotionLM: Multi-Agent Motion Forecasting as Language Modeling"
---
## 2023 – MotionLM

**arXiv:** [2309.16534](https://arxiv.org/abs/2309.16534)

### Method and reported result

MotionLM asks a simple question: what if multi-agent motion forecasting is a language modeling problem? It tokenizes future trajectories and trains an autoregressive Transformer to predict sequences of motion tokens.

## Summary

> That reframing removes several pieces of hand-built forecasting machinery. The model does not need anchors or explicit latent-variable optimization for multimodality, and it can generate joint futures for interacting agents in one decoding process.

## Core Insights

The problem is forecasting plausible, interactive futures for multiple road agents. MotionLM turns continuous trajectories into discrete motion tokens and maximizes the average log probability of those tokens with a standard language-model objective. Its sequential factorization gives the model temporally causal conditional rollouts, which matters when one agent's future should react to another's predicted motion.

The main evidence is performance on the Waymo Open Motion Dataset. The paper reports state-of-the-art multi-agent motion prediction and a first-place rank on the interactive challenge leaderboard. The caveat is shared with most tokenized forecasting systems: discretization simplifies the objective, but token design and decoding strategy become part of the modeling assumptions.

### Joint likelihood is useful even when the endpoint metric is not lowest

MotionLM is trained on 1.1 million nine-second examples formed from one second of history and eight seconds of future motion. In the interactive setting it emits six joint modes, and the evaluation can roll out up to eight agent replicas with 512 samples per replica. That rollout contract makes the interaction claim concrete: the decoder is choosing later tokens after earlier agents' predicted tokens are in the context, rather than drawing six independent paths and pairing them afterward.

The numbers show why a single metric is insufficient. MotionLM reports interactive minADE 0.8911 and minFDE 2.0067, slightly above JFP's 0.8817 and 1.9905, but improves miss rate from 0.4233 to 0.4115 and interactive mAP from 0.2050 to 0.2178. The conditional factorization is also measurable: on the paper's marginal comparison, temporally causal decoding moves minADE/minFDE from 0.6069/1.2236 for the marginal model to 0.5997/1.2034, while the acausal variant reaches 0.5899/1.1804. Causal ordering therefore trades a little endpoint accuracy for a rollout that can react to the generated scene state, which is the behavior a planner needs.

![Figure 2 from MotionLM showing scene encoding, autoregressive motion-token decoding, and rollout aggregation](/assets/images/motionlm-multi-agent-motion-forecasting-as-language-modeling-paper-figure.png)
*Fig 1: Shows the language-model analogy concretely: scene features condition an autoregressive decoder that rolls out discrete motion tokens. | source: [MotionLM paper](https://arxiv.org/abs/2309.16534)*

![Figure 4 from MotionLM: Multi-Agent Motion Forecasting as Language Modeling](/assets/images/motionlm-multi-agent-motion-forecasting-as-language-modeling-source-figure-4.webp)
*Fig 2: The causal graph orders each agent’s motion token at every step and links later states across agents, defining an autoregressive joint rollout conditioned on the scene. | source: [MotionLM: Multi-Agent Motion Forecasting as Language Modeling](https://arxiv.org/abs/2309.16534)*

![Figure 1 from MotionLM: Multi-Agent Motion Forecasting as Language Modeling](/assets/images/motionlm-multi-agent-motion-forecasting-as-language-modeling-source-figure-1.webp)
*Fig 3: Our model autoregressively generates sequences of discrete motion tokens for a set of agents to produce consistent interactive trajectory forecasts. | source: [MotionLM: Multi-Agent Motion Forecasting as Language Modeling](https://arxiv.org/abs/2309.16534)*




**Compact result slice:**

| Model | Interactive minADE | Interactive miss rate | Interactive mAP |
| ----- | ------------------ | --------------------- | --------------- |
| MTR | 0.9181 | 0.4411 | 0.2037 |
| JFP | 0.8817 | 0.4233 | 0.2050 |
| MotionLM | 0.8911 | 0.4115 | 0.2178 |

## High-Level Takeaways

- MotionLM informs whether continuous multi-agent futures should be generated jointly as discrete motion tokens. The atomic unit is a quantized displacement token for a particular agent and timestep; autoregressive ordering lets each predicted move condition on the emerging joint future.
- Tokenization makes interaction modeling compatible with language-model training, but quantization error, sequence ordering, and exposure bias become part of the planner. The missing study holds backbone and sampling budget fixed across tokenized autoregression and continuous joint decoders. At 10× agents or horizon, sequence length and sampling latency grow linearly while joint combinations grow much faster. The claim would fail if a continuous decoder matched joint metrics and controllable diversity with lower latency and no quantization artifacts.
- MotionLM made the language-model analogy concrete for autonomous-driving behavior prediction.
- A good tokenization can turn motion forecasting into sequence modeling, but the planner still has to care about calibration, coverage, and interaction quality.
