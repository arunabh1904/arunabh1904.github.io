---
title: "Unified Pedestrian Path Prediction Using Inverse Reinforcement Learning"
date: '2026-08-16T00:00:00.000Z'
section: paper-shorts
postSlug: unified-pedestrian-path-prediction-using-inverse-reinforcement-learning
legacyPath: /paper shorts/2026/08/16/unified-pedestrian-path-prediction-using-inverse-reinforcement-learning.html
tags:
  - Autonomous Driving
  - Pedestrian Prediction
  - Reinforcement Learning
field: 'Motion Forecasting & Planning'
summary: "2026 – Unified Pedestrian Path Prediction Using Inverse Reinforcement Learning"
---

## 2026 – Unified Pedestrian Path Prediction Using Inverse Reinforcement Learning

**arXiv:** [2608.15929](https://arxiv.org/abs/2608.15929)<br />
**Code:** [project repository](https://github.com/Simsuk/IRL_STGAT)

## Summary

> This paper reformulates pedestrian path prediction as a decision process instead of treating it only as supervised coordinate regression. It adapts a Spatial-Temporal Graph Attention Network with task-specific state and action definitions, then compares deterministic and stochastic policies, one-time and sequential decisions, REINFORCE, and PPO. The reported formulations improve prediction across the selected benchmarks over the standard supervised objective.

## Core Insights

The technical change is in the learning contract. STGAT already represents pedestrians and their interactions as a graph over time; the paper changes what the network is asked to optimize. A state can encode the current graph and motion context, while actions represent the next path decision. That interface supports both policy-gradient methods and a deterministic policy, so the same predictor can be studied under multiple decision formulations.

![Pedestrian trajectory-prediction overview comparing supervised sequence prediction with decision-oriented policy formulations](/assets/images/unified-pedestrian-path-prediction-paper-figure.webp)

_The same social trajectory encoder can be trained under supervised, one-step, sequential, deterministic, or stochastic decision formulations; the paper studies the objective rather than replacing the perception backbone. Source: [Unified Pedestrian Path Prediction](https://arxiv.org/abs/2608.15929), Figure 1._

The result is a comparative study rather than a new perception architecture. Its claim is that advanced graph predictors can still benefit from a decision-oriented objective, extending earlier formulation comparisons that used shallow models. The abstract does not report a single cross-dataset percentage gain in the available source summary, so the note should not invent one; the important evidence is the controlled comparison against supervised learning.

The open decision is whether inverse reinforcement learning improves behavior because it captures a useful latent cost or because its rollout and objective choices regularize the predictor. A matched study should hold STGAT, data, horizon, and evaluation fixed while varying only state/action definitions and policy objective.

## High-Level Takeaways

- The paper informs whether pedestrian prediction should be trained as a sequential decision problem when the architecture already models social interaction.
- The atomic unit is a pedestrian graph state and next-path action, with deterministic or stochastic policy variants.
- Reformulation can matter independently of model size, but the reported gains do not establish better closed-loop vehicle safety.
- The conclusion would weaken if supervised STGAT matches the policy formulations under equal rollout, reward-design, and compute budgets.
