---
title: 'Scene Transformer: A Unified Architecture for Predicting Multiple Agent Trajectories'
date: '2021-06-15T00:00:00.000Z'
section: paper-shorts
postSlug: scene-transformer-a-unified-architecture-for-predicting-multiple-agent-trajectories
legacyPath: /paper shorts/2021/06/15/scene-transformer-a-unified-architecture-for-predicting-multiple-agent-trajectories.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2021 – Scene Transformer: A Unified Architecture for Predicting Multiple Agent Trajectories"
---

**arXiv:** [2106.08417](https://arxiv.org/abs/2106.08417)

**OpenReview:** [ICLR 2022 submission](https://openreview.net/forum?id=7a2BujHKS7)

## Summary

> Scene Transformer makes the forecasting query a mask over a scene tensor. Instead of maintaining separate networks for marginal, joint, conditional, and goal-conditioned prediction, it keeps one scene-centric representation and changes which agent-time cells are visible. The same Transformer can therefore answer different planning questions by changing the information pattern at its input. The design targets a real failure mode of independent predictions: two individually plausible futures may be mutually inconsistent, so Scene Transformer predicts a set of coherent futures across agents and evaluates them with scene-level metrics when joint consistency is the goal. Its contribution is the interface between masking, factorized attention, and a joint loss, more than a new trajectory decoder.

## Core Insights

### A mask is a reusable forecasting interface

The model embeds agent histories and static/dynamic road graph features into an `[A, T, D]` tensor and decodes `[F, A, T, D]` future hypotheses. A visible cell is conditioning information; a hidden cell is imputed by the model. Motion prediction (MP), conditional motion prediction (CMP), and goal-conditioned prediction (GCP) are different masks over the same tensor, so task changes do not require new output heads.

![Scene Transformer's masking strategies and factorized encoder-decoder](/assets/images/scene-transformer-a-unified-architecture-for-predicting-multiple-agent-trajectories-paper-figure.png)
*Fig 1: The left masks define MP, CMP, and GCP queries; the right alternates attention over time and agents and cross-attends to the road graph. | source: [Scene Transformer, Figure 2](https://arxiv.org/abs/2106.08417)*

The decoder predicts seven values per agent-time cell: three position coordinates, three Laplace uncertainty parameters, and heading. A six-future output is trained with a reduce-min objective. For marginal prediction, each agent chooses its closest future independently; for joint prediction, the displacement loss is aggregated over the selected agents and time before choosing one coherent future. That small change in the loss changes the semantics of the output modes.

### Factorized attention preserves the two dependencies that matter

Naively attending over all `A × T` states is expensive and can make identical masked agents indistinguishable. Scene Transformer alternates time attention and agent attention: the time operation can learn smooth trajectories, while the agent operation can learn interactions independent of a particular timestep. Cross-attention injects road graph features, which are shared and permutation-equivariant rather than re-encoded separately for every agent. The resulting model is permutation-equivariant over the agent ordering.

![Scene Transformer's marginal and joint prediction illustration](/assets/images/scene-transformer-a-unified-architecture-for-predicting-multiple-agent-trajectories-source-figure-1.webp)
*Fig 2: Marginal futures can be individually plausible while joint futures preserve a consistent interaction between agents. | source: [Scene Transformer, Figure 1](https://arxiv.org/abs/2106.08417)*

### Joint loss improves consistency, but marginal and joint scores are different questions

On Argoverse, the marginal model reports minADE 0.80, minFDE 1.23, and miss rate 0.13 at six predictions. On the WOMD test split, its vehicle/pedestrian/cyclist minADE is 1.17/0.60/1.17, with mAP 0.27/0.23/0.20. For the joint interactive task, the multi-task joint model reports scene-level minSADE 1.74/1.41/1.95 and minSFDE 4.06/3.26/4.68 for vehicles/pedestrians/cyclists, with SMR 0.50/0.64/0.71 and mAP 0.13/0.04/0.03.

Those numbers should not be compared as if they were one leaderboard. Marginal minADE asks whether each agent has a close trajectory among six; minSADE asks whether one shared future is close for the scene. The paper's interaction ablation makes the causal point: on the WOMD test split, the joint multi-task model improves vehicle minSADE from 2.08 for “marginal-as-joint” to 1.74, and vehicle mAP from 0.08 to 0.13.

![Scene Transformer's goal-conditioned prediction examples](/assets/images/scene-transformer-a-unified-architecture-for-predicting-multiple-agent-trajectories-source-figure-3.webp)
*Fig 3: Supplying a goal changes the same scene model into a goal-conditioned predictor, with the remaining agents and road graph still in context. | source: [Scene Transformer, Figure 3](https://arxiv.org/abs/2106.08417)*

## High-Level Takeaways

- Scene Transformer is a useful template when several forecasting tasks should share scene understanding: the agent-time mask is the query, and the attention stack stays fixed.
- Joint prediction only becomes joint in the loss and the mode semantics. A marginal model evaluated with a scene metric does not acquire interaction consistency automatically.
- Factorizing time and agent attention keeps the architecture tractable while preserving the two dependencies that matter most for motion; road-graph cross-attention supplies the static context.
- The unification trades specialized capacity for a reusable interface. Larger scenes, more agents, and many masked patterns still increase memory and may favor specialized decoders for rare conditional queries.
