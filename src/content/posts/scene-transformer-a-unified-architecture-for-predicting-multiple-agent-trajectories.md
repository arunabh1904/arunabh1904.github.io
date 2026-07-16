---
title: 'Scene Transformer: A Unified Architecture for Predicting Multiple Agent Trajectories'
date: '2021-06-15T04:00:00.000Z'
section: paper-shorts
postSlug: scene-transformer-a-unified-architecture-for-predicting-multiple-agent-trajectories
legacyPath: /paper shorts/2021/06/15/scene-transformer-a-unified-architecture-for-predicting-multiple-agent-trajectories.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: Scene Transformer uses one scene-centric attention architecture for marginal, joint, conditional, and goal-conditioned multi-agent forecasting.
---
## 2021 - Scene Transformer

**arXiv:** [2106.08417](https://arxiv.org/abs/2106.08417)

**OpenReview:** [ICLR 2022 submission](https://openreview.net/forum?id=7a2BujHKS7)

**Summary:** Scene Transformer makes multi-agent trajectory prediction a masking problem over a scene tensor. Instead of training separate systems for marginal prediction, joint prediction, conditional prediction, and goal-conditioned prediction, it uses one attention architecture and changes what the model can see.

That framing is important for planning. Independent per-agent futures can be inconsistent with each other; a scene-centric model can represent futures where agents react to one another.

## Paper Insights

The model represents the scene across agents, time, and features. Attention operates across time and agents, with cross-attention to road graph information. Different prediction tasks are expressed by masking parts of the agent-time tensor: visible cells are conditioning information, hidden cells are outputs to predict.

The paper's contribution is less about a new primitive and more about unification. A single Transformer can handle marginal, joint, conditional, and goal-conditioned prediction by changing the query/mask pattern. The tradeoff is computational: attention over heterogeneous scene elements is flexible, but scaling it requires careful factorization.

![Figure 2 from Scene Transformer showing masking strategies and the attention-based encoder-decoder architecture](/assets/images/scene-transformer-a-unified-architecture-for-predicting-multiple-agent-trajectories-paper-figure.png)
_Figure 2 shows the two key ideas: prediction tasks become mask patterns, and one attention encoder-decoder handles agent-time-road interactions. From the [Scene Transformer paper](https://arxiv.org/abs/2106.08417), via the arXiv PDF._

**What to look at:**
- Masking defines the prediction task.
- Joint prediction keeps interactions between future agents visible to the model.
- The same architecture can answer several forecasting queries.

**Evals / Benchmarks / Artifacts:**

| Idea | Detail | Why it matters |
| ---- | ------ | -------------- |
| Scene-centric tensor | Agents, time, and features in one representation | Avoids re-encoding each target independently. |
| Masked tasks | MP, CMP, and GCP as visibility patterns | Turns a family of tasks into one model interface. |
| Attention axes | Time, agents, and road graph | Lets heterogeneous scene information interact. |
| Evidence | Waymo Open Motion Dataset and Argoverse | Tests both large-scale scene prediction and public map-rich forecasting. |

## Decision Lens

Scene Transformer informs whether marginal, joint, conditional, and goal-conditioned forecasting require separate architectures. The atomic unit is an agent-time state token; masks and conditioning inputs alter the forecasting query while the same scene-centric attention stack models interactions.

Unification reduces task-specific machinery, but dense attention over agents and time makes missing-state handling and scene size part of the compute budget. The missing control compares one universal model with specialized models at matched total parameters and training examples, not merely shared implementation. At 10× agents or horizon, the interaction tensor dominates memory. The unification claim would fail if specialized decoders consistently improved joint calibration or rare conditional queries at equal aggregate cost.

**Context:** Scene Transformer pushed the field toward unified heterogeneous scene attention, where the model reasons over agents, time, and map context together.

**Takeaway:** The forecasting question can be a mask over the scene, not a separate model for every task.
