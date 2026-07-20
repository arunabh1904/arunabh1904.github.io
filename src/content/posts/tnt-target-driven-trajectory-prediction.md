---
title: 'TNT: Target-driveN Trajectory Prediction'
date: '2020-08-19T04:00:00.000Z'
section: paper-shorts
postSlug: tnt-target-driven-trajectory-prediction
legacyPath: /paper shorts/2020/08/19/tnt-target-driven-trajectory-prediction.html
tags:
  - Other
field: 'Motion Forecasting & Planning'
summary: "2020 – TNT: Target-driveN Trajectory Prediction"
---
## 2020 – TNT: Target-driveN Trajectory Prediction

**arXiv:** [2008.08294](https://arxiv.org/abs/2008.08294)

**PMLR:** [CoRL 2020 proceedings](https://proceedings.mlr.press/v155/zhao21b.html)

**Project:** [Waymo research page](https://waymo.com/research/tnt-target-driven-trajectory-prediction/)

**Summary:** TNT says that for moderate-horizon motion forecasting, most of the multimodality lives in where the agent is trying to end up. Instead of sampling latent variables and hoping they cover the futures, TNT predicts explicit target states, then predicts trajectories conditioned on each target.

That design makes the intermediate outputs interpretable. A planner can inspect possible destinations, target-conditioned rollouts, and trajectory scores instead of receiving only opaque samples from a latent distribution.

## Paper Insights

The paper decomposes trajectory prediction into target uncertainty and control uncertainty. The target predictor estimates a distribution over candidate endpoints, using lane-centerline samples for vehicles and grid samples for pedestrians. Given each selected target, a motion estimator predicts one trajectory toward it. A final scoring and selection stage ranks the hypotheses and suppresses near-duplicates to produce a small set of trajectories.

TNT uses VectorNet as the HD-map context encoder when maps are available and a ResNet image encoder for the Stanford Drone Dataset. The important modeling assumption is that once the target is fixed, the remaining trajectory distribution is close enough to unimodal for a simple regression head. That assumption is reasonable for short and moderate horizons, but the paper itself notes that longer horizons may need intermediate targets.

![Figure 2 from TNT showing context encoding, target prediction, target-conditioned motion estimation, and trajectory scoring](/assets/images/tnt-target-driven-trajectory-prediction-paper-figure.png)
_Figure 2 shows the three-stage TNT pipeline: encode the scene, score candidate target states, decode a trajectory for each selected target, then score and select a compact set. From the [TNT paper](https://arxiv.org/abs/2008.08294), via the arXiv PDF._

**What to look at:**
- Targets make intent interpretable: turning, lane changes, and speed choices become endpoint hypotheses.
- The framework avoids test-time latent sampling by producing diverse futures in parallel.
- Target candidates can come from map structure for vehicles or a grid for pedestrians.
- The scoring stage matters because good endpoints do not automatically imply good full trajectories.

**Evals / Benchmarks / Artifacts:**

| Component | Detail | Why it matters |
| --------- | ------ | -------------- |
| Context encoder | VectorNet for vectorized HD maps; ResNet for image-only SDD | Keeps TNT compatible with both map-rich and image-only settings. |
| Target prediction | Samples many candidate endpoints, then keeps top targets | Moves multimodality into an explicit endpoint space. |
| Motion estimation | Regresses one trajectory conditioned on each selected target | Treats target-conditioned control as mostly unimodal. |
| Scoring and selection | Ranks trajectories and removes near-duplicates | Produces a small deployable set of diverse forecasts. |

**Compact result slice:**

| Dataset | Comparison | TNT result |
| ------- | ---------- | ---------- |
| Argoverse validation | MultiPath: 1.68 minFDE, 0.80 minADE, 0.14 MR | 1.29 minFDE, 0.73 minADE, 0.09 MR |
| INTERACTION validation | MultiPath: 0.99 minFDE, 0.30 minADE | 0.67 minFDE, 0.21 minADE |
| PAID pedestrians | MultiPath: 0.43 minFDE, 0.23 minADE | 0.32 minFDE, 0.18 minADE |
| Stanford Drone | PECNet: 25.98 minFDE, 12.79 minADE | 21.16 minFDE, 12.23 minADE |

## Decision Lens

TNT informs whether multimodal forecasting should first choose a destination and then generate the path, rather than regress complete trajectories in one step. The atomic hierarchy is an actor history, a candidate target state, and a target-conditioned trajectory; a final scorer selects a compact diverse set.

The factorization gives modes a semantic endpoint, but target discretization and candidate pruning can exclude valid futures before decoding. The missing ablation holds total proposals fixed while comparing endpoint-first, anchor-trajectory, and direct set prediction across map-rich and map-free datasets. At 10× candidate density, target scoring dominates and duplicates crowd out rare modes. TNT's claim would fail if direct trajectory-set prediction matched miss rate and diversity without a target bottleneck.

**Context:** TNT made goal-conditioned motion forecasting feel practical for autonomous driving. It kept the multimodal structure visible and showed that endpoint candidates can be a cleaner intent representation than opaque latent samples.

**Takeaway:** Predict the destination first, then make the trajectory explain how to get there.
