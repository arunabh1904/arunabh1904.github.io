---
title: 'PerceptDrive: Perception Prior World-Action Modeling with Adaptive Expert Routing for End-to-End Autonomous Driving'
date: '2026-07-22T00:00:00.000Z'
section: paper-shorts
postSlug: perceptdrive-perception-prior-world-action-modeling-with-adaptive-expert-routing-for-end-to-end-autonomous-driving
legacyPath: /paper shorts/2026/07/24/perceptdrive-perception-prior-world-action-modeling-with-adaptive-expert-routing-for-end-to-end-autonomous-driving.html
tags:
  - Autonomous Driving
  - World Models
  - Expert Routing
field: 'Autonomous Driving: VLA & Planning'
topics:
  - autonomy
  - multimodal
  - learning
summary: '2026 – PerceptDrive: Perception Prior World-Action Modeling with Adaptive Expert Routing for End-to-End Autonomous Driving'
---

## 2026 – PerceptDrive: Perception Prior World-Action Modeling with Adaptive Expert Routing for End-to-End Autonomous Driving

**arXiv:** [2607.20175](https://arxiv.org/abs/2607.20175)

## Summary

PerceptDrive treats perception-to-planning transfer as an information-bottleneck problem. A large frozen model may encode geometry, semantics, and dynamics, but a small set of planner queries can compress those signals into redundant features. The method creates separate query branches for three perception priors, anchors each compressed branch to its designated prior, and learns scene-dependent soft weights before a flow-matching actor generates one trajectory.

## Core Insights

The full system reports 90.4 PDMS on NAVSIM v1 and 90.2 EPDMS on NAVSIM v2. More useful than the cross-paper ranking is its nested three-seed ablation: future conditioning, metric supervision, prior retention, and metric-distilled routing raise EPDMS monotonically from 84.6 to 90.2. The result also exposes an important qualification: privileged NAVSIM sub-metrics supervise the planner and router during training, so part of the gain is evaluator knowledge amortized into feed-forward inference.

![PerceptDrive architecture with distilled geometry semantic and dynamics priors routed into a shared world-action model](/assets/images/perceptdrive-perception-prior-world-action-modeling-with-adaptive-expert-routing-for-end-to-end-autonomous-driving-paper-figure.png)
_Figure 2 traces the full information path: specialist teachers produce retained priors, query banks compress them, and a scene-conditioned router controls their contribution to one future-conditioned trajectory. Source: [PerceptDrive](https://arxiv.org/abs/2607.20175)._

The frozen perception provider begins as a driving-adapted VLM and distills geometric, semantic, and dynamic knowledge into explicit expert slots. A frozen V-JEPA 2-L video encoder supplies a separate dense observation stream and future-latent targets. During world-action training, learnable query banks read the combined pool: three prior-specific banks, plus global, action, and temporal banks.

Per-branch retention losses prevent all three expert readouts from collapsing into the same imitation features. Each expert may attend to the full perception pool, but its compressed readout must remain predictive of the corresponding prior. A two-layer router then produces a dense simplex over the three conditions. All experts stay active; “expert routing” here means scene-conditioned soft fusion, not sparse top-$k$ execution.

The planner predicts an action-free future V-JEPA latent and uses it to condition a flow-matching actor. A second action-conditioned future branch is auxiliary. During training, an offline pool of trajectories scored with privileged rule-based sub-metrics supervises both a quality regressor and the router. At inference those scorers, branch drafts, and privileged signals disappear: the model predicts one future condition, one set of gates, and one trajectory without candidate search or reranking.

| Nested NAVSIM ablation | PDMS v1 | EPDMS v2 | Incremental mechanism |
| --- | ---: | ---: | --- |
| Imitation only | 86.9 ± 0.2 | 84.6 ± 0.3 | Base world-action model |
| + future conditioning | 87.5 ± 0.2 | 85.4 ± 0.1 | Predict an action-free future latent |
| + metric supervision | 88.6 ± 0.1 | 87.1 ± 0.2 | Distill rule-based driving quality |
| + per-branch prior retention | 89.8 ± 0.2 | 88.7 ± 0.2 | Preserve distinct compressed priors |
| + metric-distilled routing | 90.4 ± 0.1 | 90.2 ± 0.1 | Scene-conditioned expert weighting |

Routing diagnostics are unusually specific. Geometry receives more weight for turns, while dynamics dominates straight cruising. A command-only predictor explains 38% of gate variance, and the within-command gate standard deviation averages 0.11, suggesting that scene content contributes beyond the navigation command. A static learned weighting performs like uniform fusion; an end-to-end router without metric distillation stays near-uniform. The routing gain therefore comes largely from privileged quality supervision, not merely adding a gate.

The model reaches 34.5 EPDMS on NAVSIM v2 navhard, 3.9 points above the strongest listed baseline, but all methods lose substantial lane-keeping and extended-comfort performance in the second pseudo-simulation stage. Training costs 409.6 MI308X GPU-hours. Inference keeps 2.82B active parameters; reducing the flow solver from 25 to 10 Euler steps changes PDMS from 90.4 to 90.3 and latency from 68 to 53 ms on one MI308X.

## High-Level Takeaways

- PerceptDrive informs how a planner should consume multiple frozen perception models. Concatenating features assumes that imitation learning will preserve the right information and assign the right scene-specific weights. This paper instead makes preservation and weighting explicit: branch targets maintain specialization, and a distilled evaluator teaches when each prior should matter.
- The design is attractive when a trusted offline evaluator exists, but that dependency defines the deployment risk. NAVSIM’s scoring rules determine the quality vectors used to train the branches and gates. The paper explicitly notes that transfer to differently weighted objectives is untested. A decisive experiment would train with one scorer, evaluate under held-out metric weights and closed-loop interventions, and test whether routing remains useful rather than overfitting evaluator structure.
- At ten times the prior count, dense soft fusion and target engineering will become expensive. Every expert remains active, and each requires a meaningful retention target. The likely scaling bottleneck is not router capacity but defining reliable, non-conflicting priors and quality signals. Sparse routing is only justified after demonstrating that dropping experts preserves safety across rare scenes.
- PerceptDrive separates frozen geometry, semantics, and dynamics into retained planner branches, then uses training-time metric distillation to fuse them per scene.
- All evaluations use NAVSIM’s non-reactive protocols; no closed-loop vehicle response is tested. The deterministic future head extrapolates beyond demonstrated actions, and scorer transfer to other driving objectives remains unknown.
- Preserve each perception prior through the planner bottleneck and supervise routing explicitly—but treat evaluator distillation as a core dependency, not free generalization.
