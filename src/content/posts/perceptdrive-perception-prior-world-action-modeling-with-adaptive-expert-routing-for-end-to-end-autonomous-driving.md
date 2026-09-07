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

> PerceptDrive turns frozen perception and prediction models into explicit interfaces for a driving policy. It distills geometry, semantic, and dynamics priors into separate query banks, retains them with branch-specific losses, and uses a scene-conditioned soft router to mix all branches before a flow-based action head. The full model reports 90.4 +/- 0.05 PDMS on NAVSIM v1 navtest, 90.2 +/- 0.11 EPDMS on navtest v2, and 34.5 EPDMS on navhard v2. The benchmark is non-reactive and the privileged trajectory pool is used offline, so the result does not yet measure closed-loop interaction.

## Core Insights

### Turn frozen specialists into retained interfaces

PerceptDrive begins with a practical compromise. A single VLM can learn to drive, but its compressed representation may discard geometry, motion, or semantic detail that a specialist already encodes. Stage 1 adapts InternVL3 to driving questions. Stage 1b then distills three frozen providers into separate geometry, semantic, and dynamics priors while replaying the language data. Stage 2 feeds those priors, ego state, and command into a shared world-action model.

The separation is architectural and supervisory. Each prior gets its own query bank, while global, action, and temporal banks provide shared context. Retention losses keep a branch from collapsing into the easiest signal during joint training. The model then predicts an action-free future latent and an action-conditioned future alongside the final trajectory, so the world representation has to say what the scene may do before it is asked to choose a maneuver.

![PerceptDrive world-action architecture with specialist priors and adaptive routing](/assets/images/perceptdrive-perception-prior-world-action-modeling-with-adaptive-expert-routing-for-end-to-end-autonomous-driving-paper-figure.png)
*Fig 1: The source overview traces frozen expert providers into separate retained query banks, a routed shared model, future prediction, and one trajectory output. | source: [PerceptDrive, Figure 2](https://arxiv.org/abs/2607.20175)*

The useful intuition is that the query banks are contracts. Geometry can preserve where things are, dynamics can preserve how they move, and semantics can preserve what they mean. The router is allowed to use all three, but the branch losses prevent the shared backbone from making them indistinguishable before that choice happens.

### Routing is trained against action quality

The router is a dense soft simplex: every expert remains active, with scene-dependent weights, rather than a sparse top-k switch. Its supervision comes from an offline privileged trajectory pool. A rule-based scorer evaluates candidate trajectories with NAVSIM-style submetrics, and a quality regressor teaches the router which prior mixture tends to support better actions. At inference, the shared model runs once; the flow action head then integrates its solver steps to produce one trajectory. The model does not generate candidates for scoring or reranking.

![Comparison of direct, fused, and routed expert-prior paradigms](/assets/images/perceptdrive-perception-prior-world-action-modeling-with-adaptive-expert-routing-for-end-to-end-autonomous-driving-source-figure-1.webp)
*Fig 2: The source comparison contrasts implicit VLM conditioning, unseparated expert fusion, and PerceptDrive's separated priors with scene-conditioned routing. | source: [PerceptDrive, Figure 1](https://arxiv.org/abs/2607.20175)*

That comparison clarifies what the router buys. Static fusion assumes the same prior mixture is useful for every scene; direct VLM conditioning leaves the mixture implicit. PerceptDrive makes the choice observable and trainable while keeping the runtime graph fixed. The cost is a second source of supervision: the policy depends on a quality target derived from privileged trajectories that will not be available on the road.

### The matched ablations support the routing story

The nested ablation on NAVSIM navtest adds one capability at a time. Imitation alone reaches 86.9 +/- 0.2 PDMS and 84.6 +/- 0.3 EPDMS. Adding future prediction gives 87.5 +/- 0.2 and 85.4 +/- 0.1. Metric supervision reaches 88.6 +/- 0.1 and 87.1 +/- 0.2; retention losses reach 89.8 +/- 0.2 and 88.7 +/- 0.2; metric routing reaches 90.4 +/- 0.05 and 90.2 +/- 0.11. The gains do not come from adding a larger decoder alone. Each step targets a distinct failure mode in the representation or decision.

The routing diagnostics make the mechanism less abstract. Geometry receives more weight on turns, dynamics matters more on straighter motion, and command-only features explain 38% of gate variance. Within a command, the gate standard deviation is 0.11; static weighting is close to uniform, while an end-to-end router without metric distillation also becomes nearly uniform. The offline action-quality signal is therefore what keeps the router scene-sensitive.

The paper reports 2.82B active parameters and 409.6 MI308X GPU-hours for training. Reducing the flow solver from 25 to 10 Euler steps changes PDMS from 90.4 to 90.3 while reducing latency from 68 ms to 53 ms on one MI308X. The qualitative examples show the resulting trajectory over front-view and BEV context, but they should be read as illustrations of the selected path, not as a substitute for the aggregate ablation.

![PerceptDrive qualitative trajectories with front-view and BEV context](/assets/images/perceptdrive-perception-prior-world-action-modeling-with-adaptive-expert-routing-for-end-to-end-autonomous-driving-source-figure-3.webp)
*Fig 3: The source qualitative examples pair the camera observation and BEV context with the generated trajectory on NAVSIM navtest. | source: [PerceptDrive, Figure 3](https://arxiv.org/abs/2607.20175)*

The evaluation boundary matters. NAVSIM is non-reactive, so other agents do not respond to the ego plan. PerceptDrive's 34.5 EPDMS on navhard v2 shows a harder distribution, but it still does not establish recovery from policy-induced state changes. The next experiment should carry the routed representation and its one-pass constraint into a reactive benchmark while measuring how much privileged-data construction contributes.

## High-Level Takeaways

- PerceptDrive’s representational bet is a retained query bank for each frozen geometry, semantic, or dynamics prior, plus shared action and temporal queries.
- Metric routing matters because it turns privileged trajectory quality into a scene-conditioned mixture without candidate search at inference.
- The ablation's five-step progression separates future prediction, metric supervision, retention, and routing instead of bundling them into one gain.
- The 90.2 EPDMS result is tied to NAVSIM’s scorer and privileged distillation; whether that training dependence transfers to another objective or reactive traffic remains untested.
