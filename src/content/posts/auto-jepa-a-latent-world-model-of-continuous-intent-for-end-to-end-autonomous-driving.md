---
title: 'Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving'
date: '2026-07-31T09:00:00.000Z'
section: paper-shorts
postSlug: auto-jepa-a-latent-world-model-of-continuous-intent-for-end-to-end-autonomous-driving
legacyPath: /paper shorts/2026/07/31/auto-jepa-a-latent-world-model-of-continuous-intent-for-end-to-end-autonomous-driving.html
tags:
  - Autonomous Driving
  - World Models
  - Planning
field: 'Motion Forecasting & Planning'
topics:
  - autonomy
  - learning
summary: '2026 – Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving'
---

## 2026 – Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving

**arXiv:** [2607.29031](https://arxiv.org/abs/2607.29031)

**Code and models:** [NoctYang/Auto-JEPA](https://github.com/NoctYang/Auto-JEPA)

## Summary

> Dense driving world models predict future video, occupancy, or agent state even though a planner ultimately needs an ego trajectory. Auto-JEPA replaces that reconstruction target with a narrower one: the latent representation of the future ego motion. A frozen V-JEPA 2 encoder processes four front-camera frames; ego-motion history and a route command provide additional context; and a 24-layer Transformer predicts eight latent tokens aligned with the encoded ground-truth trajectory.

## Core Insights

The predicted latent is not decoded directly into waypoints. It retrieves 300 candidates from a fixed memory of 110,335 logged trajectories, after which a scene-conditioned scorer ranks them and a separate drivable-area gate rejects unsafe candidates. On NAVSIM v1, the resulting camera-only planner reaches 91.3 PDMS. Under the updated NAVSIM v2 evaluator it reaches 89.1 EPDMS. These results support action-oriented prediction as a useful planning interface, but they do not show that trajectory intent can replace a full world model for simulation or counterfactual reasoning.

Auto-JEPA first trains a trajectory autoencoder on eight future $(x,y)$ waypoints. Its loss combines coordinate, endpoint, velocity, and acceleration terms. The decoder is then discarded, while the frozen trajectory encoder supplies both the training target and every key in the retrieval memory. This shared encoder makes the predicted intent directly usable as a nearest-neighbor query rather than an intermediate representation that needs another learned generator.

![Auto-JEPA training and inference pipeline, from future-trajectory latent alignment to memory retrieval and gated candidate selection](/assets/images/auto-jepa-architecture.png)
*The predictor learns an eight-token future-motion latent; at inference, that latent retrieves logged trajectories for a scorer and drivable-area gate to select. org/abs/2607.29031). source: [paper](https://arxiv.org/abs/2607.29031)*

![Figure 3 from Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving](/assets/images/auto-jepa-a-latent-world-model-of-continuous-intent-for-end-to-end-autonomous-driving-source-figure-3.webp)
*Figure 3 Overview of Auto-JEPA. During training, the predictor learns a continuous future ego-motion intent by aligning its predicted latent with the representation of the ground-truth future trajectory. During inference, the predicted intent retrieves 300 candidates from a ground-truth-only latent trajectory memory; a scene-conditioned scorer ranks candidate quality and an independent feasibility gate filters drivable-area violations before final selection. Snowflake symbols indicate frozen modules. source: [Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving](https://arxiv.org/abs/2607.29031)*

![Figure 1 from Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving](/assets/images/auto-jepa-a-latent-world-model-of-continuous-intent-for-end-to-end-autonomous-driving-source-figure-1.webp)
*Figure 1 Selective response to action-relevant scene information. Occluding a non-interacting vehicle causes little change in the predicted intent and plan, whereas occluding the interacting lead vehicle shifts the intent and selected trajectory. Annotations are used only for analysis. source: [Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving](https://arxiv.org/abs/2607.29031)*


The predictor is trained with three alignment signals: normalized feature matching, token-wise cosine distance, and batch InfoNCE. The visual encoder and trajectory encoder remain frozen. This supervision contains no object boxes, occupancy maps, semantic maps, or surrounding-agent trajectories; planning relevance must be induced by matching the future ego-motion latent.

| Comparison | Result | What it isolates |
| --- | --- | --- |
| Full system | 91.3 PDMS on NAVSIM v1 | Intent retrieval plus learned ranking and feasibility filtering |
| Fixed medoid instead of predicted intent | 52.6 PDMS | A scene-conditioned retrieval query is load-bearing |
| Intent retrieval + gate, no scorer | 87.6 PDMS | Retrieval supplies strong candidates, but ranking adds 3.7 points |
| Intent retrieval + scorer, no gate | 91.0 PDMS; DAC falls from 98.3 to 97.9 | The gate mainly protects drivable-area compliance |
| Top-1 vs Top-200 vs Top-300 retrieval | 87.6 / 91.1 / 91.3 PDMS | Candidate diversity matters until roughly 200 retrieved trajectories |

The paper also probes whether the intent responds to planning-relevant agents. Across 15,364 validation scenes, masking dynamic-agent regions changes the latent by 0.080 in cosine distance on average, versus 0.027 for equal-area random masks. The dynamic-agent intervention is larger in 71.1% of scenes. Individual occlusions likewise produce larger trajectory shifts for interacting vehicles than for nearby non-interacting vehicles. This is stronger than a qualitative saliency claim, although it remains an intervention on one deterministic checkpoint rather than a causal guarantee of safe behavior.

## High-Level Takeaways

- Auto-JEPA informs whether a driving team should spend model capacity reconstructing the future scene or learn only a planning-oriented predictive state. Its atomic unit is an eight-step future trajectory latent. Visual, route, and ego-history features are fused in the predictor, while executable geometry remains outside the network in a fixed memory. The evidence favors this split when the product needs a strong trajectory proposal under a bounded logged-motion vocabulary and does not need explicit forecasts for other agents.
- The expensive commitment is the memory-and-selector interface. Retrieval avoids a learned trajectory generator, but the planner cannot synthesize a maneuver missing from the memory, and selection still depends on a separately adapted scorer plus feasibility gate. The paper reports one deterministic full benchmark evaluation, so seed variance and scorer calibration under distribution shift remain unknown. At 10× route diversity, memory coverage and nearest-neighbor recall are more likely to fail than latent prediction compute.
- A decisive test would compare three planners under identical visual backbones, training scenes, candidate budgets, and scorer capacity: direct waypoint regression, intent-conditioned retrieval, and intent-conditioned generation or refinement. Evaluation should stratify rare maneuvers and interactive scenes, measure recall before ranking, and include multiple seeds. The retrieval claim should be rejected if a compact generator matches safety and progress while covering maneuvers absent from the logged memory.
- Auto-JEPA narrows the JEPA idea from predicting a general future representation to predicting the part of the future that directly indexes ego action.
- The planner uses only a front camera and a non-reactive NAVSIM evaluation. Its intent does not forecast surrounding agents, its motion space is bounded by a fixed trajectory memory, and the benchmark numbers come from one selected checkpoint rather than independent retraining runs.
- Predicting future ego-motion latents can be enough to retrieve strong plans, but the gain depends on whether a fixed memory actually contains the maneuver the scene requires.
