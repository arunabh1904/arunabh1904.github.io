---
title: "SIPTraj: Map-Free End-to-End Trajectory Prediction via Physics-Guided Scene Interaction"
date: '2026-08-01T00:00:00.000Z'
section: paper-shorts
postSlug: siptraj-map-free-end-to-end-trajectory-prediction-via-physics-guided-scene-interaction
legacyPath: /paper shorts/2026/08/01/siptraj-map-free-end-to-end-trajectory-prediction-via-physics-guided-scene-interaction.html
tags:
  - Autonomous Driving
  - Motion Forecasting
  - Map-Free Prediction
field: 'Motion Forecasting & Planning'
summary: "2026 – SIPTraj: Map-Free End-to-End Trajectory Prediction via Physics-Guided Scene Interaction"
---

## 2026 – SIPTraj: Map-Free End-to-End Trajectory Prediction via Physics-Guided Scene Interaction

**arXiv:** [2608.00779](https://arxiv.org/abs/2608.00779)

## Summary

> SIPTraj addresses two priors that map-free trajectory predictors usually lose together: lane-like scene structure and physical feasibility. Its Hierarchical Agent-Scene Encoder grounds each agent progressively in BEV evidence, while its Physics-Guided Iterative Decoder conditions the internal prediction state on instantaneous kinematics. On nuScenes and Argoverse 2 Sensor, the paper reports lower displacement and miss metrics than the compared map-free and map-based baselines, without using an HD map at inference.

## Core Insights

Map-free prediction replaces explicit lane topology with sensor-derived BEV features, but a single scene-fusion step does not tell each agent which local evidence matters. SIPTraj first builds agent tokens from observed histories, then repeatedly performs agent-guided scene retrieval, relation-aware interaction refinement, and late task refinement. The result is a scene representation tied to the predicted agent rather than a generic BEV summary.

The second change moves physical supervision upstream. PGID feeds the agent's instantaneous state into iterative decoding and combines this with acceleration, jerk, and curvature penalties. These constraints therefore shape the features used to generate multimodal hypotheses, rather than only rejecting implausible output trajectories after decoding. The distinction matters because a feasible trajectory is a property of the whole rollout, not just its final points.

![SIPTraj architecture with hierarchical agent-scene grounding and physics-guided iterative decoding](/assets/images/siptraj-architecture-paper-figure.png)
_The paper's overview couples agent-conditioned BEV grounding with physics-aware iterative decoding. Source: [SIPTraj](https://arxiv.org/abs/2608.00779)._

| Dataset | mADE@5 | mADE@10 | mFDE@1 | mFDE@10 | Miss rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| nuScenes | 1.1367 | 0.8766 | 6.6205 | 2.0079 | 0.2749 |
| Argoverse 2 Sensor | 0.7438 | 0.4635 | 4.8421 | 1.3422 | 0.1654 |

The reported qualitative cases show why the two components are paired: HASE keeps long-horizon modes anchored to local scene curvature, while PGID suppresses kinematically implausible modes. The paper does not provide a single matched ablation that isolates the value of internal physics conditioning from the added output penalties across all metrics, so the relative contribution of those two forms of supervision remains an open question.

## High-Level Takeaways

- SIPTraj informs whether a map-free predictor should recover topology through agent-conditioned scene grounding instead of importing a map encoder.
- The training unit is a multimodal future trajectory conditioned on an agent history, local BEV evidence, and instantaneous kinematics; the decoder keeps physical supervision in the representation pathway.
- The reported gains are benchmark results under a 2-second history and 6-second forecast on nuScenes and Argoverse 2 Sensor. They do not establish closed-loop planning safety or robustness to sensor-domain shift.
- The decisive follow-up is a matched ablation of output-only constraints, PGID conditioning, and HASE under identical compute. The conclusion would weaken if output penalties recover the same feasibility and error gains without internal state conditioning.
