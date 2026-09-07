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

**arXiv:** [2608.00779](https://arxiv.org/abs/2608.00779)

## Summary

> SIPTraj replaces two priors lost with an HD map: agent-specific scene grounding and physical feasibility. Its Hierarchical Agent–Scene Encoder repeatedly reads BEV evidence for each agent, writes local evidence back into the agent representation, and refines relations. Its Physics-Guided Iterative Decoder injects speed, acceleration, and curvature as queries while acceleration, jerk, and curvature losses train the internal pathway. On the standard 2-second-history/6-second-future setup, the map-free model reports mADE5 of 1.1367 m on nuScenes and 0.7438 m on Argoverse 2 Sensor.

## Core Insights

### HASE makes scene context agent-specific before decoding

SIPTraj uses a BEVFusion sensor encoder for camera and LiDAR inputs and a pre-encoder for agent histories. HASE has three stages. Agent-to-scene aggregation uses each agent feature to retrieve deformable BEV evidence. Scene-to-agent grounding and relation refinement then write local scene context back into the agent token and model nearby agents with relation-biased attention. Global scene-context fusion lets the agent consult the full scene memory before decoding.

This sequence matters because a generic BEV summary answers “what is in the scene,” whereas a trajectory predictor needs “which part of that scene constrains this agent.” The representation is repeatedly tied to the target agent's history and local geometry before the model asks for multimodal futures.

![SIPTraj couples agent-conditioned BEV grounding with physics-guided iterative decoding](/assets/images/siptraj-architecture-paper-figure.png)
*Fig 1: The architecture routes sensor BEV features and agent history through HASE, then conditions PGID on a physical descriptor before producing trajectories. | source: [SIPTraj: Map-Free End-to-End Trajectory Prediction via Physics-Guided Scene Interaction, Figure 2](https://arxiv.org/abs/2608.00779)*

The HASE ablation supports this intuition. With only agent-to-scene aggregation, mADE5 is 1.3821 and mFDE1 is 7.9634. Adding scene-to-agent grounding and relation refinement improves them to 1.2754 and 7.4218. Global scene fusion reaches 1.2031 and 7.0852 before PGID is added. The largest early gain comes when local scene evidence is written back into the agent state, rather than when another generic global feature is appended.

### PGID turns kinematics into a representation pathway

Prior map-free predictors can impose acceleration or curvature penalties only on the output trajectory. SIPTraj instead extracts an instantaneous physical descriptor from each agent's speed, acceleration, and curvature, converts it into a learnable physics query, and injects it through physics cross-attention in every iterative decoding layer. Physics-conditioned trajectory tokens can therefore adjust the internal mode content before the final Gaussian-mixture output.

The decoder also predicts dynamic intention points and coordinate corrections, and trains with acceleration, jerk, and curvature consistency losses. The useful distinction is where the gradient travels: with the physics query, consistency loss can reach the internal scene-context representation through the PCAR path. Without the query, it behaves more like an output regularizer.

The PGID ablation is deliberately matched on full HASE. Removing the physics query but keeping losses gives mADE5 1.1842, mFDE1 6.8934, and miss rate 0.2812. Keeping the query but removing physics losses gives 1.1756, 6.7810, and 0.2793. The full combination reaches 1.1367, 6.6205, and 0.2749. Both parts help, and neither row isolates a learned dynamics simulator; the paper measures prediction errors and feasibility proxies.

### The gains survive map-free comparison, with a clear protocol boundary

On nuScenes, SIPTraj reports mADE5/mADE10 of 1.1367/0.8766 m, mFDE1/mFDE10 of 6.6205/2.0079 m, and miss rate 0.2749. Against the other map-free BEVTraj row, the comparison is 1.4556/0.9438, 8.4384/2.0527, and 0.3082. On Argoverse 2 Sensor the corresponding SIPTraj values are 0.7438/0.4635, 4.8421/1.3422, and 0.1654, versus BEVTraj's 0.9820/0.6249, 5.2608/1.5832, and 0.1896.

Both datasets use two seconds of history and six seconds of future. The paper uses approximately 32k/9k nuScenes train/validation samples and extracts 35k/7k Argoverse 2 samples after validity and proximity filtering. The BEVFusion sensor encoder is frozen during trajectory training, and nuScenes history is interpolated from 2 Hz to the 10 Hz Argoverse sampling rate. These details make the cross-dataset comparison informative, but the results are offline trajectory prediction, not closed-loop driving safety.

![Map-based, prior map-free, and SIPTraj designs show where topology and physics enter](/assets/images/siptraj-map-free-end-to-end-trajectory-prediction-via-physics-guided-scene-interaction-source-figure-1.webp)
*Fig 2: The comparison shows HD-map dependence in one family, shallow agent–scene coupling in prior map-free models, and HASE plus physics conditioning in SIPTraj. | source: [SIPTraj: Map-Free End-to-End Trajectory Prediction via Physics-Guided Scene Interaction, Figure 1](https://arxiv.org/abs/2608.00779)*

The method's decision is attractive when map coverage is unreliable and sensor-derived BEV already exists. Its unresolved cost is a larger, more structured decoder and a need for reliable instantaneous kinematics. A physics-aware representation can enforce smoothness while still missing a road rule or reacting incorrectly to a sensor-induced BEV error.

## High-Level Takeaways

- SIPTraj recovers map-like structure through agent-conditioned scene interaction and recovers physics through an internal query path, rather than treating either as a post-processing filter.
- The most informative ablations are matched: local scene grounding supplies the largest HASE step, while physics query and physics losses are complementary.
- The evidence is 2-second-history/6-second-future offline prediction with a frozen sensor encoder; it does not establish robustness to map-domain shift or closed-loop safety.
- A useful deployment test would compare output-only penalties, PGID conditioning, and a dynamics-integrated decoder under identical compute and sensor corruption.
- The paper's core idea is to make the agent's current state participate in how scene evidence is represented before future modes are generated.
