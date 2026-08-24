---
title: "Not All History Helps: Velocity-Aware Selective Memory for Long-Horizon End-to-End Autonomous Driving"
date: '2026-08-16T00:00:00.000Z'
section: paper-shorts
postSlug: not-all-history-helps-velocity-aware-selective-memory-for-long-horizon-end-to-end-autonomous-driving
legacyPath: /paper shorts/2026/08/16/not-all-history-helps-velocity-aware-selective-memory-for-long-horizon-end-to-end-autonomous-driving.html
tags:
  - Autonomous Driving
  - Planning
  - Temporal Memory
field: 'Autonomous Driving: VLA & Planning'
summary: "2026 – Not All History Helps: Velocity-Aware Selective Memory for Long-Horizon End-to-End Autonomous Driving"
---

## 2026 – Not All History Helps: Velocity-Aware Selective Memory for Long-Horizon End-to-End Autonomous Driving

**arXiv:** [2608.15573](https://arxiv.org/abs/2608.15573)

## Summary

> StableDrive argues that self-generated planning history is useful only when it matches the current motion stage. Its Selective Momentum Memory uses a Mamba state-space operator to gate the previous plan, while a Motion-Stage Training Scaffold teaches stage-aware long-horizon behavior and is removed before inference. A fixed midpoint between two architecture-aligned endpoints gives one deployable planner without an ensemble or extra serving cost.

## Core Insights

Long-horizon planners often feed their own previous predictions back as context. That creates a stale-prior failure: a plan from a different velocity or interaction stage can pull the next prediction away from the current scene. StableDrive makes memory conditional and uses motion-stage, future-trajectory, and longitudinal-motion supervision to teach the policy when history should matter.

The displayed ablation separates the pieces. An endpoint SMM variant has average L2 2.00 and collision 1.24%; adding the training scaffold gives L2 1.89 but collision 1.35%; the fixed midpoint reaches L2 1.83, collision 1.19%, and TPC 1.20. The paper also reports 23.3% lower average collision rate, 30.9% lower TPC, and 11.8% lower L2 relative to the best prior values in its nuScenes comparison.

![StableDrive framework with selective cross-cycle memory and motion-stage training scaffold](/assets/images/stabledrive-framework-paper-figure.png)
*The scaffold supervises stage-aware planning during training, while selective memory remains in the deployed planner. source: [StableDrive](https://arxiv.org/abs/2608.15573)*

![Figure 2 from Not All History Helps: Velocity-Aware Selective Memory for Long-Horizon End-to-End Autonomous Driving](/assets/images/not-all-history-helps-velocity-aware-selective-memory-for-long-horizon-end-to-end-autonomous-driving-source-figure-2.webp)
*Figure 2 Fig. 2: Overall framework of StableDrive. StableDrive combines selective cross-cycle planning memory with a train-and-retire motion-stage scaffold. Multi-view images are encoded into a sparse scene representation, and joint scene interaction generates command-conditioned planning queries. SMM selectively updates the current queries using score-modulated one-cycle history before candidate generation and selection. MSTS provides horizon-wise motion-stage supervision during training and is removed at inference. source: [Not All History Helps: Velocity-Aware Selective Memory for Long-Horizon End-to-End Autonomous Driving](https://arxiv.org/abs/2608.15573)*

![Figure 1 from Not All History Helps: Velocity-Aware Selective Memory for Long-Horizon End-to-End Autonomous Driving](/assets/images/not-all-history-helps-velocity-aware-selective-memory-for-long-horizon-end-to-end-autonomous-driving-source-figure-1.webp)
*Figure 1 Fig. 1: Selective planning memory enables safer long-horizon planning. (a) A nuScenes example comparing MomAD [ 45 ] and StableDrive with the ground-truth future trajectory. (b) Retaining a stale planning prior causes MomAD [ 45 ] to diverge after and collide at . (c) StableDrive suppresses unreliable history and remains collision-free. (d) StableDrive improves six-second L2, Col. Rate, and TPC by 5.83%, 13.11%, and 11.34%, respectively, over our local MomAD [ 45 ] reproduction under the same evaluation protocol. source: [Not All History Helps: Velocity-Aware Selective Memory for Long-Horizon End-to-End Autonomous Driving](https://arxiv.org/abs/2608.15573)*


## High-Level Takeaways

- StableDrive informs whether a planner should retain all history or gate it by motion stage and current scene evidence.
- The atomic unit is a planning query plus a self-generated prior; training adds long-horizon and stage labels that are retired before inference.
- The midpoint choice avoids an ensemble, but it encodes a fixed compromise between memory endpoints rather than learning a full uncertainty distribution.
- The conclusion would weaken under longer unseen routes, reactive agents, and ablations that compare selective memory with a tuned recurrent baseline at equal parameters.
