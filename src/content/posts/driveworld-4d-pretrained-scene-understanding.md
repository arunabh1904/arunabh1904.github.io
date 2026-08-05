---
title: 'DriveWorld: 4D Pre-Trained Scene Understanding'
date: '2024-05-07T04:00:00.000Z'
section: paper-shorts
postSlug: driveworld-4d-pretrained-scene-understanding
legacyPath: /paper shorts/2024/05/07/driveworld-4d-pretrained-scene-understanding.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2024 – DriveWorld: pretrain persistent dynamic and static scene state for many driving tasks'
---
## 2024 – DriveWorld

**arXiv:** [2405.04390](https://arxiv.org/abs/2405.04390)

### Method and reported result

DriveWorld pretrains a 4D scene representation through a Memory State-Space Model. A Dynamic Memory Bank carries changing actors, Static Scene Propagation maintains map-like context, and task prompts adapt the latent to detection, mapping, tracking, occupancy, motion, and planning. Pretraining predicts current and future occupancy and actions.

## Summary

The architecture separates two temporal regimes: static structure can be propagated geometrically, while dynamic state needs learned recurrent updates.

## Core Insights

On the paper's OpenScene pretraining setup, reported downstream gains include 7.5 mAP for detection, 3 IoU for mapping, 5 AMOTA for tracking, 0.1 m lower minADE, 3 IoU for occupancy, and 0.34 m lower planning L2. The pretraining geometry comes from dense occupancy labels generated with fused LiDAR, so the framework uses rich automatic supervision rather than pure camera-only self-supervision.

| State | Update rule | Consumer |
| --- | --- | --- |
| Dynamic memory | Learned state-space recurrence | Actors, tracking, prediction. |
| Static scene | Ego-motion propagation | Maps and persistent layout. |
| Future occupancy/action | Pretraining target | Geometry and consequence. |
| Task prompt | Consumer conditioning | Multi-task transfer. |

## High-Level Takeaways

- DriveWorld is useful when many tasks must reuse a long-lived scene latent. The core ablation should compare one general memory with separate static and dynamic memories at matched capacity, then measure staleness, re-observation, and error propagation over long clips.
- Reported broad transfer is promising but does not establish causal planning benefit. Planning evaluation must include closed-loop behavior, latency, and scenario regressions.
- BEVDet4D carries one previous dense grid; sparse detectors carry instances; DriveWorld makes persistent 4D state itself the pretrained asset.
- Temporal pretraining becomes more reusable when the memory distinguishes what should move, what should remain fixed, and which task is reading it.
