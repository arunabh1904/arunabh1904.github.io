---
title: 'UniLION: Towards a Unified Autonomous Driving Model with Linear Group RNNs'
date: '2025-11-03T00:00:00.000Z'
section: paper-shorts
postSlug: unilion-towards-unified-autonomous-driving-model-with-linear-group-rnns
legacyPath: /paper shorts/2025/11/03/unilion-towards-unified-autonomous-driving-model-with-linear-group-rnns.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2025 – UniLION: one linear-RNN backbone across sensors, time, and driving tasks'
---
## 2025 – UniLION

**arXiv:** [2511.01768](https://arxiv.org/abs/2511.01768)

**Code:** [happinesslz/UniLION](https://github.com/happinesslz/UniLION)

## Summary

> UniLION's important result is conditional reuse, not just a long task list. A model trained with temporal LiDAR and cameras can run without history or cameras at inference, using the same backbone and heads, although removing both inputs still produces a measurable accuracy penalty.

UniLION converts LiDAR, multi-view images, and temporal observations into sparse 3D voxels, processes them with grouped linear RNNs, and serves detection, tracking, map segmentation, occupancy, motion prediction, and planning from a shared BEV feature. All results are reported on nuScenes or Occ3D-nuScenes; the paper does not report a matched end-to-end latency or memory comparison that establishes the system-level advantage of linear rather than quadratic sequence mixing.

## Core Insights

LiDAR and camera observations retain separate input encoders, but their voxels enter the same 3D backbone. Each UniLION block partitions sparse features along spatial axes, applies a linear group RNN for long-range interaction, and alternates voxel merging with voxel expansion. A learned 3D spatial descriptor restores positional structure that a linear recurrence alone models weakly. This replaces explicit camera-LiDAR and temporal fusion modules with one recurrent spatial operator; it does not eliminate calibration, view lifting, or modality-specific preprocessing.

![UniLION: Towards a Unified Autonomous Driving Model with Linear Group RNNs source figure: (a) presents the mainstream methods in implementing multi-modal fusion or temporal fusion.](/assets/images/unilion-towards-unified-autonomous-driving-model-with-linear-group-rnns-paper-figure.webp)
_(a) presents the mainstream methods in implementing multi-modal fusion or temporal fusion. Source: [UniLION: Towards a Unified Autonomous Driving Model with Linear Group RNNs](https://arxiv.org/abs/2511.01768), Fig. 1, via arXiv HTML._


The strongest unification test trains one LiDAR-camera-temporal model and removes inputs only at inference, without masked-modality training. Removing temporal history while retaining LiDAR and cameras is nearly neutral: 74.9 NDS and 50.7 RayIoU versus 75.4 and 51.3 with full temporal input. Removing both cameras and history drops the same model to 70.6 NDS and 43.4 RayIoU. A separately trained LiDAR-only model reaches 72.3 NDS and 46.8 RayIoU, so architectural compatibility does not erase the value of matching the training distribution to the deployed sensor set.

| Inference configuration | NDS | Tracking AMOTA | Map mIoU | Occupancy RayIoU |
| --- | ---: | ---: | ---: | ---: |
| LiDAR only, trained LiDAR only | 72.3 | 72.6 | 71.7 | 46.8 |
| LiDAR only, trained with LiDAR + camera + time | 70.6 | 70.2 | 68.6 | 43.4 |
| LiDAR + camera, trained with LiDAR + camera + time | 74.9 | 76.2 | 72.2 | 50.7 |
| LiDAR + camera + time | 75.4 | 76.5 | 73.3 | 51.3 |

The multi-task ablations expose a second trade-off. Adding detection and map segmentation together improves map mIoU from 68.3 to 71.7; adding occupancy then improves RayIoU by 2.7 points but slightly reduces detection. Dynamic loss balancing helps detection, tracking, and mapping while slightly degrading occupancy. One backbone can share representation and compute, but it does not make task gradients automatically compatible.

The training contract is also less monolithic than the architecture diagram suggests. Perception is trained in stages, temporal variants are initialized from single-frame perception, and motion and planning are trained after freezing the temporal perception model. UniLION therefore demonstrates a reusable architecture and representation, not one simultaneous end-to-end optimization over every task and sensor condition.

## High-Level Takeaways

- UniLION makes sensor, temporal, and task unification a backbone decision: sparse voxels from different sources pass through one grouped recurrent operator and one shared BEV state.
- Training with the richest sensor set gives a useful fallback path, but the LiDAR-only fallback remains weaker than a LiDAR-specialized model; graceful degradation still needs an explicit acceptance threshold.
- Joint training helps some tasks and harms others, so loss balancing is a deployment decision rather than bookkeeping.
- The evidence is confined to nuScenes-family benchmarks and does not establish the claimed efficiency advantage under matched hardware, latency, memory, calibration error, and sensor-failure tests.
