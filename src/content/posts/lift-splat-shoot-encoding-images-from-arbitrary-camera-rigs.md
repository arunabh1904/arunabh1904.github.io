---
title: 'Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D'
date: '2020-08-13T00:00:00.000Z'
section: paper-shorts
postSlug: lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs
legacyPath: /paper shorts/2020/08/13/lift-splat-shoot-encoding-images-from-arbitrary-camera-rigs.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2020 – Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D'
---
## 2020 – Lift, Splat, Shoot

**arXiv:** [2008.05711](https://arxiv.org/abs/2008.05711)

**Project and code:** [nv-tlabs.github.io/lift-splat-shoot](https://nv-tlabs.github.io/lift-splat-shoot/)

### Method and reported result

Lift, Splat, Shoot (LSS) turns an arbitrary camera rig into a metric bird's-eye-view representation. Each image pixel predicts both a context feature and a distribution over depth bins. Their outer product “lifts” the pixel into a camera frustum; known calibration moves the frustum into the vehicle frame; pooling “splats” the features into a BEV grid.

## Summary

> The durable idea is the interface, not the rhyme in the title. Perspective images disagree about where evidence should live, while planning needs a vehicle-centered spatial map. LSS makes the camera-to-BEV transformation differentiable and lets every camera contribute to the same grid.

## Core Insights

LSS separates appearance from depth uncertainty. A pixel does not choose one depth before the rest of the network can reason about it. It distributes its context feature across candidate depths, then geometry places those candidates in the ego frame. The BEV encoder can resolve overlapping or contradictory evidence after all cameras have been pooled.

That factorization explains both the method's influence and its cost. A dense depth distribution preserves alternatives but creates a large frustum tensor. The paper introduces a cumulative-sum pooling trick to reduce aggregation cost; later systems such as BEVFusion make the pooling path much faster. The source paper also tests camera dropout, calibration noise, unseen camera rigs, and a planning probe. It reports that the camera model trails a LiDAR PointPillars baseline, especially at night and at increasing range, so learned lifting does not erase camera depth limits.

![Figure 1 from Lift, Splat, Shoot, showing surround-camera images mapped into a vehicle-centered BEV prediction](/assets/images/lift-splat-shoot-paper-figure-1.png)
_Figure 1 shows the contract LSS establishes: evidence from an arbitrary multi-camera rig lands in one vehicle-centered semantic map. Source: [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711), Figure 1._

| Design object | Choice | Consequence |
| --- | --- | --- |
| Camera token | Pixel context plus categorical depth distribution | Retains depth uncertainty before geometric pooling. |
| Shared space | Dense ego-frame BEV grid | Makes camera count and viewpoint external to downstream heads. |
| Fusion | Sum-pool lifted frustum features | Handles overlap simply, but can blur conflicting evidence. |
| Robustness training | Camera dropout and calibration noise | Improves the corresponding test-time failure mode, with a clean-input tradeoff. |

## High-Level Takeaways

- LSS informs whether the expensive reusable representation should be built before the task heads. Its atomic unit is a pixel-by-depth feature, and its main compression occurs when height and depth hypotheses are pooled into a 2D BEV cell. Camera encoders share weights; calibration, rather than camera identity, tells the model where each feature belongs.
- The missing matched control is a comparison against attention-based lifting at equal image resolution, BEV resolution, memory, and latency. At 10× image resolution or depth bins, the frustum tensor and pooling bandwidth dominate. The LSS design would be rejected when a sparse query mechanism preserves long-range geometry and camera-rig transfer with materially less memory, or when explicit depth supervision is required to meet range-specific safety targets.
- LSS supplied the explicit camera-to-BEV mechanism that BEVDet, BEVDepth, BEVFusion, and many radar-camera systems later reuse or optimize.
- A shared BEV becomes useful only after the model defines how uncertain perspective evidence enters metric space.
