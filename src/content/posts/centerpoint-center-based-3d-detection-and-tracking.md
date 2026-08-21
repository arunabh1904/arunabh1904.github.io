---
title: 'CenterPoint: Center-Based 3D Detection and Tracking'
date: '2020-06-19T04:00:00.000Z'
section: paper-shorts
postSlug: centerpoint-center-based-3d-detection-and-tracking
legacyPath: /paper shorts/2020/06/19/centerpoint-center-based-3d-detection-and-tracking.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2020 – CenterPoint: represent 3D actors by BEV centers, attributes, and velocity'
---
## 2020 – CenterPoint

**arXiv:** [2006.11275](https://arxiv.org/abs/2006.11275)

**Code:** [tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint)

### Method and reported result

CenterPoint detects objects as center points in BEV, then regresses height, dimensions, orientation, and velocity. An optional refinement stage samples features near predicted box faces. Tracking uses predicted center velocity to associate detections greedily across frames.

## Summary

> The representation removes an awkward transplant from 2D detection: oriented 3D anchors have many sizes and headings, while an object's center remains a stable geometric primitive.

## Core Insights

A class-specific heatmap supplies candidate centers; lightweight attribute heads complete the 3D state. The paper reports 58.0 mAP and 65.5 NDS for detection and 63.8 AMOTA for tracking on nuScenes. The center head adds roughly 3–4 mAP over the compared anchor formulation; refinement adds about 2 mAP with under 10% overhead in the reported setup.

![CenterPoint: Center-Based 3D Detection and Tracking source figure: We present a center-based framework to represent, detect and track objects.](/assets/images/centerpoint-center-based-3d-detection-and-tracking-paper-figure.webp)
_We present a center-based framework to represent, detect and track objects. Source: [CenterPoint: Center-Based 3D Detection and Tracking](https://arxiv.org/abs/2006.11275), Figure 1, via arXiv HTML._


| Output | Representation | Downstream value |
| --- | --- | --- |
| Location | BEV center heatmap | Simple proposal and association unit. |
| Shape | Size, height, rotation | Completes metric box state. |
| Motion | Planar velocity | Links detection to short-term tracking. |
| Refinement | Face-center features | Repairs coarse center-local evidence. |

## High-Level Takeaways

- CenterPoint is useful beyond its backbone because it defines a compact interface between dense BEV features and object-centric temporal state. Fusion systems can combine sensors in BEV, then expose centers and velocity to tracking or prediction without carrying the full grid forever.
- Center-only representations can under-use boundary evidence for large or articulated objects. Evaluate localization, velocity, crowded-scene identity switches, and delayed measurements separately.
- BEVFusion and many LiDAR-camera systems adopt CenterPoint-style heads; Sparse4D later makes object instances themselves the recurrent state.
- A center is an economical bridge from dense geometry to sparse actor state, but it must be paired with explicit shape, motion, and uncertainty.
