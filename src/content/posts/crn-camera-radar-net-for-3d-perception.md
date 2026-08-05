---
title: 'CRN: Camera Radar Net for 3D Perception'
date: '2023-04-03T04:00:00.000Z'
section: paper-shorts
postSlug: crn-camera-radar-net-for-3d-perception
legacyPath: /paper shorts/2023/04/03/crn-camera-radar-net-for-3d-perception.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2023 – CRN: let radar guide camera lifting before BEV fusion'
---
## 2023 – CRN

**arXiv:** [2304.00670](https://arxiv.org/abs/2304.00670)

**Code:** [youngskkim/CRN](https://github.com/youngskkim/CRN)

### Method and reported result

Camera Radar Net uses radar twice: Radar-Assisted View Transformation supplies range evidence while camera features are lifted into BEV, and Multimodal Feature Aggregation uses deformable attention to reconcile residual camera-radar misalignment. Radar therefore affects geometry before the shared BEV encoder can discard its distinctive range and velocity signal.

## Summary

This is different from rasterizing radar as another occupancy channel. The sparse measurements control where dense image evidence should enter metric space.

## Core Insights

Radar occupancy augments the camera depth distribution, while a top-k BEV-query scheme limits deformable-attention cost. The paper reports a 20 FPS real-time configuration and an offline model at 57.5 mAP and 62.4 NDS on nuScenes test. At 256×704 with a ResNet-50, its reported per-class mean is 49.0 mAP versus 34.8 for the cited BEVDepth configuration.

| Mechanism | Reported evidence | Tradeoff |
| --- | --- | --- |
| Radar-guided lifting | Large camera-baseline gain | Depends on radar-camera calibration. |
| Deformable BEV fusion | Repairs spatial mismatch | Query sampling can miss weak regions. |
| 4,096 top-k queries | 21.01 to 4.96 ms fusion latency | Some true-positive metrics degrade. |
| Temporal frames | Improves detection | Adds alignment and memory cost. |

## High-Level Takeaways

CRN is appropriate when radar's main job is to resolve camera depth rather than act as an independent detector. Its failure audit should cover multipath, ghost returns, radial-velocity ambiguity, weather, timestamp offsets, and radar dropout—not only aggregate nuScenes scores.

Top-k query count is a safety-relevant budget because it directly controls which BEV regions can receive cross-modal correction.

RCBEVDet strengthens the radar representation before BEV fusion; CRN makes radar a guide for camera geometry.

Radar is most valuable when its range and motion evidence changes how camera features are localized, not when it is reduced to a generic feature map.
