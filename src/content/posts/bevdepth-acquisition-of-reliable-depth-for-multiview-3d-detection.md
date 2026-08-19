---
title: 'BEVDepth: Acquisition of Reliable Depth for Multi-View 3D Object Detection'
date: '2022-06-21T00:00:00.000Z'
section: paper-shorts
postSlug: bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection
legacyPath: /paper shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – BEVDepth: Acquisition of Reliable Depth for Multi-View 3D Object Detection'
---
## 2022 – BEVDepth

**arXiv:** [2206.10092](https://arxiv.org/abs/2206.10092)

**Code:** [Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

### Method and reported result

BEVDepth shows that a camera-only detector can use LiDAR as training supervision without requiring LiDAR at inference. It adds explicit sparse depth targets to a Lift-Splat camera-to-BEV transform, conditions the depth network on camera intrinsics, refines the lifted feature volume, and pools it efficiently into BEV.

## Summary

> The key distinction is lifecycle. LiDAR supplies targets while the model learns; runtime inputs remain surround-camera images and calibration. This is privileged sensing, not sensor fusion.

## Core Insights

The paper first audits the learned depth inside Lift-Splat-style detectors. Replacing learned depth with ground-truth depth raises its controlled detector from 0.282 to 0.470 mAP and from 0.327 to 0.515 NDS, showing that the view transform leaves substantial geometry on the table. BEVDepth then supervises the categorical depth distribution using projected LiDAR points, supplies intrinsics and extrinsics to the depth network, and uses a refinement module to reduce errors caused by imperfect unprojection.

The full system adds efficient voxel pooling and temporal fusion. On nuScenes test, the paper reports 60.9 NDS, the first camera-only result above 60 NDS at publication. That number does not mean sparse LiDAR labels solve monocular ambiguity everywhere: supervision is sparse, dynamic-object alignment can be imperfect, and the runtime camera still loses direct range observability in novel conditions.

![Figure 4 from BEVDepth, showing LiDAR depth supervision used during training and a camera-only inference path](/assets/images/bevdepth-paper-figure-4.png)
_The red supervision arrow exists only during training; the deployed path runs from multi-view images through depth prediction and BEV pooling. Source: [BEVDepth](https://arxiv.org/abs/2206.10092), Figure 4._

| Depth mechanism | Input at training | Input at inference | Role |
| --- | --- | --- | --- |
| Depth prediction | Images, calibration, sparse LiDAR targets | Images and calibration | Places context features along camera rays. |
| Camera awareness | Intrinsics and extrinsics | Intrinsics and extrinsics | Adapts depth features to focal length and camera pose. |
| Depth refinement | Lifted image feature volume | Lifted image feature volume | Corrects local errors before voxel pooling. |
| Temporal fusion | Adjacent camera frames and ego motion | Adjacent camera frames and ego motion | Adds motion and multi-view geometric cues. |

## High-Level Takeaways

- BEVDepth informs whether a camera-only production model should spend data-collection budget on LiDAR-equipped teacher vehicles. Its atomic unit is a pixel-by-depth-bin feature. The camera backbone and depth head are shared across views, while calibration conditions the prediction; the depth loss is auxiliary at training but structurally changes the runtime BEV.
- The missing control compares sparse LiDAR supervision, dense offline reconstruction, stereo/video self-supervision, and no depth target under equal camera data and training compute. At 10× fleet data, target generation, calibration quality, and long-range label sparsity become the bottleneck rather than model capacity. The privileged-LiDAR recipe would fail if video-only geometry or future-point-cloud pretraining matches range-bucketed detection, calibration, and uncertainty without a LiDAR collection fleet.
- LSS made depth a latent distribution; BEVDepth showed that explicitly supervising that latent can materially improve the camera-only detector.
- A runtime sensor budget and a training supervision budget are different design variables.
