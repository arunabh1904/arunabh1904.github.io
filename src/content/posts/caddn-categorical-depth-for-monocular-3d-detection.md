---
title: 'CaDDN: Categorical Depth for Monocular 3D Detection'
date: '2021-03-01T05:00:00.000Z'
section: paper-shorts
postSlug: caddn-categorical-depth-for-monocular-3d-detection
legacyPath: /paper shorts/2021/03/01/caddn-categorical-depth-for-monocular-3d-detection.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2021 – CaDDN: supervise a depth distribution that lifts image features into 3D'
---
## 2021 – CaDDN

**arXiv:** [2103.01100](https://arxiv.org/abs/2103.01100)

**Code:** [TRAILab/CaDDN](https://github.com/TRAILab/CaDDN)

### Method and reported result

CaDDN predicts a categorical depth distribution for every image feature, takes an outer product between that distribution and semantic image features to build a frustum volume, then transforms the volume into voxels and BEV for 3D detection. Projected LiDAR supplies depth labels during training; inference uses one camera.

## Summary

> The central idea is to preserve depth uncertainty through lifting instead of committing every pixel to one regressed range.

## Core Insights

Depth bins turn a 2D feature into a probability-weighted ray. The detector loss and explicit depth loss train the same view transformation, connecting image semantics to metric geometry. On KITTI, the paper reports car BEV AP gains of 2.91, 1.59, and 2.22 points for easy, moderate, and hard splits over its cited prior baseline.

![CaDDN architecture lifting image features through a categorical depth distribution into a frustum and then a voxel grid](/assets/images/caddn-categorical-depth-for-monocular-3d-detection-paper-figure.webp)
*The categorical depth distribution keeps several ranges alive while image features are lifted into 3D, after which the frustum is resampled into a voxel grid for detection. source: [CaDDN](https://arxiv.org/abs/2103.01100)*

![Figure 2 from CaDDN: Categorical Depth for Monocular 3D Detection](/assets/images/caddn-categorical-depth-for-monocular-3d-detection-source-figure-2.webp)
*Figure 2 CaDDN Architecture. The network is composed of three modules to generate 3D feature representations and one to perform 3D detection. Frustum features are generated from an image using estimated depth distributions , which are transformed into voxel features . The voxel features are collapsed to bird’s-eye-view features to be used for 3D object detection. source: [CaDDN: Categorical Depth for Monocular 3D Detection](https://arxiv.org/abs/2103.01100)*

![Figure 1 from CaDDN: Categorical Depth for Monocular 3D Detection](/assets/images/caddn-categorical-depth-for-monocular-3d-detection-source-figure-1.webp)
*Figure 1 (a) Input image. (b) Without depth distribution supervision, BEV features from CaDDN suffer from smearing effects. (c) Depth distribution supervision encourages BEV features from CaDDN to encode meaningful depth confidence, in which objects can be accurately detected. source: [CaDDN: Categorical Depth for Monocular 3D Detection](https://arxiv.org/abs/2103.01100)*


| Choice | Benefit | Limit |
| --- | --- | --- |
| Categorical depth | Retains multi-bin uncertainty | Memory grows with bin count. |
| LiDAR supervision | Direct metric learning signal | Sparse and potentially misaligned targets. |
| Frustum outer product | Couples semantics and range | Produces a large intermediate volume. |
| Camera-only inference | Cheap deployed sensor set | Appearance cannot resolve every ambiguity. |

## High-Level Takeaways

- CaDDN matters when the runtime contract excludes LiDAR but an instrumented training fleet can provide it. The supervision pipeline should track occlusion, time alignment, ignore regions, and confidence; sparse projected points are not dense ground truth.
- Compare categorical, continuous, and query-based lifting at matched resolution and latency. Better depth metrics do not guarantee better 3D detection if errors fall in task-irrelevant regions.
- LSS popularizes distributional lifting; CaDDN adds direct depth supervision, and BEVDepth adapts that lesson to surround-camera BEV detection.
- A camera-to-BEV transform becomes trainable and auditable when depth uncertainty is explicit and privileged LiDAR is confined to the training target.
