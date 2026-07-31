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

**Summary:** CaDDN predicts a categorical depth distribution for every image feature, takes an outer product between that distribution and semantic image features to build a frustum volume, then transforms the volume into voxels and BEV for 3D detection. Projected LiDAR supplies depth labels during training; inference uses one camera.

The central idea is to preserve depth uncertainty through lifting instead of committing every pixel to one regressed range.

## Paper Insights

Depth bins turn a 2D feature into a probability-weighted ray. The detector loss and explicit depth loss train the same view transformation, connecting image semantics to metric geometry. On KITTI, the paper reports car BEV AP gains of 2.91, 1.59, and 2.22 points for easy, moderate, and hard splits over its cited prior baseline.

| Choice | Benefit | Limit |
| --- | --- | --- |
| Categorical depth | Retains multi-bin uncertainty | Memory grows with bin count. |
| LiDAR supervision | Direct metric learning signal | Sparse and potentially misaligned targets. |
| Frustum outer product | Couples semantics and range | Produces a large intermediate volume. |
| Camera-only inference | Cheap deployed sensor set | Appearance cannot resolve every ambiguity. |

## Decision Lens

CaDDN matters when the runtime contract excludes LiDAR but an instrumented training fleet can provide it. The supervision pipeline should track occlusion, time alignment, ignore regions, and confidence; sparse projected points are not dense ground truth.

Compare categorical, continuous, and query-based lifting at matched resolution and latency. Better depth metrics do not guarantee better 3D detection if errors fall in task-irrelevant regions.

**Context:** LSS popularizes distributional lifting; CaDDN adds direct depth supervision, and BEVDepth adapts that lesson to surround-camera BEV detection.

**Takeaway:** A camera-to-BEV transform becomes trainable and auditable when depth uncertainty is explicit and privileged LiDAR is confined to the training target.
