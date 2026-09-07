---
title: 'BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View'
date: '2021-12-22T05:00:00.000Z'
section: paper-shorts
postSlug: bevdet-high-performance-multicamera-3d-object-detection-in-bev
legacyPath: /paper shorts/2021/12/22/bevdet-high-performance-multicamera-3d-object-detection-in-bev.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2021 – BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View'
---
## 2021 – BEVDet

**arXiv:** [2112.11790](https://arxiv.org/abs/2112.11790)

**Code:** [HuangJunJie2017/BEVDet](https://github.com/HuangJunJie2017/BEVDet)

## Summary

> BEVDet turns a camera-only detector into a measurable systems pipeline. An image encoder, a categorical-depth Lift-Splat view transformer, a BEV encoder, and a CenterPoint head each spend a different resolution or memory budget. The paper’s most useful lessons are that BEV-space augmentation is needed to prevent the BEV trunk from overfitting, category-aware Scale-NMS repairs small-object suppression, and fixed-rig pooling can remove a large sequential bottleneck. Its accuracy–speed points are strong for the time, but the fixed calibration contract and weaker appearance-attribute prediction remain part of the result.

## Core Insights

### The detector is a contract between two coordinate spaces

BEVDet has four modules: an image-view encoder and neck, a Lift-Splat-style view transformer, a BEV encoder, and a CenterPoint detection head. The view transformer predicts a categorical depth distribution, places image features at calibrated 3D candidates, and pools them into the ego-frame grid. In the default setup the depth range is 1–60 metres, with a bin interval tied to the BEV feature resolution as $1.25r$; the BEV region covers roughly 51.2 metres and the common tiny model uses 0.8-metre cells.

The point is not just modularity. Image resolution controls the evidence available before lifting, BEV resolution controls the metric grid after lifting, and the BEV encoder and head consume the result in yet another representation. A change called “higher resolution” therefore has no single cost or meaning.

![Figure 1 from BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View](/assets/images/bevdet-high-performance-multicamera-3d-object-detection-in-bev-source-figure-1.webp)
*Fig 1: BEVDet separates image-view encoding, categorical-depth lifting, BEV encoding, and the task head; each boundary exposes a different accuracy and compute trade-off. | source: [BEVDet, Figure 1](https://arxiv.org/abs/2112.11790)*

### BEV augmentation repairs a training-distribution mismatch

Each nuScenes sample contains six camera images, so the image encoder sees roughly six times as many image tensors as the downstream BEV trunk sees training examples. Image-space augmentation changes the features before the view transform, and the pixel-wise geometry largely decouples that change from the BEV encoder and detection head. In the paper’s ablation, image-view augmentation alone peaks at 23.0 mAP by epoch 4 and finishes at 17.4; adding BEV-space augmentation raises the peak to 31.6 by epoch 17 and finishes at 31.2. A BEV encoder makes image-only augmentation actively harmful in some configurations, while BEV augmentation targets the coordinate system that is actually being overfit.

BEVDet’s BEV augmentation flips, scales, and rotates the transformed feature together with the 3D targets. The operation is valid because the view transformer has already converted camera coordinates into a common plane. That dependency is a real boundary: an image-view detector or a transformer that fuses features before an explicit BEV tensor cannot automatically inherit the same augmentation argument.

The paper also adapts non-maximum suppression to the ground plane. In image space, a fixed IoU threshold works because perspective makes object footprints more similar. In BEV, a pedestrian or traffic cone may occupy less than one 0.8-metre output cell, so redundant predictions can have little overlap with the true positive and survive classical NMS. Scale-NMS enlarges boxes by category before applying the ordinary suppression rule, then restores their original sizes. On the validation ablation it raises mAP from 29.5 to 31.2; traffic-cone AP rises from 42.5 to 50.0 and pedestrian AP from 29.7 to 34.5.

### Resolution buys two different kinds of evidence

The controlled resolution sweep makes the cost contract concrete. At $704\times256$ input and a 0.8-metre BEV grid, BEVDet-Tiny reports 31.2 mAP, 39.2 NDS, and 15.6 FPS. Increasing only the image to $1056\times384$ gives 33.3 mAP and 41.0 NDS at 8.9 FPS. Keeping the small image but halving the BEV cell to 0.4 metres gives 31.5 mAP and 41.0 NDS at 10.0 FPS. Increasing both to $1056\times384$ and 0.4 metres gives 34.8/41.7 at 7.1 FPS, while the largest tested $1408\times512$ and 0.4-metre configuration reaches 36.0/43.8 at 5.0 FPS.

The pattern is useful because the two changes help different failure modes. More image pixels can recover visual detail before depth assignment; finer BEV cells give the head a more precise metric address. Their costs accumulate through the frustum and BEV tensors, so the best operating point depends on whether long-range appearance or localization is the bottleneck.

### Pooling is part of the model, not a footnote

The ordinary view transform accumulates every frustum feature into its voxel. BEVDet adds an auxiliary index that records the occurrence number of each voxel index. With fixed camera intrinsics and extrinsics, both indices can be computed during initialization. Features are placed into a two-dimensional matrix and summed along the auxiliary axis, replacing a sequential accumulation over points. The paper limits the auxiliary index to 300, drops later points, and reports negligible accuracy impact; the extra memory is set by the voxel count and that maximum index.

![Figure 3 from BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye View](/assets/images/bevdet-high-performance-multicamera-3d-object-detection-in-bev-source-figure-3.webp)
*Fig 2: The auxiliary index turns repeated voxel accumulation into a regular reduction over occurrence slots; the optimization is valid when the camera rig is fixed at inference. | source: [BEVDet, Figure 3](https://arxiv.org/abs/2112.11790)*

That implementation reduces BEVDet-Tiny’s reported latency from 137 ms to 64 ms, a 53.3% reduction. The same paper therefore supports two separate conclusions: a view transform can dominate wall time even when it is one arrow in the architecture diagram, and a fixed-rig optimization can be a deployment constraint. A changing camera mount, calibration update, or variable camera count requires rebuilding the mapping rather than reusing the cached indices.

### The benchmark trades appearance for geometry

On the nuScenes validation set, the tiny configuration reaches 31.2 mAP and 39.2 NDS at 15.6 FPS with 215.3 GFLOPs; the larger BEVDet-Base reaches 39.3 mAP and 47.2 NDS at 1.9 FPS. On the test set, a single BEVDet model with test-time augmentation reports 42.2 mAP and 48.2 NDS. The tiny model uses only $704\times256$ images, about one eighth the input pixel count of the $1600\times900$ FCOS3D comparison, yet reports higher validation mAP and much higher throughput in the paper’s hardware setup.

The metric breakdown explains what the BEV representation buys. BEVDet is strong on translation, scale, orientation, and velocity, where a ground-plane coordinate system helps the detector reason about metric relationships. Its attribute error is worse than image-view FCOS3D and PGD, consistent with compressing color and texture into a spatial grid. The result is a useful geometry-first baseline, not evidence that BEV makes every visual property easier.

## High-Level Takeaways

- BEVDet’s contribution is a complete camera-BEV contract: image evidence, categorical depth, metric pooling, BEV augmentation, and a task head must be evaluated together.
- The image/BEV augmentation ablation shows that the downstream BEV trunk overfits unless the transformed coordinate space is regularized directly.
- Scale-NMS repairs a ground-plane mismatch in classical IoU suppression and gives the clearest gains on small categories such as pedestrians and traffic cones.
- Image and BEV resolution improve different error sources, but the reported 36.0 mAP / 43.8 NDS point costs 5.0 FPS; resolution is a systems choice.
- Cached indexed pooling removes 53.3% of reported tiny-model latency under a fixed-rig assumption, while the weaker attribute breakdown marks the cost of geometry-first compression.
