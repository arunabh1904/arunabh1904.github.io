---
title: 'VoTr: Voxel Transformer for 3D Object Detection'
date: '2021-09-06T04:00:00.000Z'
section: paper-shorts
postSlug: votr-voxel-transformer-for-3d-object-detection
legacyPath: /paper shorts/2021/09/06/votr-voxel-transformer-for-3d-object-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2021 – VoTr: Voxel Transformer for 3D Object Detection'
---
## 2021 – VoTr

**arXiv:** [2109.02497](https://arxiv.org/abs/2109.02497)

### Method and reported result

VoTr replaces a sparse-convolutional LiDAR backbone with attention over occupied voxels. Local attention supplies dense nearby context; dilated attention samples a much larger spatial region; and Fast Voxel Query retrieves those neighbors without scanning the full voxel grid.

## Summary

> The paper's useful question is narrower than “should LiDAR use transformers?” It asks when content-dependent long-range retrieval is worth the indexing and latency that convolution avoids.

## Core Insights

VoTr has sparse voxel modules that may create features at new active positions and submanifold modules that update only existing positions. Each query attends to a fixed-size set of local or dilated occupied voxels. The fixed budget keeps attention bounded even when the scene contains tens of thousands of active voxels.

![VoTr: Voxel Transformer for 3D Object Detection source figure: The overall architecture of Voxel Transformer (VoTr).](/assets/images/votr-voxel-transformer-for-3d-object-detection-paper-figure.webp)
*The overall architecture of Voxel Transformer (VoTr). source: [VoTr: Voxel Transformer for 3D Object Detection](https://arxiv.org/abs/2109.02497)*

![Figure 3 from VoTr: Voxel Transformer for 3D Object Detection](/assets/images/votr-voxel-transformer-for-3d-object-detection-source-figure-3.webp)
*Figure 3 Illustration of Local and Dilated Attention. We note that this is a 2D example and can be easily extended to 3D cases. For each query (red), Local Attention (yellow) focuses on the local region while Dilated Attention (green) searches the whole space with gradually enlarged steps. The non-empty voxels (light blue) which meet the searching locations are selected as the attending voxels (dark blue). source: [VoTr: Voxel Transformer for 3D Object Detection](https://arxiv.org/abs/2109.02497)*

![Figure 1 from VoTr: Voxel Transformer for 3D Object Detection](/assets/images/votr-voxel-transformer-for-3d-object-detection-source-figure-1.webp)
*Figure 1 (a) 3D convolutional network. source: [VoTr: Voxel Transformer for 3D Object Detection](https://arxiv.org/abs/2109.02497)*


Replacing SECOND's convolutional backbone with VoTr improves Waymo Level-1 vehicle mAP by 1.05 points. The gains grow with range: 1.42 points at 30–50 m and 1.72 beyond 50 m. On KITTI, adding dilated attention raises moderate car AP from 75.48 to 78.27. Increasing attended voxels from 24 to 48 adds 1.19 AP, which makes the accuracy-cost relationship explicit.

| Backbone | KITTI speed | Moderate car AP | Interpretation |
| --- | ---: | ---: | --- |
| SECOND | 20.73 Hz | 76.48 | Fast local convolutional context. |
| VoTr-SSD | 14.65 Hz | 78.27 | Larger adaptive context for about 20 ms extra latency. |
| PV-RCNN | 9.25 Hz | 83.69 | Stronger two-stage baseline at lower throughput. |
| VoTr-TSD | 7.17 Hz | 84.04 | Transformer context inside a heavier two-stage detector. |

These rows use the paper's KITTI validation setup and are not an accelerator-independent ranking. The important result is that long-range context helps sparse, distant objects, while the attention implementation is slower than the convolutional backbone it replaces.

## High-Level Takeaways

- VoTr informs whether to buy LiDAR receptive field through deeper sparse convolution or explicit voxel attention. Its atomic unit is an occupied voxel, but capacity is allocated through a bounded neighbor list. Local and dilated attention share weights across voxels; the coordinate query policy determines which evidence each token can reach.
- The decisive control matches receptive field, parameter count, active voxels, and deployed-kernel quality. VoTr should be rejected if large-kernel or multi-stage sparse convolution matches far-range recall at lower P99 latency. At 10× active voxels, neighbor lookup and irregular gathers are more likely to fail first than the attention matrix itself.
- SECOND established sparse convolution for voxel detection. VoTr introduced sparse voxel attention; SST and DSVT later use windowed or rotated-set designs that are easier to batch and deploy.
- Sparse attention earns its cost when distant, incomplete objects need context that a local voxel kernel cannot reach cheaply.
