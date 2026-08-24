---
title: 'PanoOcc: Unified Occupancy Representation for Camera-Based 3D Panoptic Segmentation'
date: '2023-06-16T00:00:00.000Z'
section: paper-shorts
postSlug: panoocc-unified-occupancy-representation-for-camera-based-3d-panoptic-segmentation
legacyPath: /paper shorts/2023/06/16/panoocc-unified-occupancy-representation-for-camera-based-3d-panoptic-segmentation.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2023 – PanoOcc: coarse-to-fine voxel queries for camera-based 3D panoptic occupancy'
---
## 2023 – PanoOcc

**arXiv:** [2306.10013](https://arxiv.org/abs/2306.10013)

**Code:** [Robertwyq/PanoOcc](https://github.com/Robertwyq/PanoOcc)

## Summary

> PanoOcc shows that a full 3D voxel representation need not carry the usual dense-grid cost. Its coarse-to-fine decoder preserves height and instance structure while using less memory than either a high-resolution voxel grid or the compared tri-plane system.

PanoOcc takes multi-frame, surround-view images and predicts one panoptic occupancy volume: every occupied voxel receives a semantic label, while foreground voxels also receive an instance identity. This unifies camera-based detection and semantic occupancy on nuScenes, but the panoptic evaluation samples predictions at LiDAR points rather than evaluating every voxel in the dense output.

## Core Insights

The model first lifts image features into low-resolution 3D voxel queries. Voxel cross-attention gathers multi-view image evidence, voxel self-attention exchanges information inside the volume, and a temporal encoder aligns and concatenates historical voxel features. The decoder then upsamples the volume and optionally prunes predicted empty regions. Detection queries select features from the same voxel representation, so detection and segmentation update a shared geometric state.

![PanoOcc: Unified Occupancy Representation for Camera-Based 3D Panoptic Segmentation source figure: The overall framework of PanoOcc.](/assets/images/panoocc-unified-occupancy-representation-for-camera-based-3d-panoptic-segmentation-paper-figure.webp)
*The overall framework of PanoOcc. source: [PanoOcc: Unified Occupancy Representation for Camera-Based 3D Panoptic Segmentation](https://arxiv.org/abs/2306.10013)*

![Figure 3 from PanoOcc: Unified Occupancy Representation for Camera-Based 3D Panoptic Segmentation](/assets/images/panoocc-unified-occupancy-representation-for-camera-based-3d-panoptic-segmentation-source-figure-3.webp)
*Figure 3 Illustration of occupancy sparsify. It serves as an optional technique to boost efficiency. We use BEV representation for simple illustration, while it is actually a 3D process. The light yellow region will be pruned according to occupancy masks. source: [PanoOcc: Unified Occupancy Representation for Camera-Based 3D Panoptic Segmentation](https://arxiv.org/abs/2306.10013)*


The controlled query-form ablation is the clearest reason to keep 3D voxels. At roughly 40,000 queries, PanoOcc's 50 x 50 x 16 voxel grid reports 70.7 mIoU, compared with 68.8 for TPVFormer's tri-plane queries and 56.2 for BEVFormer-style 2D BEV queries. Holding horizontal resolution at 50 x 50 and increasing vertical bins from 4 to 16 raises mIoU from 60.8 to 66.1, showing that height resolution carries information that a flat BEV map drops.

| Representation or decoder | Train / inference memory | Latency | mIoU |
| --- | ---: | ---: | ---: |
| Direct 200 x 200 x 8 voxels | 37.0 / 9.5 GB | 255 ms | 67.9 |
| Coarse 50 x 50 x 16 voxels with upsampling | 18.0 / 5.7 GB | 149 ms | 68.3 |
| PanoOcc-Base, full comparison | 24.0 / 6.0 GB | 203 ms | 71.7 |
| TPVFormer-Base, full comparison | 33.5 / 7.1 GB | 268 ms | 68.9 |

The coarse-to-fine result changes the architecture decision: the expensive object is not a voxel representation by itself, but self-attention over a dense high-resolution volume. PanoOcc keeps the representation three-dimensional, performs global interaction while it is coarse, and spends high-resolution computation only during decoding. Its sparse variant keeps 5% of voxels after three pruning stages, reducing inference memory from 15 GB to 9 GB and latency from 126 ms to 112 ms, but mIoU falls from 65.4 to 63.9.

On nuScenes, the camera-only large temporal model reports 62.1 panoptic quality and 48.4 detection mAP. That is comparable to some LiDAR panoptic baselines in the paper, but well behind the reported LiDARMultiNet result of 81.8 PQ and 63.8 mAP. PanoOcc also extends to the [Occ3D benchmark](/paper%20shorts/2023/04/27/occ3d-large-scale-3d-occupancy-prediction-benchmark.html), although its best memory and panoptic results come from different experimental settings and should not be combined into one deployment claim.

## High-Level Takeaways

- PanoOcc changes occupancy from a semantic-only grid into a shared object-and-surface representation by adding instance-aware detection on top of voxel features.
- Coarse-to-fine decoding makes 3D voxels competitive with compressed 2D representations because long-range interaction happens before the expensive spatial expansion.
- The matched-query ablation supports retaining height explicitly; the sparsification ablation shows that more aggressive pruning trades 1.5 mIoU for lower memory.
- The result does not establish camera-only parity with strong LiDAR panoptic systems, and dense-output quality is only partially tested by evaluation at LiDAR sample points.
