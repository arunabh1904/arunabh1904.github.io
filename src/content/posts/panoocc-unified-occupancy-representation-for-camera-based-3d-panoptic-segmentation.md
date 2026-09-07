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

**arXiv:** [2306.10013](https://arxiv.org/abs/2306.10013)

**Code:** [Robertwyq/PanoOcc](https://github.com/Robertwyq/PanoOcc)

## Summary

> PanoOcc uses one camera-derived 3D voxel state for semantic occupancy, detection, and instance assignment. It keeps height explicit while doing the expensive interaction at coarse resolution, then restores detail with 3D upsampling and refines foreground instances using the detection head.
>
> The result is a useful architecture decision rather than a claim that dense voxels are free. The paper's matched query and efficiency ablations show when a 3D volume is worth its cost, while its evaluation on sparse LiDAR points leaves a clear limit on what the panoptic score measures.

## Core Insights

### A voxel query keeps the missing axis visible

PanoOcc starts with learnable queries shaped as a 3D grid. Voxel cross-attention projects each query into the camera views and gathers deformable image features; voxel self-attention exchanges information with nearby queries on the BEV plane at the same height. The temporal encoder transforms previous voxel grids into the current ego frame in 3D, then concatenates them and fuses them with residual 3D convolutions. This is more than adding a height index to a BEV token: the history alignment can represent an uphill or downhill surface without assuming that the road is flat.

The occupancy decoder upsamples the coarse volume with 3D deconvolutions. A detection head pools selected foreground voxels into a 2D BEV and a segmentation MLP queries the full volume. The refine module takes high-confidence 3D boxes, assigns their class to contained foreground voxels, and gives overlapping instances distinct IDs. Detection therefore supplies object-level structure to the panoptic output instead of asking a voxel classifier to invent instance boundaries alone.

![PanoOcc's unified voxel, temporal, detection, and panoptic pipeline](/assets/images/panoocc-unified-occupancy-representation-for-camera-based-3d-panoptic-segmentation-paper-figure.webp)
*Fig 1: Multi-view and multi-frame images become aligned voxel features, which are decoded into shared 3D detection and panoptic segmentation outputs. | source: [PanoOcc, Figure 2](https://arxiv.org/abs/2306.10013)*

### Coarse-to-fine changes the cost profile

At roughly 40,000 queries, the matched query-form ablation reports 70.7 mIoU for 50 x 50 x 16 3D voxels, versus 68.8 for TPVFormer's tri-plane representation and 56.2 for BEVFormer-style 2D BEV queries. Holding the horizontal grid at 50 x 50, increasing the height from 4 to 16 raises mIoU from 60.8 to 66.1. The experiment isolates why a volume helps: vertical structure carries semantic evidence that a flat feature map must compress or infer later.

The efficiency ablation compares a direct high-resolution volume with the coarse-to-fine path:

| Decoder design | Train / inference memory | Latency | mIoU |
| --- | ---: | ---: | ---: |
| Direct 200 x 200 x 8 voxels | 37 / 9.5 GB | 255 ms | 67.9 |
| Coarse 50 x 50 x 16 plus upsampling | 18 / 5.7 GB | 149 ms | 68.3 |

In the paper's full comparison, PanoOcc-Base uses 24 / 6.0 GB and 203 ms for 71.7 mIoU, versus TPVFormer-Base at 33.5 / 7.1 GB and 268 ms for 68.9. These are separate matched settings, so the ablation should be read as a compute explanation and the full comparison as a benchmark result.

![PanoOcc's optional occupancy sparsification](/assets/images/panoocc-unified-occupancy-representation-for-camera-based-3d-panoptic-segmentation-source-figure-3.webp)
*Fig 2: Occupancy sparsification prunes predicted empty regions during the 3D upsampling path; the illustration is a simplified BEV view of a 3D process. | source: [PanoOcc, Figure 3](https://arxiv.org/abs/2306.10013)*

### Temporal context helps the hard classes, but the metric is sparse

PanoOcc-Base reaches 70.7 mIoU on nuScenes semantic segmentation, Base-T reaches 71.6 with four frames at 0.5-second intervals, and Large-T reaches 74.5 with an InternImage-XL backbone. On camera-based panoptic segmentation, Large-T reports PQ 62.1, PQ-dagger 66.2, RQ 75.1, SQ 82.1, and detection mAP 48.4. The temporal category ablation shows particularly large gains for motorcycle (+11.7 mIoU) and trailer (+8.2), classes that are often occluded.

The caveat is built into the evaluation: semantic and panoptic predictions are assigned to sparse LiDAR points, and PQ is not a dense voxel metric. The camera model remains below LiDARMultiNet's reported PQ 81.8 and mAP 63.8. The sparse architecture keeps only 5% of voxels after three pruning stages and lowers mIoU from 65.4 to 63.9 in its own small-model ablation, making the quality-efficiency trade explicit.

## High-Level Takeaways

- PanoOcc's distinctive choice is a shared 3D voxel state that serves both dense semantics and object instances.
- Explicit height and 3D temporal alignment matter when geometry is not a flat road plane.
- Coarse-to-fine interaction preserves the volume while avoiding high-resolution global computation; sparsification pushes the same idea further.
- The paper supports camera-based panoptic occupancy as a useful shared representation, while sparse LiDAR-point evaluation and the LiDAR comparison leave dense-world quality only partly measured.
