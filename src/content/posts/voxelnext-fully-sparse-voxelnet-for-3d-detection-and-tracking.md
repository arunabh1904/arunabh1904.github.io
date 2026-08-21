---
title: 'VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking'
date: '2023-03-20T04:00:00.000Z'
section: paper-shorts
postSlug: voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking
legacyPath: /paper shorts/2023/03/20/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2023 – VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking'
---
## 2023 – VoxelNeXt

**arXiv:** [2303.11301](https://arxiv.org/abs/2303.11301)

**Code:** [dvlab-research/VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt)

### Method and reported result

VoxelNeXt keeps LiDAR detection sparse through the prediction head. Instead of compressing sparse 3D features into a dense BEV heatmap and looking for hand-defined centers or anchors, it lets selected occupied voxels predict boxes directly. Sparse max pooling replaces dense heatmap peak extraction, and the predicting voxel can also support tracking association.

## Summary

> The paper tests a stronger claim than sparse backbones: the output interface does not have to become dense merely because earlier detectors did.

## Core Insights

Extra downsampling stages enlarge the sparse receptive field without filling the grid. Feature-magnitude pruning removes up to half of selected voxels with little validation loss in the reported setting. Sparse height compression then combines a 3D backbone with a 2D sparse head, retaining vertical reasoning early while avoiding an expensive 3D prediction stage.

![VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking source figure: Detailed structure of VoxelNeXt framework.](/assets/images/voxelnext-fully-sparse-voxelnet-for-3d-detection-and-tracking-paper-figure.webp)
_Detailed structure of VoxelNeXt framework. Source: [VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking](https://arxiv.org/abs/2303.11301), Figure 4, via arXiv HTML._


On nuScenes test, VoxelNeXt reports 64.5 mAP and 70.0 NDS at 66 ms; its double-flip variant reaches 66.2 mAP and 71.4 NDS. On tracking, the corresponding variants report 69.5 and 71.0 AMOTA. A controlled quarter-data comparison against CenterPoint improves mAP by 0.9 and NDS by 1.0.

| Property | Reported evidence | Design consequence |
| --- | --- | --- |
| Direct voxel prediction | Most predicting voxels are not near box centers | A center proxy is convenient, not required. |
| Spatial pruning | Up to 50% pruning has little reported decay | Sparse capacity can be allocated by learned feature strength. |
| Postprocessing | Sparse max pool reaches 56.2 mAP without NMS | Peak selection can remain on the active set. |
| Systems cost | 38.7G FLOPs versus CenterPoint's 186.6G | Wall-clock improvement is much smaller than the FLOP ratio. |

The authors explicitly identify that final mismatch as a limitation: sparse operations depend heavily on implementation and hardware. Direct voxel predictions can also come from outside the box, which makes their provenance less geometrically intuitive than center-based detections.

## High-Level Takeaways

- VoxelNeXt informs whether a sparse LiDAR stack should densify for detection and tracking. Its atomic unit remains the occupied voxel from input through output. The expensive architectural decision is the sparse runtime itself: coordinate maps, pruning, pooling, and gathers must all remain efficient on the deployed accelerator.
- A matched rejection test compares VoxelNeXt and CenterPoint with equal backbone capacity, range, voxel resolution, and compiler effort. The fully sparse head loses if dense heatmaps give better small-object recall or comparable latency, or if pruning destabilizes degraded and crowded scenes. At larger range, the active voxel count and sparse-kernel memory traffic become the likely limits.
- SECOND makes the 3D backbone sparse; CenterPoint densifies into BEV for prediction. VoxelNeXt removes that final dense proxy and links the predicting voxel to tracking.
- A sparse backbone is only half a sparse detector; VoxelNeXt asks the prediction head and tracker to operate on the active set too.
