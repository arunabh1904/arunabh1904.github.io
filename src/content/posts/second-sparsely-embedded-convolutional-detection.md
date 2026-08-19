---
title: 'SECOND: Sparsely Embedded Convolutional Detection'
date: '2018-10-06T04:00:00.000Z'
section: paper-shorts
postSlug: second-sparsely-embedded-convolutional-detection
legacyPath: /paper shorts/2018/10/06/second-sparsely-embedded-convolutional-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2018 – SECOND: Sparsely Embedded Convolutional Detection'
---
## 2018 – SECOND

**Paper:** [Sensors 18(10), 3337](https://doi.org/10.3390/s18103337)

### Method and reported result

SECOND turns VoxelNet's dense 3D middle encoder into a sparse one. It computes only around occupied voxels, then densifies after the vertical dimension has been compressed enough for a 2D proposal network. The paper also contributes ground-truth database sampling and an orientation loss that respects the fact that boxes separated by $\pi$ can describe the same footprint.

## Summary

> The lasting contribution is the execution rule: preserve 3D structure while it is informative, but do not convolve over empty road volume. That rule became a foundation for later LiDAR detectors and sparse voxel transformers.

## Core Insights

SECOND begins with voxel feature encoding, applies a sparse 3D convolutional middle network, converts the result to a dense BEV feature map, and predicts boxes with an RPN. Its GPU rule-generation step builds the active input-output pairs needed by each sparse convolution. On KITTI, the paper reports roughly $3\times$ faster inference than the dense counterpart; the small model runs at about 40 FPS and the larger model at 20 FPS on a GTX 1080 Ti.

Two details matter beyond speed. Database sampling inserts complete labeled objects and their points into training scenes, reducing foreground imbalance and accelerating convergence. The angle objective separates overlap from direction: sine-based regression avoids a large penalty for geometrically equivalent headings, while an auxiliary classifier recovers the discrete direction.

| Decision | SECOND choice | Boundary |
| --- | --- | --- |
| 3D computation | Sparse convolution on active voxels | Rule generation and irregular memory access remain real costs. |
| Height compression | Densify after sparse 3D encoding | Later BEV stages no longer retain full vertical structure. |
| Long-tail training | Sample labeled objects into scenes | Inserted objects must remain physically and contextually plausible. |
| Orientation | Periodic regression plus direction class | Symmetric geometry and semantic heading are handled separately. |

The paper reports lower pedestrian and cyclist performance than for cars and identifies camera fusion as future work. Its evidence also comes from KITTI's relatively small range and class set, so the exact speed and accuracy numbers should not be transferred directly to a modern surround-sensor stack.

## High-Level Takeaways

- SECOND informs where a LiDAR model should first become dense. Its atomic unit is an occupied voxel; parameter sharing occurs through sparse kernels over a coordinate map. The expensive decision is not whether the mathematical tensor is sparse, but whether the target accelerator can generate and traverse active rules cheaply enough to beat dense kernels at the observed occupancy.
- A matched test should compare sparse and dense 3D middle encoders with the same voxel size, receptive field, BEV head, and P99 latency measurement. The sparse design loses if indexing and memory movement erase its arithmetic savings or if early BEV conversion preserves the same detection quality more cheaply. At larger range and finer voxels, active-set bookkeeping and the final dense BEV map become the likely bottlenecks.
- VoxelNet established learned voxel features; SECOND made sparse 3D convolution practical for detection. VoTr and DSVT later replace parts of the convolutional neighborhood with bounded attention, while VoxelNeXt removes the dense prediction head as well.
- Sparse LiDAR modeling starts by refusing to compute over empty volume, but the real system boundary is where sparse indexing stops paying for itself.
