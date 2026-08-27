---
title: 'SST: Single-Stride Sparse Transformer for 3D Detection'
date: '2021-12-13T05:00:00.000Z'
section: paper-shorts
postSlug: sst-single-stride-sparse-transformer-for-3d-detection
legacyPath: /paper shorts/2021/12/13/sst-single-stride-sparse-transformer-for-3d-detection.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2021 – SST: preserve high-resolution sparse LiDAR features without a downsampling pyramid'
---
## 2021 – SST

**arXiv:** [2112.06375](https://arxiv.org/abs/2112.06375)

### Method and reported result

SST argues that the conventional downsampling hierarchy is a poor fit for small 3D actors. It keeps one high-resolution stride and applies Sparse Regional Attention only to non-empty voxel tokens inside local windows. Alternating window shifts allow information to cross regional boundaries without densifying the whole plane.

## Summary

> Its claim is not that attention is inherently better than convolution. The advantage comes from matching attention's variable receptive field to a sparse input while preserving spatial resolution.

## Core Insights

Sparse windows bound the quadratic attention cost by local token count. Empty locations never become tokens, while region shifts create cross-window context. The supplementary comparison reports 64.69 AP for a standard convolutional variant versus 51.57 for a submanifold sparse-convolution variant, illustrating how a sparse operator that never activates new sites can struggle to exchange context.

![SST: Single-Stride Sparse Transformer for 3D Detection source figure: Architecture overview for Single-stride Sparse Transformer (SST).](/assets/images/sst-single-stride-sparse-transformer-for-3d-detection-paper-figure.webp)
*Fig 1: SST voxelizes the point cloud, alternates regional and shifted-regional self-attention blocks, recovers a dense feature map, and applies a standard detection head. | source: [SST: Single-Stride Sparse Transformer for 3D Detection](https://arxiv.org/abs/2112.06375)*

![Figure 2 from SST: Single-Stride Sparse Transformer for 3D Detection](/assets/images/sst-single-stride-sparse-transformer-for-3d-detection-source-figure-2.webp)
*Fig 2: Relative object area is much smaller in the Waymo 3D detection set than in COCO: only 0.54% of Waymo objects exceed the 0.04 threshold versus 73.03% in COCO, motivating higher-resolution sparse features. | source: [SST: Single-Stride Sparse Transformer for 3D Detection](https://arxiv.org/abs/2112.06375)*


| Property | SST choice | Consequence |
| --- | --- | --- |
| Spatial stride | One | Small-object detail survives. |
| Compute support | Non-empty voxels | Cost follows evidence rather than area. |
| Context | Local shifted windows | Global context needs depth or larger windows. |
| Pedestrian result | 83.8 level-1 AP | Strong evidence for resolution-sensitive classes. |

## High-Level Takeaways

- SST is useful when the primary failure is lost small-object detail rather than insufficient global scene context. Window occupancy, maximum tokens, sorting, padding, and memory movement must be profiled; sparse FLOPs do not guarantee low wall-clock latency.
- The model also cannot represent unobserved free space through absent tokens. Dense BEV context can remain valuable for maps, occupancy, and safety envelopes even when actor detection is sparse.
- DSVT makes sparse token grouping more hardware-oriented; UniTR later applies a related sparse-transformer interface across modalities.
- Sparse attention earns its place when it preserves resolution and grows context without making empty road cells part of the compute bill.
