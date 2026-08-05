---
title: 'NLSPN: Non-Local Spatial Propagation Network for Depth Completion'
date: '2020-07-20T04:00:00.000Z'
section: paper-shorts
postSlug: nlspn-non-local-spatial-propagation-network-for-depth-completion
legacyPath: /paper shorts/2020/07/20/nlspn-non-local-spatial-propagation-network-for-depth-completion.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2020 – NLSPN: Non-Local Spatial Propagation Network for Depth Completion'
---
## 2020 – NLSPN

**arXiv:** [2007.10042](https://arxiv.org/abs/2007.10042)

**Code:** [zzangjinsun/NLSPN](https://github.com/zzangjinsun/NLSPN)

### Method and reported result

NLSPN predicts an initial dense depth map, confidence, non-local neighbor offsets, and affinities. Iterative propagation then gathers depth from learned relevant locations rather than a fixed local window. Confidence enters the affinity normalization so noisy sparse-depth measurements do not spread indiscriminately.

## Summary

The method targets the mixed-depth problem: a local kernel near an object boundary often copies background depth into the foreground or the reverse.

## Core Insights

Each pixel selects a small set of neighbors that can lie on the same object or plane even when they are not adjacent. Learnable normalization expands the feasible affinity space while maintaining stable iteration. Confidence jointly suppresses unreliable source pixels during propagation instead of masking them only after a dense estimate is formed.

On KITTI Depth Completion test, NLSPN reports 741.68 mm RMSE and 199.59 mm MAE, compared with 758.38 and 226.50 for DeepLiDAR in the cited table. On NYU Depth V2, it reports 0.092 m RMSE. The ablation attributes the improvement to all three choices:

| Propagation choice | KITTI validation RMSE |
| --- | ---: |
| Fixed local neighbors, no confidence | 908.4 mm |
| Fixed local neighbors with confidence | 890.4 mm |
| Non-local neighbors with learned affinity | 886.0 mm |
| Non-local neighbors, learned normalization, confidence | 884.1 mm |

The reported non-local neighbors have lower depth variance than fixed local ones, supporting the boundary argument. The method still assumes sparse depth is present at runtime and pays for iterative dense image-space refinement; it is not a recipe for removing LiDAR from deployment.

## High-Level Takeaways

- NLSPN informs whether depth completion should use a large feed-forward decoder or a learned propagation rule. Its atomic unit is a depth pixel, but the neighborhood graph is predicted from RGB and sparse depth. The expensive decision is iterative full-resolution propagation and its memory access pattern.
- The matched alternative compares non-local propagation with a transformer or convolutional decoder at equal resolution, iterations, and P99 latency, with boundary- and range-stratified errors. NLSPN loses if the learned offsets become unstable under domain shift or if a simpler local kernel matches boundary quality. At higher image resolution, the dense state and repeated gathers dominate.
- Sparse-to-Dense regresses depth directly; DeepLiDAR injects surface normals; GuideFormer uses transformer guidance. NLSPN isolates learned propagation as a way to preserve boundaries around sparse measurements.
- Depth completion should move measurements along learned geometric neighborhoods, not assume the nearest pixels belong to the same surface.
