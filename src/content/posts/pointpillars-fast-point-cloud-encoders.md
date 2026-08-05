---
title: 'PointPillars: Fast Point Cloud Encoders'
date: '2018-12-14T05:00:00.000Z'
section: paper-shorts
postSlug: pointpillars-fast-point-cloud-encoders
legacyPath: /paper shorts/2018/12/14/pointpillars-fast-point-cloud-encoders.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2018 – PointPillars: collapse height early and run LiDAR perception as 2D convolution'
---
## 2018 – PointPillars

**arXiv:** [1812.05784](https://arxiv.org/abs/1812.05784)

### Method and reported result

PointPillars partitions a point cloud into vertical columns, learns a PointNet-style feature for each non-empty pillar, scatters those features into a BEV pseudo-image, and applies a conventional 2D backbone and detection head. It removes 3D convolutions and manual feature engineering from the runtime path.

## Summary

The architecture is an explicit compression decision: vertical detail is summarized inside each pillar so nearly all expensive spatial processing happens on the ground plane.

## Core Insights

Each point is augmented with offsets from the pillar mean and pillar center before pointwise encoding and max pooling. Only non-empty pillars pass through the encoder, but the scattered BEV tensor is dense. The paper reports a 1.3 ms encoder, 16.2 ms full pipeline, 62 Hz model, and a 105 Hz faster variant on its hardware.

| Knob | Effect | Failure mode |
| --- | --- | --- |
| Pillar width | Sets BEV resolution and active-pillar count | Wide pillars merge small actors. |
| Maximum pillars/points | Bounds latency | Dense scenes can be truncated. |
| Height collapse | Enables 2D backbones | Loses explicit vertical topology. |
| Learned point features | Beats fixed pillar statistics | Depends on task and sampling distribution. |

## High-Level Takeaways

- PointPillars is appropriate when road-plane structure dominates and latency matters more than retaining a full 3D latent. For multi-sensor fusion, the pseudo-image offers a convenient BEV interface, but camera fusion cannot recover height information that was already compressed.
- The relevant benchmark is not only detector throughput. Measure active-pillar saturation, distant small-object recall, vertical classes, memory transfer, and P99 latency under dense traffic and accumulated sweeps.
- VoxelNet learns features in 3D voxels; PointPillars makes an aggressive height-for-speed exchange that later BEV fusion systems often inherit.
- Pillars are fast because they decide early that most downstream reasoning can live in BEV; that assumption should be task-tested rather than treated as free compression.
