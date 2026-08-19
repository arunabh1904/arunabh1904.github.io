---
title: "Towards Real-Time and Adaptable LiDAR Scene Completion"
date: '2026-08-17T00:00:00.000Z'
section: paper-shorts
postSlug: towards-real-time-and-adaptable-lidar-scene-completion
legacyPath: /paper shorts/2026/08/17/towards-real-time-and-adaptable-lidar-scene-completion.html
tags:
  - Autonomous Driving
  - LiDAR
  - Scene Completion
field: 'BEV Perception & Mapping'
summary: "2026 – Towards Real-Time and Adaptable LiDAR Scene Completion"
---

## 2026 – Towards Real-Time and Adaptable LiDAR Scene Completion

**arXiv:** [2608.16490](https://arxiv.org/abs/2608.16490)<br />
**Method:** RapidLiDAR

## Summary

> RapidLiDAR makes initialization a learned, spatially varying displacement of the observed point cloud instead of fixed noise or random Gaussian noise. A multi-scale reconstruction module then refines the coarse scene through voxel and BEV features. On SemanticKITTI and KITTI-360, the paper reports state-of-the-art-level completion quality at 0.1 seconds per scene—2.3× faster than the fastest prior method—and adapts to different input resolutions without manual noise recalibration.

## Core Insights

Generative scene completion spends time refining noise, while non-generative methods often start from a fixed perturbation that cannot cover large gaps and must be retuned for each sensor. RapidLiDAR predicts where each observed point should move to form a coarse, geometry-aware initialization. The refinement stage queries multi-scale 3D voxel and 2D BEV features, replacing expensive point-neighborhood operations with resolution-agnostic feature extraction.

![RapidLiDAR architecture with adaptive initialization and multi-scale refinement](/assets/images/rapidlidar-overview-paper-figure.jpg)
_The learned initialization expands partial observations before voxel/BEV refinement fills the remaining scene. Source: [RapidLiDAR](https://arxiv.org/abs/2608.16490)._

The paper's main evidence is a systems-quality trade-off: completion is reported at 0.1 seconds, close to the 10 Hz acquisition rate of automotive LiDAR, while quality remains competitive. The benchmark does not establish downstream detection or planning benefit, and the 10 Hz comparison depends on the paper's hardware and measurement boundary.

## High-Level Takeaways

- RapidLiDAR informs whether a scene-completion system should learn the coarse initialization rather than refine generic noise.
- The atomic unit is a partial point cloud transformed into a spatially adapted coarse scene and then refined from multi-scale voxel/BEV features.
- Replacing point-neighborhood searches improves speed and resolution adaptability, but it shifts the burden to the learned displacement field.
- The conclusion would weaken if completion speed does not translate into downstream perception gains under new sensors, ranges, and occlusion patterns.
