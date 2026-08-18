---
title: "Geometry-Grounded Unified 3D Perception for Autonomous Driving"
date: '2026-08-13T00:00:00.000Z'
section: paper-shorts
postSlug: geometry-grounded-unified-3d-perception-for-autonomous-driving
legacyPath: /paper shorts/2026/08/13/geometry-grounded-unified-3d-perception-for-autonomous-driving.html
tags:
  - Autonomous Driving
  - 3D Perception
  - Multi-Task Learning
field: 'BEV Perception & Mapping'
summary: "2026 – Geometry-Grounded Unified 3D Perception for Autonomous Driving"
---

## 2026 – Geometry-Grounded Unified 3D Perception for Autonomous Driving

**arXiv:** [2608.13147](https://arxiv.org/abs/2608.13147)<br />
**Project:** [GeoUP](https://buaa-colalab.github.io/geoup_page)

## Summary

GeoUP adapts a reconstruction-oriented latent to calibrated streaming driving scenes and uses it as a shared representation for depth, 3D detection, and semantic occupancy. It separates self-, temporal-, and view-attention and adds calibration-aware raymaps so the latent carries metric camera geometry. Joint multi-task training across nuScenes, Argoverse 2, Waymo, KITTI, and DDAD yields strong reported results across the three readouts.

## Core Insights

Recognition-pretrained image features can preserve semantics while leaving metric 3D structure to downstream heads. GeoUP reverses that priority: the shared latent is trained to reconstruct geometry, then decoded at surface, instance, and volume levels. Raymaps provide camera intrinsics and extrinsics explicitly, while factorized attention separates temporal correspondence from cross-view interaction.

On the displayed nuScenes detection table, GeoUP with the listed multi-frame configuration reaches 59.2 mAP and 65.3 NDS, compared with 52.1 mAP and 60.8 NDS for StreamPETR. The appendix reports an efficiency cost: the ViT-L model runs at 2.18 FPS for one frame and 0.81 FPS for four frames at the displayed input size. The accuracy result therefore does not imply a ready-made real-time stack.

![GeoUP pipeline for geometry-aware tokens from calibrated streaming multi-view input](/assets/images/geoup-pipeline-paper-figure.jpg)
_GeoUP combines image patches, raymaps, and camera tokens before decoding depth, boxes, and occupancy from one latent. Source: [GeoUP](https://arxiv.org/abs/2608.13147)._

## High-Level Takeaways

- GeoUP informs whether one geometry-grounded latent can replace separate task-specific 3D representations.
- The training unit is a calibrated multi-camera temporal window with heterogeneous depth, detection, and occupancy labels.
- Shared geometry improves transfer potential, but the multi-frame ViT-L cost is a first-order deployment constraint.
- The conclusion would weaken if task-specific models match accuracy at a fraction of the latency, or if raymap and calibration noise break cross-dataset transfer.
