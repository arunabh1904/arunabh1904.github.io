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

**arXiv:** [2608.13147](https://arxiv.org/abs/2608.13147)<br />
**Project:** [GeoUP](https://buaa-colalab.github.io/geoup_page)

## Summary

> GeoUP adapts the reconstruction-oriented latent of VGGT to calibrated, streaming driving scenes and uses one shared geometry representation for metric depth, 3D detection, and semantic occupancy. Plücker raymaps inject camera geometry, factorized self/temporal/view attention separates different correspondences, and heterogeneous datasets supervise different readouts of the same scene. The multi-dataset model improves nuScenes detection from 57.9/64.4 to 59.2/65.3 mAP/NDS, but the ViT-L geometry backbone runs at 2.18 FPS for one frame and 0.81 FPS for four on the reported H20 setup.

## Core Insights

### Geometry enters before the task heads diverge

A recognition backbone can tell two image patches apart while leaving their metric relationship to a downstream head. GeoUP changes the starting representation. DINOv2 patch tokens are augmented with a six-dimensional Plücker ray derived from each patch center, camera intrinsics, and camera pose. Camera and register tokens then carry frame/view structure through a compact VGGT-derived backbone. The result is decoded at three levels: per-view metric depth for surfaces, RayDN-style queries for object instances, and an OPUS-V2-style decoder for volumetric occupancy. A camera-pose head is kept during training as an explicit geometric constraint, even though its prediction is not consumed by the downstream heads at inference.

![Recognition features can encode semantics without explicit metric geometry](/assets/images/geometry-grounded-unified-3d-perception-for-autonomous-driving-source-figure-1.webp)
*Fig 1: The source’s Figure 1 contrasts semantic recognition with the missing geometric and cross-view consistency that motivates GeoUP. | source: [Geometry-Grounded Unified 3D Perception for Autonomous Driving, Figure 1](https://arxiv.org/abs/2608.13147)*

GeoUP also changes how tokens communicate. Self-attention stays within an image, temporal attention connects the same camera across timestamps, and view attention exchanges information across cameras at one timestamp. Historical patch tokens are cached while only current-frame images are encoded, which makes the streaming interpretation concrete: time is a structured axis rather than one undifferentiated global attention pool.

### The ablations show where structure helps

On the nuScenes validation ablation, global cross-image attention alone gives 53.8 mAP / 61.7 NDS, 37.3 mIoU / 41.8 RayIoU. Temporal attention alone gives 54.6 / 62.0 and 38.8 / 42.6; combining temporal and view attention reaches 55.5 / 62.3 and 39.9 / 43.6; adding the Plücker ray prior reaches 56.4 / 63.0 and 40.3 / 44.5. The ordering suggests that the useful inductive bias is not simply “more views.” Same-camera history is easier to align than cameras with limited overlap, while the raymap supplies the metric cue that appearance alone lacks.

The frame-count study gives a second intuition. With temporal heads, one frame gives 55.1 mAP / 62.5 NDS and 40.8 mIoU / 45.0 RayIoU; four frames reach 56.5 / 63.6 and 41.5 / 45.9; eight frames reach 56.9 / 63.6 and 41.6 / 46.1. Most of the gain arrives by four frames, so the default is a measured tradeoff between continuity and backbone cost rather than an assumption that longer context always helps.

### Heterogeneous supervision buys transfer at a heavy cost

The single-dataset GeoUP model reports 57.9 mAP / 64.4 NDS on nuScenes, 37.2 mAP / 28.2 CDS on Argoverse 2, and 51.5 mAPL / 67.0 mAP / 63.5 mAPH on Waymo. The joint model trains over nuScenes, Argoverse 2, Waymo, DDAD, and KITTI, masking losses when a dataset lacks a task label. It improves the corresponding scores to 59.2 / 65.3, 43.6 / 33.8, and 54.3 / 70.7 / 67.7. Occupancy rises from 41.5 mIoU / 45.9 RayIoU to 42.3 / 47.0, and DDAD depth reaches 0.123 Abs Rel with 87.6% threshold accuracy.

The shared latent also transfers to NAVSIMv2 through the unchanged DriveSuprim planning decoder: GeoUP reaches 87.9%/91.4% EPDMS under the paper’s original/corrected evaluators, versus 87.1%/90.5% for the DA-ViT-L backbone. The cost is concentrated in the geometry backbone. On one H20 with batch size 1, the full model is 2.18 FPS for one frame and 0.81 FPS for four; the backbone accounts for 66.5% and 87.8% of the measured one- and four-frame latency. Task heads remain separate, so the current contribution is a shared representation rather than a fully unified decoder.

## High-Level Takeaways

- GeoUP’s main bet is that depth, boxes, and occupancy become easier to transfer when they read one explicitly metric scene latent.
- Factorizing temporal and view attention is a structural choice: same-camera history helps more than unrestricted cross-view mixing, and raymaps add the missing coordinate frame.
- Multi-dataset supervision improves all reported readouts, but the comparison changes both data scale and optimization, so it is evidence for useful supervision diversity rather than a pure architecture-only gain.
- Four frames capture most of the measured temporal benefit; beyond that, the geometry backbone becomes the dominant cost.
- The next practical step is a lighter geometry backbone or shared decoder tested under calibration noise, dropped cameras, and matched latency against task-specific systems.
