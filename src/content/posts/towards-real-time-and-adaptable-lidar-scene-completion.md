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

**arXiv:** [2608.16490](https://arxiv.org/abs/2608.16490) · **Code:** [RapidLiDAR](https://github.com/AzharSindhi/RapidLiDAR)

## Summary

> RapidLiDAR learns where to place candidate completion points before refining them. Spatially varying displacements expand a partial scan toward missing geometry, then voxel and BEV features guide residual corrections. The base model completes 18,000 input points into 180,000 output points in 0.10 seconds on the reported RTX 6000 Ada, versus 0.23 seconds for LiNeXt, with slightly better Chamfer distance. Its strongest result is this measured quality–runtime trade; broader sensor-beam transfer and downstream safety are not established.

## Core Insights

### Refinement cannot easily recover a surface that initialization never reaches

A fixed perturbation around observed points tends to populate space close to already visible surfaces. If a large region is missing, a local refiner begins with too few candidates near it. Starting from scene-wide random noise solves coverage differently, but diffusion methods then need many iterations to recover useful geometry.

RapidLiDAR instead expands the observed point set and uses a learned displacement for each candidate. It still adds small Gaussian perturbations, 0.1 m in the reported setup, to separate repeated points. What changes is that the subsequent displacement is conditioned on scene features, rather than making one global noise radius responsible for covering every gap.

![Different initialization strategies leave different coverage for scene refinement](/assets/images/towards-real-time-and-adaptable-lidar-scene-completion-source-figure-1.webp)
*Fig 1: Compare the highlighted missing region before and after refinement. Fixed local perturbations remain near observed surfaces, while learned displacements place candidates across the gap before detailed correction begins. | source: [RapidLiDAR, Figure 1](https://arxiv.org/abs/2608.16490)*

The displacement is bounded by a scene-scale parameter, defaulting to 50 m. This remains a configured geometric range; the method removes the need to encode local coverage solely through manually retuned initialization noise, rather than eliminating all scale choices.

### Grid features replace repeated point-neighborhood searches

The input is voxelized at 0.3 m and encoded at four 3D scales. A dense BEV head merges height into channels and applies self-attention over the coarsest BEV grid to distribute broader context. For each candidate point, trilinear sampling from the 3D grids and bilinear sampling from the dense BEV map produce a feature vector used to predict its initial displacement.

After moving the points, the model samples their features again. The reconstruction module projects the multi-scale volumes into BEV maps and uses deformable cross-attention, with the moved points as queries. A final MLP predicts residual 3D corrections. Vertical information remains in the sampled 3D features and projected channels even though the attention operator works on 2D grids.

![RapidLiDAR shares multi-scale voxel and BEV features between learned initialization and reconstruction](/assets/images/rapidlidar-overview-paper-figure.jpg)
*Fig 2: The initializer moves candidate points using local and global features; the reconstruction branch reads features at the new positions. This second lookup lets the model refine geometry where it has just placed the candidates. | source: [RapidLiDAR, Figure 2](https://arxiv.org/abs/2608.16490)*

Avoiding farthest-point sampling and repeated k-nearest-neighbor grouping reduces a scene-scale bottleneck. Runtime is still affected by voxel resolution, output-point count, and deformable sampling; the architecture is not computationally independent of how much geometry it produces.

### The base-model speed and extra-refinement accuracy belong to different rows

Training uses SemanticKITTI sequences 00–10 except validation sequence 08. Zero-shot transfer is evaluated on KITTI-360 sequence 00 without fine-tuning. The base-model comparison reports:

| Method | SemanticKITTI CD | KITTI-360 CD | Time per scan |
| --- | ---: | ---: | ---: |
| LiNeXt | 0.214 | 0.217 | 0.23 s |
| RapidLiDAR | 0.206 | 0.211 | 0.10 s |

Lower Chamfer distance measures closer predicted and target point sets. The reported JSD metrics also compare their spatial distributions. Neither metric establishes semantic correctness or that a completed occluded region is safe to treat as observed.

An additional separately trained refinement network upsamples the output sixfold. That version reaches 0.138 SemanticKITTI CD and 0.140 KITTI-360 CD, compared with refined LiNeXt's 0.149 on each. The 0.10-second timing table refers to the base model at 0.206 CD, so the strongest refinement score should not inherit that latency without a separate measurement.

### The ablations support both stages without a monotone scaling story

Replacing learned initialization with LiNeXt-style fixed perturbation worsens CD from 0.206 to 0.218. Removing multi-scale reconstruction gives 0.215. Both stages matter, and their similar-sized effects support improving coverage before refining surfaces.

The resolution sweep exposes the practical trade. A 0.5 m voxel grid gives 0.214 CD at 0.07 seconds; 0.3 m gives 0.206 at 0.10 seconds; a finer 0.2 m grid worsens CD slightly to 0.208 while increasing latency to 0.14 seconds. More resolution is not automatically better. Likewise, displacement bounds of 50, 70, and 100 m give nearly equal scores in a separate downsampled evaluation, rather than demonstrating a need for ever-larger movement.

The default model has 11.8 million parameters, more than LiNeXt's 1.99 million, despite being faster. Operator choice matters more here than parameter count alone. Matching a 10 Hz scan interval on the test GPU is encouraging, but the full perception stack must also process, interpret, and act on the completed geometry. Diverse sensor-beam configurations and uncertainty-aware downstream use remain important next tests.

## High-Level Takeaways

- Learn candidate coverage before refinement when fixed local perturbations cannot reach missing surfaces.
- Grid sampling and BEV attention reduce expensive point-neighborhood work without making resolution and output count free.
- Keep base completion, optional sixfold refinement, and their runtime measurements separate.
- Completion quality measures plausible recovered geometry; it does not turn hallucinated occluded surfaces into sensor observations.
