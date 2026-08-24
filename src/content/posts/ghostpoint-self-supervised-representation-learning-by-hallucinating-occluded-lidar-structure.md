---
title: "GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure"
date: '2026-08-14T00:00:00.000Z'
section: paper-shorts
postSlug: ghostpoint-self-supervised-representation-learning-by-hallucinating-occluded-lidar-structure
legacyPath: /paper shorts/2026/08/14/ghostpoint-self-supervised-representation-learning-by-hallucinating-occluded-lidar-structure.html
tags:
  - Autonomous Driving
  - LiDAR
  - Self-Supervised Learning
field: 'BEV Perception & Mapping'
summary: "2026 – GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure"
---

## 2026 – GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure

**arXiv:** [2608.14428](https://arxiv.org/abs/2608.14428)

## Summary

> GhostPoint targets a visible-surface bias in LiDAR self-supervision. Point-wise objectives supervise measured returns, but 3D detection must reason about the unobserved parts of an object. GhostPoint discovers instance proposals, dilates them into local neighborhoods, and trains a predictor to hallucinate latent features for occluded voxels. The paper reports 59.5 mAP/64.2 NDS in its probabilistic nuScenes setting and 67.5 mAP/71.2 NDS after fine-tuning.

## Core Insights

The method uses two views of a scan. An EMA teacher encodes unmasked observations, while a student encodes masked observations. Observed or masked voxels match teacher-encoder targets; unobserved voxels in the dilated neighborhood match teacher-predictor hallucinations. This makes the prediction target include structure that the sensor did not directly measure.

| Setting | nuScenes mAP | nuScenes NDS | Waymo mAP |
| --- | ---: | ---: | ---: |
| PointINS supervised | 56.7 | 62.5 | 57.5 |
| GhostPoint probabilistic | 59.5 | 64.2 | 60.0 |
| GhostPoint fine-tuned | 67.5 | 71.2 | 70.1 |

The downstream gains are largest under sparse scans and limited labels, which is consistent with the proposed missing-structure target. The main uncertainty is whether hallucinated features encode object shape or simply dataset-specific priors. A cross-sensor test with held-out object geometries and an ablation that removes instance-aware dilation would separate those explanations.

![GhostPoint method overview with instance-neighborhood feature hallucination](/assets/images/ghostpoint-method-paper-figure.png)
*The predictor supplies targets in neighborhoods around discovered instances, extending self-supervision beyond visible LiDAR returns. source: [GhostPoint](https://arxiv.org/abs/2608.14428)*

![Figure 2 from GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure](/assets/images/ghostpoint-self-supervised-representation-learning-by-hallucinating-occluded-lidar-structure-source-figure-2.webp)
*Figure 2 Overview of GhostPoint : A scan is augmented into two views. The student encodes a masked version of each view , while the teacher encoder (EMA) processes the unmasked views. Teacher features are used to discover instances. Discovered instances drive Neighborhood Sampling, which dilates their occupancy to form a neighborhood voxel set (visible, masked, and unobserved voxels). source: [GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure](https://arxiv.org/abs/2608.14428)*

![Figure 1 from GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure](/assets/images/ghostpoint-self-supervised-representation-learning-by-hallucinating-occluded-lidar-structure-source-figure-1.webp)
*Figure 1 GhostPoint learns beyond visible LiDAR surfaces. Raw scans observe only sparse object returns, causing discovered pseudo-instance centers to be biased toward visible surfaces. From these pseudo instances, GhostPoint samples adjacent unobserved regions and hallucinates their latent representations, producing features whose centers better align with the ground-truth object centers, marked by yellow crosses, while recovering object-level structure as illustrated by the PCA visualization. source: [GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure](https://arxiv.org/abs/2608.14428)*


## High-Level Takeaways

- GhostPoint informs whether LiDAR pretraining should model occluded structure explicitly instead of treating only measured surfaces as valid targets.
- The atomic unit is a masked scan plus an instance-dilated voxel neighborhood with separate teacher-encoder and teacher-predictor targets.
- The representation transfers to detection, especially in sparse and low-label regimes, but hallucination quality is the critical hidden variable.
- The conclusion would weaken if gains vanish under new sensors, object categories, or instance-dilation ablations.
