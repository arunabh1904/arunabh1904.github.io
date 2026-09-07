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

**arXiv:** [2608.14428](https://arxiv.org/abs/2608.14428)

## Summary

> GhostPoint targets the visible-surface bias in LiDAR self-supervision. Point-wise objectives supervise measured returns, while 3D detection must infer the full extent of an object behind occlusion. GhostPoint discovers pseudo-instances, dilates them into local neighborhoods, and trains a predictor to hallucinate latent features at masked and no-return voxels. The representation reaches 59.5 mAP / 64.2 NDS under decoder probing and 67.5 / 71.2 after full nuScenes fine-tuning, with the largest evidence appearing in sparse and low-label regimes.

## Core Insights

### The missing target is object extent, not another surface loss

Existing LiDAR SSL can learn excellent features on visible returns while still transferring weakly to detection. A partial scan may group the visible points of a car correctly, yet its pseudo-centroid is displaced toward the exposed side. Adding box regression to that same visible cluster does not fix the underlying target: the model is still asked to reason about a full box from features defined only where the sensor returned points. Segmentation tolerates this mismatch because its labels live on observed points; detection does not.

GhostPoint makes that mismatch explicit. The teacher encoder sees an unmasked scan, the student sees a randomly masked view, and an EMA update keeps the teacher stable. Softmap semantics and center-offset predictions discover pseudo-instances from the teacher’s visible features. The method then keeps only proposals with some student-visible evidence and dilates their voxel occupancy to create a neighborhood containing visible, masked, and genuinely unobserved positions.

![GhostPoint motivation: hallucinated neighborhoods extend visible LiDAR structure toward the object center](/assets/images/ghostpoint-self-supervised-representation-learning-by-hallucinating-occluded-lidar-structure-source-figure-1.webp)
*Fig 1: The source’s Figure 1 shows how visible-surface centroids can miss the object center and how GhostPoint’s hallucinated features recover object-level structure. | source: [GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure, Figure 1](https://arxiv.org/abs/2608.14428)*

### Hallucination is localized and asymmetrically supervised

Uniformly sampling empty outdoor space would spend most computation on background. Neighborhood Sampling instead queries voxels next to discovered instances. For a new voxel, the student initializes a token from nearby visible features with distance-weighted KNN interpolation, then a lightweight two-block predictor propagates context across the neighborhood. Visible tokens are copied through the predictor; only newly queried tokens keep predictor outputs.

The target is asymmetric for a reason. Measured occupied voxels use teacher-encoder Softmap and offset targets. No-return voxels use teacher-predictor targets, because there is no encoder feature to treat as ground truth. Masked observed points and true no-return points therefore share a completion objective without pretending that the latter were directly measured.

![GhostPoint teacher-student and neighborhood-predictor training pipeline](/assets/images/ghostpoint-self-supervised-representation-learning-by-hallucinating-occluded-lidar-structure-source-figure-2.webp)
*Fig 2: The source’s Figure 2 shows random masking, EMA teacher targets, instance-driven Neighborhood Sampling, and predictor-level distillation over non-visible voxels. | source: [GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure, Figure 2](https://arxiv.org/abs/2608.14428)*

The ablation isolates what matters. On nuScenes decoder probing, PointINS is 56.7 mAP / 62.5 NDS. Adding neighborhood sampling without a predictor reaches 57.3 / 62.7; a predictor with no warmup reaches 57.2 / 63.0; the two-stage warmup reaches 59.5 / 64.2. Replacing informative hallucination targets with zeros gives only 57.6 / 63.3, while swapping the same number of masked points for hallucinated points nearly matches the full model. The gain is therefore tied to what the predictor imagines, not merely to extra queried voxels or regularization capacity.

### Detection transfer improves under sparse evidence

GhostPoint is evaluated with PTv3 and a CenterPoint sparse CNN encoder. On nuScenes, the frozen-backbone decoder probe reaches 59.5 mAP / 64.2 NDS, compared with 56.7 / 62.5 for PointINS; full fine-tuning reaches 67.5 / 71.2. On Waymo Level 2, the corresponding values are 60.0 mAP under probing and 70.1 after fine-tuning. The protocol matters: probing trains the detector around a frozen representation, while fine-tuning lets the full detector adapt.

The label-efficiency table shows why the completion target is useful. With 0.1% of nuScenes labels, GhostPoint reaches 36.9 mAP / 52.1 NDS; with 1% it reaches 53.1 / 62.5; with 10% it reaches 63.5 / 68.8. It also preserves per-point transfer, reaching 74.2 mIoU in semantic segmentation, essentially matching PointINS, and improves panoptic transfer. The boundary is equally concrete: severe occlusion can still produce blurred neighborhoods, background outliers can be extended as if they belonged to an object, and probing remains below fully supervised training.

## High-Level Takeaways

- GhostPoint changes the SSL target from “explain measured surfaces” to “propagate object-aware structure into likely missing regions.”
- Instance dilation keeps completion computation local, while asymmetric teacher targets prevent unobserved voxels from being treated as observed truth.
- The controlled target swaps show that informative hallucination, rather than extra positions alone, explains most of the probing gain.
- The strongest evidence appears under limited labels and sparse scans, exactly where visible-surface bias is most damaging to box localization.
- A useful next test would vary sensor beam count, object category, and occlusion pattern while measuring hallucination precision and false boxes in empty space.
