---
title: "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation"
date: '2022-05-26T00:00:00.000Z'
section: paper-shorts
postSlug: bevfusion-multi-task-multi-sensor-unified-bev
legacyPath: /paper shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: "2022 – BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation"
---
## 2022 – BEVFusion

**arXiv:** [2205.13542](https://arxiv.org/abs/2205.13542)

**Project:** [bevfusion.mit.edu](https://bevfusion.mit.edu/)

**Code:** [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion)

## Summary

> BEVFusion keeps camera and LiDAR features dense and aligned by translating both into one ego-centered BEV before concatenation. Camera pixels retain semantic density while LiDAR retains metric structure. The paper's precomputed grid association and interval-reduction kernel reduce camera pooling from more than 500 ms to 12 ms on an RTX 3090. On nuScenes, the camera+LiDAR model reaches 70.2 mAP and 72.9 NDS on test at 119.2 ms (8.4 FPS); its mean BEV map-segmentation IoU is 62.7, compared with 56.6 for the camera-only model.

## Core Insights

### Keep semantic density when modalities meet

Point-level fusion decorates the LiDAR points that happen to project into an image. With a typical 32-beam scanner, the paper measures fewer than 5% of camera features receiving such a match; background road and map evidence disappear before the task head sees them. BEVFusion instead runs a camera view transformer and a LiDAR voxel encoder separately, then brings both outputs into the same ego-centered grid. The fusion tensor therefore has a spatial address shared by both sensors while retaining camera features in cells where LiDAR has no return.

The first source figure is a useful diagnostic rather than just an architecture picture. In its image-space panels, nearby and distant LiDAR returns can become neighbors after projection, and camera features with no corresponding return are dropped. In the BEV panels, both modalities cover the same ground-plane cells. That is why this boundary helps map segmentation: the camera stream supplies dense semantic context and LiDAR supplies metric support without making the latter's sparsity the definition of the scene.

![Figure 1 from BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird](/assets/images/bevfusion-multi-task-multi-sensor-unified-bev-source-figure-1.webp)
*Fig 1: Image-to-LiDAR projection is semantically lossy because only a small fraction of camera features meet a LiDAR point; separate view transforms preserve both streams before they meet in a shared BEV grid. | source: [BEVFusion, Figure 1](https://arxiv.org/abs/2205.13542)*

### Make fixed camera geometry a cached data structure

The camera path predicts a discrete depth distribution for each image feature, scatters the feature along its candidate ray, and quantizes the resulting points into BEV cells. In the paper's representative setting, six cameras with a $32\times88$ feature map and 118 depth bins create about two million candidate points per frame—roughly two orders of magnitude denser than a LiDAR feature cloud. A naive BEV pooling implementation spends more than 500 ms on an RTX 3090 even though the rest of the model takes about 100 ms.

The optimization uses a property of a fixed calibrated rig: the 3D coordinates and BEV-cell indices do not change between frames. BEVFusion precomputes the association, sorts points by cell, and stores their ranks; inference only reorders the current feature values. The association path falls from 17 ms to 4 ms. After sorting, points belonging to one cell form a contiguous interval. Instead of a prefix sum that writes unused partial sums, a specialized kernel assigns one thread to each BEV interval and reduces only that interval, cutting aggregation from 500 ms to 2 ms. The full camera-to-BEV transform falls to 12 ms. The paper describes this as exact for the cached calibration, so the speedup does not come from truncating depth points or assuming a uniform distribution.

### A common trunk supports geometry and semantics

After concatenation, a small convolutional BEV encoder compensates for local misalignment caused by camera depth uncertainty. Grid sampling adapts the shared tensor to the different ranges and resolutions required by detection and map segmentation, then task-specific heads decode boxes or six independent binary map classes. The second source figure shows the contract clearly: modality-specific encoders end at BEV, one convolutional trunk is reused, and the heads specialize only after fusion.

![Camera and LiDAR features converted into a shared BEV before a common encoder and task-specific heads](/assets/images/bevfusion-multi-task-multi-sensor-unified-bev-source-figure-2.webp)
*Fig 2: Camera and LiDAR features are transformed into modality-specific BEV tensors, concatenated and refined by a common BEV encoder, then decoded by task-specific detection and map-segmentation heads. | source: [BEVFusion, Figure 2](https://arxiv.org/abs/2205.13542)*

The source tables show why the same workspace matters for the two tasks. For detection, camera+LiDAR BEVFusion reaches 70.2 mAP/72.9 NDS on the nuScenes test split and 68.5/71.4 on validation, versus 68.9/71.6 test for TransFusion; the reported BEVFusion compute and latency are 253.2 G MACs and 119.2 ms, versus 485.8 G and 156.6 ms for TransFusion. For map segmentation, camera-only BEVFusion averages 56.6 IoU across drivable area, pedestrian crossing, walkway, stop line, carpark, and divider; adding LiDAR raises that to 62.7. The drivable-area score moves from 81.7 to 85.5 and the carpark score from 50.7 to 57.0. These are separate modality rows in the paper's Table III, not a claim that one multi-task score is being compared with one single-task score.

### Weather and sparsity expose the interface's limits

The fused BEV is useful when one modality is weak. On the paper's corruption split, camera+LiDAR BEVFusion reaches 69.9 mAP in rain and 42.8 at night, while map mIoU reaches 55.9 in rain and 43.6 at night. Under one-beam LiDAR, BEVFusion improves over the point-decorating MVP baseline by 12% while using 1.6× fewer MACs. The exact gain depends on the image depth estimate and calibration: a shared grid aligns addresses, but it does not make a camera depth prediction metrically certain or repair a failed calibration.

## High-Level Takeaways

- BEVFusion's central decision is to preserve dense camera semantics until both modalities share an ego-frame BEV address; point-level LiDAR decoration discards too much background evidence.
- The main systems result is the pooling implementation: precomputed cell ranks plus interval reduction change a 500+ ms camera transform into a 12 ms exact transform for the reported rig.
- Table I and Table III separate detection from map segmentation: camera+LiDAR reaches 70.2/72.9 detection and 62.7 mean IoU, while camera-only reaches 56.6 mean IoU on the six map classes.
- Dense BEV still carries calibration, depth, memory, and sensor-failure dependencies; the reported speed is tied to the fixed camera geometry and RTX 3090 implementation.
