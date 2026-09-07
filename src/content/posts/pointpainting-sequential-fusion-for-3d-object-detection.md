---
title: 'PointPainting: Sequential Fusion for 3D Object Detection'
date: '2019-11-22T05:00:00.000Z'
section: paper-shorts
postSlug: pointpainting-sequential-fusion-for-3d-object-detection
legacyPath: /paper shorts/2019/11/22/pointpainting-sequential-fusion-for-3d-object-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2019 – PointPainting: Sequential Fusion for 3D Object Detection'
---
## 2019 – PointPainting

**arXiv:** [1911.10150](https://arxiv.org/abs/1911.10150)

## Summary

> PointPainting gives a LiDAR detector a compact semantic side channel. An image segmentation network produces per-pixel class scores, calibration projects each LiDAR point into those scores, and the decorated cloud goes through an otherwise ordinary detector. That interface transfers across PointPillars, VoxelNet, and PointRCNN, and Painted PointPillars+ improves nuScenes mAP from 40.1 to 46.4. The same design exposes its ceiling: camera evidence reaches the detector only where LiDAR has a point and only in the vocabulary and confidence emitted by segmentation.

## Core Insights

### The LiDAR point is the carrier for camera semantics

PointPainting has three sequential stages. A semantic segmentation network turns each camera image into a class-score vector at every pixel. A calibration transform projects each LiDAR point into the relevant image, and the vector at that pixel is concatenated with the point's coordinates and intensity. A standard LiDAR detector then consumes the painted cloud. For KITTI, the segmentation output has four channels—car, pedestrian, cyclist, and background—so a PointPillars point grows from 9 to 13 dimensions; for nuScenes, the 10 detection classes plus background grow a PointPillars point from 7 to 18 dimensions.

![PointPainting source Figure 2: segmentation, painting, and LiDAR detection](/assets/images/pointpainting-sequential-fusion-for-3d-object-detection-paper-figure.webp)
*Fig 1: Image semantics are computed first, projected onto LiDAR points, and passed to a LiDAR detector as extra point features. The detector interface remains unchanged apart from its input dimension. | source: [PointPainting, Figure 2](https://arxiv.org/abs/1911.10150)*

The choice is intentionally modular. Segmentation is a local per-pixel task with its own useful output, while 3D detection keeps the depth and geometry of LiDAR. PointPainting avoids building a pseudo-point cloud from the image, avoids feature interpolation between full image and BEV grids, and does not impose a proposal architecture on the detector. If two camera fields of view overlap, the implementation randomly chooses one score vector for the point; the paper leaves entropy- or margin-based selection for later work.

The projection also defines the information boundary. A pixel can see an object between LiDAR returns, but the detector cannot use that semantic evidence if no point projects there. A calibration error paints a neighboring object or background. When images are unavailable, the LiDAR path can run only if the deployment system treats the semantic channels as optional or supplies a fallback.

### The result transfers across very different LiDAR detectors

On the KITTI moderate validation split, painting improves BEV mAP for all three tested detectors. PointPillars rises from 73.78 to 76.27 mAP, VoxelNet from 71.83 to 73.55, and PointRCNN from 72.42 to 75.80. On the KITTI test benchmark, Painted PointRCNN reaches 69.86 BEV mAP, with gains over the unpainted model for every reported class and difficulty in that table. These are detector-agnostic changes: PointPillars is a pillar encoder, VoxelNet uses voxel features, and PointRCNN operates on point-wise features and a second-stage region head.

![PointPainting source Figure 1: BEV improvements across KITTI and nuScenes detectors](/assets/images/pointpainting-sequential-fusion-for-3d-object-detection-source-figure-1.webp)
*Fig 2: The source results compare original and painted PointPillars, VoxelNet, and PointRCNN on KITTI, then show the PointPillars+ gain across nuScenes classes. Painting improves the validation BEV scores for every tested LiDAR detector. | source: [PointPainting, Figure 1](https://arxiv.org/abs/1911.10150)*

For nuScenes, the authors strengthen the PointPillars baseline before measuring fusion: 0.2 m pillars, a deeper early network, better attribute estimation, sample weighting, and revised yaw augmentation produce PointPillars+. The painted version reaches 46.4 mAP and 58.1 NDS versus 40.1 mAP and 55.0 NDS. Every one of the ten classes improves; traffic-cone AP rises by 16.8 points and bicycle AP by 10.1. Cars still gain 1.9 AP from a 76.0 baseline, while trailers and construction vehicles gain less because the segmentation network recalls them poorly—39% for trailers and 40% for construction vehicles in the reported nuScenes analysis.

### Segmentation quality becomes detection quality

The dependency experiment is the most direct causal evidence in the paper. As the segmentation network's mIoU increases, Painted PointPillars' detection mAP increases with it. An oracle that paints ground-truth 3D boxes adds about 27 mAP, which is an upper bound rather than a deployable result. It still does not reach perfect detection: a box can contain ground points, nuScenes annotates objects with only one LiDAR return, and PointPillars may randomly sample away the few points carrying the useful semantic signal.

![PointPainting source Figure 4: qualitative camera, painted cloud, and detections](/assets/images/pointpainting-sequential-fusion-for-3d-object-detection-source-figure-4.webp)
*Fig 3: Each qualitative comparison shows the original cloud, the painted cloud with class-colored points, and the resulting boxes in the cloud and camera view. The figure makes the carrier rule visible: semantics color measured points rather than filling the whole scene. | source: [PointPainting, Figure 4](https://arxiv.org/abs/1911.10150)*

The segmentation scores themselves are not the main bottleneck in the reported ablation. Replacing the score vector with one-hot labels leaves NDS unchanged and increases mAP by only 0.4, within the training noise. What matters is whether the class evidence is correct at the points that survive the LiDAR encoder.

### Sequential timing can be pipelined, but alignment remains part of the model

Using the most recent image forces the LiDAR detector to wait for image segmentation. The paper's consecutive-matching variant uses the previous image: it first transforms the current point cloud into the previous ego frame, projects it into that image, and then paints it. On the timing ablation, concurrent and consecutive matching both report 33.9 mAP; the latter adds only 0.75 ms over the original PointPillars encoder, including 0.15 ms for projection and 0.6 ms for encoding the 18-dimensional painted point cloud. The pipeline therefore hides most segmentation latency without degrading the reported detection score.

The price is temporal and geometric bookkeeping. Ego-motion compensation, timestamps, camera calibration, and segmentation quality all enter before the detector sees the point cloud. The method is especially informative for sparse small objects, but one return cannot become a well-localized oriented box merely because its semantic vector is correct.

## High-Level Takeaways

- PointPainting moves image semantics onto measured LiDAR points, preserving the downstream detector while avoiding a new joint fusion representation.
- The interface transfers across PointPillars, VoxelNet, and PointRCNN; Painted PointRCNN reaches 69.86 BEV mAP on the cited KITTI test benchmark.
- Painted PointPillars+ improves nuScenes from 40.1 to 46.4 mAP and from 55.0 to 58.1 NDS, with the largest class gain on traffic cones at +16.8 AP.
- Detection quality follows segmentation quality, while the 0.75 ms consecutive-matching overhead shows that sequential fusion can be pipelined when calibration and timestamps are reliable.
