---
title: 'CenterPoint: Center-Based 3D Detection and Tracking'
date: '2020-06-19T04:00:00.000Z'
section: paper-shorts
postSlug: centerpoint-center-based-3d-detection-and-tracking
legacyPath: /paper shorts/2020/06/19/centerpoint-center-based-3d-detection-and-tracking.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2020 – CenterPoint: represent 3D actors by BEV centers, attributes, and velocity'
---
## 2020 – CenterPoint

**arXiv:** [2006.11275](https://arxiv.org/abs/2006.11275)

**Code:** [tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint)

## Summary

> CenterPoint changes the detector's output address. Instead of placing many oriented anchors over a BEV map, it finds one class-specific center per object and reads size, height, yaw, sub-voxel offset, and velocity at that location. A light second stage samples five geometric points on each proposal to recover detail that a strided center feature may miss. The same velocity output turns detection into greedy tracking. A single model reaches 58.0 mAP/65.5 NDS on nuScenes and 63.8 AMOTA, while the ablations show where the center abstraction helps and where sparse returns limit refinement.

## Core Insights

### A center removes orientation from proposal generation

CenterPoint first encodes a LiDAR cloud with a standard VoxelNet or PointPillars backbone and flattens the result into a BEV feature map. Its head predicts a class-specific heatmap whose local peaks are object centers. During training, each annotated center is rendered as a Gaussian target; the smallest radius is two map cells so supervision is less sparse than a single positive pixel. At a peak, separate regression heads read a two-dimensional sub-voxel offset, height above ground, logarithmic 3D size, $(\sin\alpha,\cos\alpha)$ for yaw, and planar velocity.

The representation is useful because a point has no intrinsic orientation. An anchor detector must decide which sizes and headings to place at every location before it can learn to correct them. CenterPoint asks the backbone to learn the relationship between a center feature and the object's extent and heading. This is especially relevant during a turn: an axis-aligned anchor is a poor geometric proxy for a rotated car, while the center remains the same kind of target.

![CenterPoint source Figure 1: anchor-based and center-based boxes during straight driving and a turn](/assets/images/centerpoint-center-based-3d-detection-and-tracking-source-figure-1.webp)
*Fig 1: The source comparison shows axis-aligned anchor boxes struggling when the vehicle turns, while center points remain rotationally invariant and the box attributes are regressed afterward. | source: [CenterPoint, Figure 1](https://arxiv.org/abs/2006.11275)*

The gain is not just a different parameterization. On Waymo validation, changing the output from anchors to centers improves level-2 mAPH by 4.3 points with a VoxelNet encoder and 4.5 points with PointPillars. On nuScenes validation, the corresponding mAP improvement is 3.8–4.1 points, with 1.1–1.8 points of NDS improvement. These comparisons keep the broad encoder and training setup fixed, so they isolate the proposal representation more closely than a leaderboard comparison does.

### Five samples repair the center feature's blind spot

The first stage reads every box property from the feature at the predicted center. That is efficient, but a LiDAR sensor may see only the side of an object; the center can be empty or poorly represented. CenterPoint's second stage keeps the first-stage box and samples backbone features at the predicted 3D center plus the four outward-facing box-face centers. The top and bottom face centers project to the same BEV location, so they are omitted. Bilinear interpolation gathers the five features, an MLP predicts an IoU-guided class-agnostic confidence and a box refinement, and inference runs this stage on the top 500 post-NMS proposals.

On Waymo, this targeted readout adds about 2 mAP with less than 10% extra computation in the headline comparison; the detailed ablation reports roughly 6–7 ms of overhead. It is cheaper than pooling a dense 3D region because the model only extracts features around an existing hypothesis. The boundary is visible in the data: the two-stage model helps on the denser Waymo scans, but it does not improve the single-stage model on nuScenes, whose 32-beam LiDAR contributes about 30,000 points per frame, roughly one-sixth of Waymo's point count. PointPillars pedestrians can collapse to roughly one input pixel, leaving little local detail for the refinement stage to recover.

### Velocity makes the same point a track state

CenterPoint predicts planar velocity from the current and previous BEV views. At inference, it moves each current center backward by the negative velocity estimate and greedily matches the result to existing tracks by closest distance. Unmatched tracks survive for three frames and retain their last velocity. This avoids a separate Kalman filter or appearance embedding; the association costs about 1 ms on top of detection in the nuScenes ablation, compared with 73 ms for the Mahalanobis-distance Kalman baseline under the paper's measurement.

The shortcut has a clear interpretation. If the velocity is good, a tracklet is just a path of points through time. If two objects cross, detections are delayed, or the velocity is wrong, closest-point matching has no appearance or global assignment signal to correct the identity. The representation simplifies tracking by accepting that failure mode.

### The benchmark gains survive the representation boundary

CenterPoint-Voxel reaches 71.8 level-2 mAPH for vehicles and 66.4 for pedestrians on the Waymo test set. On nuScenes test, a single model reports 58.0 mAP and 65.5 NDS; its tracking result is 63.8 AMOTA, 8.8 points above the cited previous state of the art. The system runs at about 11 FPS on Waymo and 16 FPS on nuScenes in the paper's setup. These are single-model results; the later nuScenes challenge appendix adds PointPainting, test-time augmentation, and a five-model ensemble, so those challenge numbers should not be confused with the base detector.

![CenterPoint source Figure 3: qualitative Waymo detections](/assets/images/centerpoint-center-based-3d-detection-and-tracking-source-figure-3.webp)
*Fig 2: The source qualitative panel shows the raw point cloud in blue, predicted boxes in green, and LiDAR points inside predicted boxes in red on Waymo validation. It is a detection visualization, not a camera image. | source: [CenterPoint, Figure 3](https://arxiv.org/abs/2006.11275)*

## High-Level Takeaways

- The center heatmap removes orientation-specific anchors; size, height, yaw, and velocity are decoded at the center feature.
- Center-to-anchor ablations improve 3D detection by 3.8–4.5 mAP or mAPH across the reported nuScenes and Waymo settings.
- Sampling the predicted center and four outward-facing box faces adds about 2 mAP on dense Waymo scans, but not on the sparser nuScenes setup.
- Velocity plus greedy closest-center matching reaches 63.8 AMOTA with a 1 ms tracker in the reported nuScenes ablation, while identity ambiguity remains the cost of the simple association rule.
