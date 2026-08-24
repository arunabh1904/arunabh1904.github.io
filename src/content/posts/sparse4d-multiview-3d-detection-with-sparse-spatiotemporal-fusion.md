---
title: 'Sparse4D: Multi-View 3D Object Detection with Sparse Spatial-Temporal Fusion'
date: '2022-11-19T00:00:00.000Z'
section: paper-shorts
postSlug: sparse4d-multiview-3d-detection-with-sparse-spatiotemporal-fusion
legacyPath: /paper shorts/2022/11/19/sparse4d-multiview-3d-detection-with-sparse-spatiotemporal-fusion.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – Sparse4D: Multi-View 3D Object Detection with Sparse Spatial-Temporal Fusion'
---
## 2022 – Sparse4D

**arXiv:** [2211.10581](https://arxiv.org/abs/2211.10581)

**Code:** [linxuewu/Sparse4D](https://github.com/linxuewu/Sparse4D)

### Method and reported result

Sparse4D avoids constructing a dense BEV video. Each learnable 3D anchor carries an instance feature and several keypoints. The model projects those keypoints into multiple cameras, feature scales, and timestamps; samples only the corresponding image features; fuses them hierarchically; and iteratively refines the anchor box.

## Summary

> “4D” here means sparse 3D hypotheses extended through time. The model spends temporal compute where objects may exist rather than aligning every BEV cell across every frame.

## Core Insights

Sparse 4D sampling starts with fixed keypoints at the anchor's center and face centers, plus learned offsets conditioned on the instance feature. Ego pose and the anchor's estimated velocity move those points across timestamps before camera projection. Hierarchical fusion then aggregates scale and camera, keypoint, and time dimensions in stages. This factorization keeps the attention problem local and makes the contribution of each axis inspectable.

The depth-reweight module addresses an ambiguity created by projecting a 3D anchor into images: an image feature can match appearance while belonging to the wrong depth along the ray. The module predicts an instance-level depth distribution and scales the sampled feature by confidence at the anchor depth. On nuScenes, the paper reports that Sparse4D surpasses earlier sparse camera detectors and most compared BEV methods while avoiding dense view transformation and global attention.

![Figure 2 from Sparse4D, showing iterative 3D anchors sampling multi-view, multi-scale, multi-timestamp image features](/assets/images/sparse4d-paper-figure-2.png)
*Sparse4D turns space-time fusion into sparse evidence retrieval around a fixed set of evolving anchors. source: [Sparse4D](https://arxiv.org/abs/2211.10581)*

![Figure 1 from Sparse4D: Multi-View 3D Object Detection with Sparse Spatial-Temporal Fusion](/assets/images/sparse4d-multiview-3d-detection-with-sparse-spatiotemporal-fusion-source-figure-1.webp)
*Figure 1 Overview of the Sparse4D. For each candidate anchor instance, we sparsely sampling multi-timestamp/view/scale features of multiple keypoints, then hierarchically fuse these feature as instance feature for precise anchor refinement. source: [Sparse4D: Multi-View 3D Object Detection with Sparse Spatial-Temporal Fusion](https://arxiv.org/abs/2211.10581)*

![Figure 5 from Sparse4D: Multi-View 3D Object Detection with Sparse Spatial-Temporal Fusion](/assets/images/sparse4d-multiview-3d-detection-with-sparse-spatiotemporal-fusion-source-figure-5.webp)
*Figure 5 Ablation study of the influence of Refinement Modules and Historical Frame. In this experiment, the input image size is set to and learnable keypoints are removed. source: [Sparse4D: Multi-View 3D Object Detection with Sparse Spatial-Temporal Fusion](https://arxiv.org/abs/2211.10581)*


| Sparse state | What it carries | Failure pressure |
| --- | --- | --- |
| 3D anchor | Position, extent, orientation, and velocity | A bad initialization samples the wrong image regions. |
| Instance feature | Appearance and historical evidence | Can preserve stale or duplicate identity. |
| 4D keypoints | Local support across views and frames | Depend on calibration, ego pose, and motion estimates. |
| Depth confidence | Compatibility along the camera ray | Cannot recover an object never covered by an anchor. |

## High-Level Takeaways

- Sparse4D informs whether temporal perception should store a dense scene field or a set of persistent object hypotheses. Its atomic unit is an anchor-instance pair. The image backbone is shared across cameras and frames; the decoder shares a sampling and refinement protocol across instances, but the state is object-specific.
- The missing matched control compares dense BEV memory and sparse anchors with the same camera backbone, history length, image tokens, and P99 latency. At 10× crowded-scene density, anchor count, self-attention, and duplicate suppression erode the sparse advantage. The method would fail for unmodeled free space, thin map elements, or novel objects if those outputs require a dense field or a much larger proposal set.
- Sparse4D establishes the anchor-sampling branch of camera-only temporal perception; StreamPETR carries object queries recurrently, while Sparse4D v3 strengthens training and turns the same state into tracks.
- Temporal fusion can follow objects through space-time instead of carrying the entire scene grid forward.
