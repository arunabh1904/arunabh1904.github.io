---
title: 'BEVDepth: Acquisition of Reliable Depth for Multi-View 3D Object Detection'
date: '2022-06-21T00:00:00.000Z'
section: paper-shorts
postSlug: bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection
legacyPath: /paper shorts/2022/06/21/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – BEVDepth: Acquisition of Reliable Depth for Multi-View 3D Object Detection'
---
## 2022 – BEVDepth

**arXiv:** [2206.10092](https://arxiv.org/abs/2206.10092)

**Code:** [Megvii-BaseDetection/BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

## Summary

> BEVDepth treats the depth distribution inside Lift-Splat as a trainable geometric interface rather than an incidental attention map. Projected LiDAR points supervise that intermediate distribution during training, camera intrinsics and extrinsics condition its prediction, and a depth-refinement module can move or aggregate features along the ray before BEV pooling. Runtime input is still images plus calibration. The paper’s controlled oracle-depth gap and component ablation show why this helps; the 60.0/60.9 NDS test results add multi-frame fusion, efficient pooling, and different backbone contracts, so they should not be read as one isolated depth-loss gain.

## Core Insights

### The view transform is the hidden bottleneck

BEVDepth begins by looking inside a Lift-Splat-style detector. Its learned depth maps appear poor even when the detector reaches reasonable mAP, so the paper replaces the predicted distribution with alternatives. On the nuScenes validation split, the controlled detector goes from 0.282 mAP, 0.768 mATE, and 0.327 NDS with learned depth to 0.470, 0.393, and 0.515 with ground-truth depth from LiDAR. Freezing a random soft depth tensor throughout training and testing still reaches 0.245 mAP, while a one-hot random tensor falls to 0.176. A soft distribution can leave some activation at the correct range and spread nearby noise; a one-hot guess either lands on the object or misses it entirely.

The depth metrics tell the same story. Across all foreground points, adding the explicit depth loss changes SILog from 54.58 to 27.62, AbsRel from 3.03 to 0.23, SqRel from 85.11 to 2.09, and RMSE from 19.45 to 5.78. When evaluating the best-predicted pixel for each object, the corresponding values are 27.87/0.38/6.96/8.29 without the loss and 14.12/0.10/1.04/4.55 with it. This is why a detector can survive a visually unconvincing depth map: the downstream head needs enough correctly placed or softly overlapping evidence, not uniformly accurate depth at every pixel.

The base detector predicts image features $F_i^{2d}$ and a categorical depth tensor $D_i^{pred}$, forms a frustum feature $F_i^{3d}=F_i^{2d}\otimes D_i^{pred}$, and pools those features into BEV. BEVDepth adds supervision before the final detector can compensate for a bad placement.

![Figure 1 from BEVDepth: Acquisition of Reliable Depth for Multi-View 3D Object Detection](/assets/images/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection-source-figure-1.webp)
*Fig 1: The source visualization contrasts Lift-Splat and BEVDepth depth maps; the dashed regions mark places where the baseline happens to predict useful depth even though much of its map is unreliable. | source: [BEVDepth, Figure 1](https://arxiv.org/abs/2206.10092)*

### LiDAR teaches the distribution, then leaves the vehicle

To make a target, BEVDepth projects each LiDAR point into each camera with its extrinsic rotation and translation and the camera intrinsic matrix. Points outside the view are discarded. Multiple points that land at one pixel are min-pooled in depth, then converted to a one-hot depth-bin target. A binary cross-entropy loss supervises the predicted categorical distribution. The LiDAR points therefore provide a sparse geometric teacher; they are not concatenated with the camera features at inference.

That lifecycle is easy to blur in the headline. The deployed view transform still has to infer range from images, and its target covers measured surfaces rather than every pixel. Occlusion boundaries, moving objects, point sparsity, and camera changes can all make the teacher incomplete. The contribution is to make the intermediate representation answer a geometric question during training, not to turn the runtime camera into a LiDAR sensor.

### Camera parameters tell DepthNet what a pixel means

A pixel’s depth statistics depend on the camera that produced it. BEVDepth concatenates flattened intrinsics, rotation, and translation parameters, maps them through an MLP, and uses the resulting conditioning vector to reweight image features with a squeeze-and-excitation block. DepthNet therefore receives camera-conditioned features carrying both projection characteristics and camera pose in the ego frame. The target remains the same categorical depth representation; camera awareness is inside DepthNet rather than a camera-specific regression rescaling.

This is a useful distinction for multi-camera systems with different fields of view. A shared DepthNet can learn that the same visual pattern carries a different geometric prior in a wide-angle side camera than in a forward camera, while the output still lands in the common BEV coordinate system.

### Refinement repairs a soft placement before pooling

Even a supervised distribution can assign a feature to a nearby depth bin. BEVDepth’s refinement module reshapes $F^{3d}$ from $[C_F,C_D,H,W]$ to $[C_FH,C_D,W]$ and applies convolutions on the $C_D\times W$ plane before reshaping back for voxel or pillar pooling. A kernel that spans the depth axis can aggregate neighboring hypotheses when confidence is low and can move an incorrectly placed feature toward a nearby consistent location. It is a rectification step on the lifted feature volume, not a second standalone monocular depth estimator.

The kernel ablation makes that intuition testable. A $1\times3$ kernel, which does not mix along depth, gives 0.315 mAP and 0.357 NDS; a $3\times1$ kernel reaches 0.320 and 0.369; a $3\times3$ kernel reaches 0.322 and 0.367. The depth-axis interaction is doing the useful work, while the final two-dimensional kernel only adds context around it.

![BEVDepth pipeline with LiDAR depth supervision during training and a camera-only inference path](/assets/images/bevdepth-acquisition-of-reliable-depth-for-multiview-3d-detection-source-figure-4.webp)
*Fig 2: Camera parameters condition DepthNet, projected LiDAR supplies the training target, and the lifted frustum is refined and pooled into BEV; the LiDAR supervision path disappears at inference. | source: [BEVDepth, Figure 4](https://arxiv.org/abs/2206.10092)*

### The ablation separates geometry from temporal and systems gains

The component study trains a ResNet-50 model for 24 epochs at 256×704 input without class-balanced group sampling. On the nuScenes validation split, the sequence starts at 0.282 mAP and 0.327 NDS. Adding depth loss gives 0.304/0.344; adding camera awareness gives 0.314/0.357; adding depth refinement gives 0.322/0.367; and multi-frame fusion gives 0.330/0.442. The source attributes the main depth-loss gain to classification, the camera-aware step to a reduction in mATE from 0.747 to 0.706 metres (0.041 metres), and refinement to a 0.8-point mAP improvement. BCE and L1 depth losses are close (0.322/0.367 versus 0.321/0.371), so the supervision target and feature path matter more than choosing between these two scalar losses.

The final multi-frame implementation aligns frustum coordinates from different frames into the current ego frame, pools them, and concatenates the resulting BEV features. Efficient Voxel Pooling assigns a CUDA thread to each frustum feature instead of sorting and cumulative-summing the whole set. The paper reports an 80× speedup for the pooling operation and reduces state-of-the-art training time from five days to 1.5 days. These are important engineering additions, but they are separate from the controlled evidence that depth supervision fixes the view transform.

On the test split, the submitted BEVDepth model reports 50.3 mAP and 60.0 NDS with a VovNet backbone and $640\times1600$ input. A ConvNeXT variant reaches 52.0 mAP and 60.9 NDS. The test submission trains on both the training and validation splits and uses test-time augmentation. The latter score is the paper’s headline camera-only result, but its backbone and full system differ from the earlier ResNet-50 ablation. Keeping those contexts separate makes the causal claim legible.

## High-Level Takeaways

- BEVDepth’s central diagnosis is the oracle-depth gap: the camera detector has downstream capacity, but inaccurate lifting places evidence at the wrong range.
- Projected LiDAR points supervise the categorical depth distribution during training; runtime still uses only images and calibration, so this is privileged supervision rather than sensor fusion.
- Camera-aware DepthNet encodes intrinsics and extrinsics inside the predictor, while the refinement module mixes neighboring depth hypotheses before pooling.
- The component ablation reaches 0.330 mAP / 0.442 NDS only after adding multi-frame fusion; the clean depth evidence is the earlier 0.282→0.322 mAP progression.
- The test headline is 60.0 NDS for the VovNet system and 60.9 for the ConvNeXT variant. Sparse teacher coverage, calibration shifts, and camera-only range ambiguity remain deployment boundaries.
