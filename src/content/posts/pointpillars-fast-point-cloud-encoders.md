---
title: 'PointPillars: Fast Point Cloud Encoders'
date: '2018-12-14T05:00:00.000Z'
section: paper-shorts
postSlug: pointpillars-fast-point-cloud-encoders
legacyPath: /paper shorts/2018/12/14/pointpillars-fast-point-cloud-encoders.html
tags: [Other]
field: 'BEV Perception & Mapping'
summary: '2018 – PointPillars: collapse height early and run LiDAR perception as 2D convolution'
---
## 2018 – PointPillars

**arXiv:** [1812.05784](https://arxiv.org/abs/1812.05784)

## Summary

> PointPillars makes a single compression decision: summarize every vertical column before the expensive backbone. A learned PointNet-style encoder preserves point geometry inside each pillar, then scatters the result into a dense BEV pseudo-image that ordinary 2D convolution can process. On KITTI this reaches 62 Hz with LiDAR alone, but the speed claim is tied to a front-camera crop that contains only about 10% of the available point cloud.

## Core Insights

### The pillar is an early height-compression contract

PointPillars discretizes only the ground plane. A point with coordinates and reflectance $[x,y,z,r]$ receives five decorations: its offsets from the mean point in the pillar, $x_c,y_c,z_c$, and its offsets from the pillar center, $x_p,y_p$. The encoder therefore sees a nine-dimensional point vector. The extra center offsets preserve where a point sits inside the cell after the height axis has been removed.

At the paper's default $0.16^2$ m ground-plane resolution, a KITTI frame contains roughly 6,000–9,000 non-empty pillars and is about 97% sparse. The implementation caps each example at 12,000 pillars and each pillar at 100 points, padding or randomly sampling to form a fixed tensor. A shared linear layer, batch normalization, and ReLU map each point to a feature; max pooling over the points in a pillar produces one vector, which is scattered back to its original $(x,y)$ location.

![PointPillars network converting pillars into a pseudo-image before a 2D backbone and detection head](/assets/images/pointpillars-fast-point-cloud-encoders-paper-figure.webp)
*Fig 1: The network groups points into pillars, learns one feature per non-empty pillar, scatters those features into a BEV pseudo-image, and predicts oriented boxes with a 2D backbone and SSD head. | source: [PointPillars, Figure 2](https://arxiv.org/abs/1812.05784)*

The height axis is gone before the backbone starts. That is the reason the method is fast: all later spatial computation is a standard 2D operation. It is also the representation boundary. A pillar feature can learn useful vertical statistics, but a later camera or radar branch cannot recover which height-specific structure was merged into that vector.

### The speed comes from the whole memory path

The backbone has three top-down blocks at strides 1, 2, and 4. Their four, six, and six convolutional layers use progressively wider channel counts, and transposed convolutions upsample the three outputs to a common stride before concatenation. An SSD-style head then predicts the same oriented-box targets and direction classes used by contemporaneous voxel detectors. The architecture is deliberately ordinary after the encoder; the innovation is making the input look like a 2D feature map without hand-coded height slices.

The paper's runtime breakdown makes that handoff concrete on its desktop CPU/GPU setup: 1.4 ms to load and filter points, 2.7 ms to organize and decorate them, 2.9 ms to upload the pillar tensor, 1.3 ms for encoding, 0.1 ms to scatter, 7.7 ms for the backbone and head, and 0.1 ms for CPU NMS, for 16.2 ms total. The reported 62 Hz headline is therefore an end-to-end number, not just the neural network forward pass.

![PointPillars speed versus BEV accuracy on KITTI](/assets/images/pointpillars-fast-point-cloud-encoders-source-figure-1.webp)
*Fig 2: The KITTI test-set plot compares LiDAR-only methods (blue circles) with LiDAR-and-vision methods (red squares) on BEV accuracy and speed. | source: [PointPillars, Figure 1](https://arxiv.org/abs/1812.05784)*

The plot shows why the 2D layout matters: PointPillars occupies a favorable speed–accuracy region among LiDAR-only methods and competes with several fusion systems. The paper also reports a PyTorch pipeline at 42.4 Hz and a 45.5% speedup after replacing the encoder, backbone, and head kernels with TensorRT. The implementation, host transfer, NMS, and GPU generation are all part of the number.

### KITTI accuracy isolates the learned encoder, with a dataset caveat

On the authors' KITTI validation split, the moderate BEV AP for cars, pedestrians, and cyclists is 87.98, 63.55, and 69.71; the corresponding moderate 3D AP is 77.98, 57.86, and 66.02. The paper reports state-of-the-art results on both BEV and 3D test benchmarks using only LiDAR, including comparisons against methods that use images. The source Figure 1 plot is a **test-set** comparison, while these numbers are validation values; the two contexts should not be conflated.

The encoder ablation keeps the downstream network and training procedure fixed. Learned encoders outperform fixed statistics across resolutions, and the gap grows when pillars become larger because a fixed summary has less capacity to describe the more varied points inside a cell. VoxelNet is marginally stronger in the matched encoder table, but it is orders of magnitude slower and has many more parameters. PointPillars wins the useful operating point by changing the representation before the backbone.

Two smaller interventions explain some of the gain. Adding the $x_p,y_p$ center decorations contributes about 0.5 mAP. Minimal per-box augmentation works better than the heavier augmentation recommended by earlier voxel methods, especially for pedestrians; the authors hypothesize that ground-truth sampling already supplies the needed variation.

### The failure cases show what the pillar has thrown away

The paper's failure panel pairs the BEV point cloud with boxes projected into the camera view.

![PointPillars failure cases on KITTI](/assets/images/pointpillars-fast-point-cloud-encoders-source-figure-4.webp)
*Fig 3: KITTI failure cases are shown in BEV and projected views, exposing misses and localization errors after the point cloud has been summarized into vertical pillars. | source: [PointPillars, Figure 4](https://arxiv.org/abs/1812.05784)*

The examples should be read with the tokenizer in mind. Returns from objects separated in height can share one pillar, distant objects contribute fewer points, and the fixed point cap can discard crowded local evidence. Those limits are less visible in the speed curve than in the failure examples. They also explain why the paper's “105 Hz” operating point, obtained with $0.28^2$ m pillars, loses most of its accuracy on pedestrians and cyclists while car performance stays relatively stable.

The authors warn that KITTI timing is optimistic for deployment: the benchmark annotations encourage using only points that project into the front image, approximately 10% of the full cloud. A surround-view system would process more points, and an embedded accelerator would not necessarily match the desktop GPU.

## High-Level Takeaways

- PointPillars learns a nine-dimensional point decoration and pools each vertical column into one BEV token.
- The early height collapse enables a dense 2D backbone and a 62 Hz end-to-end KITTI result.
- Learned point encoders, center offsets, and light augmentation explain the accuracy operating point; the pillar itself is not a free representation.
- The 105 Hz setting mainly preserves car performance and exposes the cost of small-object evidence.
- KITTI's front-camera point filtering and desktop timing make the headline throughput an upper-bound reference for full-scene deployment.
