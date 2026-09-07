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

## Summary

> Sparse4D replaces a dense scene grid with a set of 3D boxes that retrieve their own image evidence across space and time. Each box carries an instance feature, fixed and learned sampling points, and an estimated velocity used to look backward through the camera history. On nuScenes validation, the four-frame model reaches 43.6 mAP and 54.1 NDS; its strongest reported test configuration reaches 51.1 mAP and 59.5 NDS. The informative result is how those gains arise: motion alignment improves temporal evidence, while depth reweighting checks whether a visually plausible match belongs at the proposed distance.

## Core Insights

### A box predicts where its evidence should have been

DETR3D samples around one projected reference point. Sparse4D gives each hypothesis spatial extent: seven fixed points cover the box center and six face centers, and six learned points move within its rotated dimensions. A learned offset is conditioned on the instance feature, scaled by box size, rotated by yaw, and translated to the proposed center. The default decoder refines 900 hypotheses through six modules, so sampling locations change as the boxes become more accurate.

![Sparse4D follows an object through past camera views](/assets/images/sparse4d-multiview-3d-detection-with-sparse-spatiotemporal-fusion-source-figure-1.webp)
*Fig 1: Sampling points follow the hypothesized object through earlier views, then their evidence is fused into the feature used to refine its current box. The alignment depends on both ego motion and the estimated object velocity. | source: [Sparse4D, Figure 1](https://arxiv.org/abs/2211.10581)*

To find a point in an earlier frame, the model first subtracts the object's estimated displacement under constant velocity, then transforms the result into that frame's ego coordinate system. Camera calibration projects it into each image. This separates two motions that can otherwise look similar: a stationary parked car moves across the image when the ego car moves, while another vehicle also changes position in the world. The velocity estimate need not be perfect at the first layer because later refinement updates both the box and the locations queried from history.

For one hypothesis, the sampled evidence has axes for keypoint, timestamp, camera, feature scale, and channel. Fusion follows a specific order. Predicted group-wise weights first combine cameras and scales for each point and timestamp. A sequential concatenation-and-linear update then combines timestamps, and summation combines the keypoints. This is feature fusion over a finite image-history window; it should not be confused with the persistent instance-state recurrence introduced in Sparse4D v2.

### Depth reweighting challenges an appearance match

Projecting a box into a camera does not verify its depth. Two boxes at different distances along a ray can retrieve nearly the same image evidence. Sparse4D predicts a discrete depth distribution from the aggregated instance feature, samples confidence at the anchor's radial distance, and multiplies that confidence into the feature before further refinement. An attractive image match is therefore weakened if its proposed distance disagrees with the model's depth estimate.

![Sparse4D architecture with temporal feature queue and depth reweighting](/assets/images/sparse4d-paper-figure-2.png)
*Fig 2: Each refinement module combines inter-instance self-attention, sparse 4D aggregation, depth reweighting, and box updates. The image-feature queue retains past observations; the depth module tests the geometric plausibility of the evidence gathered around each anchor. | source: [Sparse4D, Figure 2](https://arxiv.org/abs/2211.10581)*

Depth supervision comes from the centers of matched, annotated 3D boxes. The method does not require an additional dense LiDAR depth target, but it still requires 3D box supervision. In the component ablation, neither learned points nor depth reweighting alone improves mAP at the displayed precision: the baseline is 43.2, depth reweighting gives 43.1, and learned points give 43.2. Together they reach 43.6, while NDS rises from 53.3 to 54.1 and orientation error falls from 0.408 to 0.363. The contribution is modest and complementary, rather than an isolated depth module explaining the whole temporal gain.

### Temporal alignment helps velocity more reliably than every metric

The motion-compensation ablation uses 320×800 images, no learned keypoints, and three historical frames. Its component scores make the trade-off visible:

| Temporal evidence | mAP ↑ | NDS ↑ | Velocity error ↓ |
| --- | ---: | ---: | ---: |
| Current frame only | 32.2 | 40.1 | 0.890 m/s |
| History, without motion compensation | 33.4 | 42.4 | 0.682 m/s |
| History aligned for ego motion | 37.6 | 48.8 | 0.398 m/s |
| Ego and object motion compensation | 37.3 | 49.5 | 0.329 m/s |

Object compensation lowers velocity error after ego alignment, while mAP slips slightly. This matters because NDS rewards several properties of a box: a higher aggregate score need not mean more objects were correctly detected. The largest retrieval improvement here comes from putting past observations into the correct ego frame.

![Sparse4D refinement and history ablations](/assets/images/sparse4d-multiview-3d-detection-with-sparse-spatiotemporal-fusion-source-figure-5.webp)
*Fig 3: Left: successive outputs from one six-module decoder. Middle: separately trained decoder depths. Right: history length at 320×800 without learned points. These are three different experiments, and their curves show diminishing returns rather than one interchangeable depth-or-time budget. | source: [Sparse4D, Figure 5](https://arxiv.org/abs/2211.10581)*

The history curve rises from 40.1 NDS without history to 49.5 with three historical frames and 51.0 with ten. The longer window still helps, but much less at its margin. Memory constraints prevented the authors from testing further; this first version must retain and sample past image features. The depth panel also distinguishes mAP from NDS: ten modules give the displayed peak mAP of 38.1, whereas the highest displayed NDS is 50.0 at twelve modules.

The main four-frame ResNet-101 validation model uses 640×1600 images. Moving to nine frames raises mAP/NDS from 43.6/54.1 to 44.5/54.7; a 48-epoch variant reaches 44.4/55.0. The strongest test row uses DD3D-pretrained VoVNet-99, nine frames with six historical fusion features randomly detached during training, and 48 epochs to reach 51.1/59.5, without CBGS or test-time augmentation. Those configurations should not be substituted for the smaller motion ablation. Likewise, the separate 900×1600 compute comparison reports 1019.2→1113.8 GFLOPs from one to four frames, not a measured end-to-end latency guarantee.

## High-Level Takeaways

- Sparse4D retrieves evidence around evolving boxes instead of reconstructing a dense BEV history. Fixed points provide coverage; learned points adapt that coverage to each hypothesis.
- Ego alignment provides the largest gain in the motion ablation. Object compensation then improves velocity accuracy while slightly reducing mAP.
- Depth reweighting checks the distance of a sampled appearance match, using sparse 3D-box targets rather than an additional dense depth map.
- Longer history helps with diminishing returns and a growing image-feature memory burden. Sparse4D v2 addresses that remaining cost by carrying instance state recurrently.
- The original tracking extension still uses a learned association matrix and Hungarian matching; persistent detector state alone does not make this version an association-free tracker.
