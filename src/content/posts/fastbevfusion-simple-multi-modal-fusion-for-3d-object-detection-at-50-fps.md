---
title: 'FastBEVFusion: Simple Multi-Modal Fusion for 3D Object Detection at 50 FPS'
date: '2026-10-27T09:00:00.000Z'
section: paper-shorts
postSlug: fastbevfusion-simple-multi-modal-fusion-for-3d-object-detection-at-50-fps
legacyPath: >-
  /paper
  shorts/2026/10/27/fastbevfusion-simple-multi-modal-fusion-for-3d-object-detection-at-50-fps.html
tags:
  - Autonomous Driving
  - Perception
field: BEV Perception & Mapping
summary: >-
  2026 – FastBEVFusion: Simple Multi-Modal Fusion for 3D Object Detection at 50
  FPS
---
## 2026 – FastBEVFusion: Simple Multi-Modal Fusion for 3D Object Detection at 50 FPS

**Paper:** [arXiv:2610.37153](https://arxiv.org/abs/2610.37153) · [Full text, v1](https://arxiv.org/html/2610.37153v1)

## Summary

> FastBEVFusion trades precise camera depth estimation for a cheap image-to-BEV projection and one learned fusion block. On nuScenes validation, its ResNet-18 version reaches 62.1 mAP and 65.9 NDS at 17 ms per frame on an RTX 4090, without TensorRT. That makes it a useful candidate when perception latency matters more than the highest detection score. The cross-paper speed comparisons use different hardware, and larger fusion systems still detect better.

## Core Insights

### Use LiDAR to resolve an ambiguous camera projection

A camera feature gives a viewing direction but does not directly give depth. Conventional BEV fusion estimates depth to place that feature into a shared bird's-eye-view grid. FastBEVFusion spreads the feature along its entire camera ray using a uniform depth distribution, then collapses the resulting volume into image BEV. The direction is aligned with the LiDAR grid; the distance remains ambiguous.

That ambiguity changes the fusion problem. Adding the two grids would mix a LiDAR return with image features from several possible depths. Instead, PointPillars encodes the point cloud with 0.2 m pillars, while a ResNet encodes six camera views. Both BEV representations are downsampled and flattened to 4,096 tokens. One cross-attention block uses LiDAR tokens as queries and image tokens as keys and values. A query can gather semantic evidence from several locations rather than accepting one estimated depth.

The source overview makes the division of work visible: geometry enters through the LiDAR branch, while the camera branch supplies appearance without a depth network.

![FastBEVFusion source Figure 2 showing camera features spread into image BEV and fused with PointPillars LiDAR features](/assets/images/fastbevfusion-source-figure-2.webp)
*Fig 1: Uniform-depth projection keeps camera processing inexpensive; the learned fusion module resolves associations before the 3D detection head. The auxiliary camera detection head is used only during training. | source: [FastBEVFusion, Figure 2](https://arxiv.org/html/2610.37153v1#S2.F2)*

[View the figure at full resolution](/assets/images/fastbevfusion-source-figure-2.webp).

After attention, an MLP and upsampling restore the spatial representation, with a residual connection from the original LiDAR BEV. A TransFusion detection head then predicts 3D objects. This preserves LiDAR information even when the image association is weak. Relative to [BEVFusion](</paper shorts/2022/05/26/bevfusion-multi-task-multi-sensor-unified-bev.html>)'s aligned-grid fusion, the key change is where depth uncertainty is handled: feature association absorbs it instead of a camera depth estimator resolving it first.

### Train detection before training calibration tolerance

Training has three stages. First, the LiDAR encoder and detector train for 20 epochs, with copy-paste augmentation for the first 15. Cross-balanced sampling addresses class imbalance; flips, rotation, translation, and scaling provide geometric augmentation. The loss follows the TransFusion detection head.

Second, the image encoder and fusion module join for another 20 epochs. The image encoder starts from nuImages pretraining. An auxiliary 2D detector receives labels projected from the 3D ground truth, encouraging image features to retain object information. Copy-paste is omitted during joint fusion training. Finally, one epoch perturbs camera extrinsics with Gaussian noise while training only the fusion module; the remaining system stays fixed.

The last stage targets a specific failure mechanism. It teaches association to tolerate calibration error rather than asking the image or LiDAR encoders to relearn their representations. The paper does not fully specify the optimizer, image resolution, or Gaussian perturbation scale in its training description, which limits exact reproduction from the manuscript alone.

### The speed claim belongs to the smaller encoder

The following results are from nuScenes validation, with latency measured on an RTX 4090 without deployment libraries. They show the cost of changing only the image backbone within this system.

| Image backbone | mAP | NDS | Latency | Throughput |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 | 62.1 | 65.9 | 17 ms | 58.8 FPS |
| ResNet-50 | 63.4 | 66.8 | 41 ms | 24.4 FPS |

The smaller model gives up 0.9 NDS for a large latency reduction. Its component profile assigns 6.5 ms to the image backbone, 4.7 ms to LiDAR encoding, 2.8 ms to fusion, and 3.1 ms to detection. Encoder cost therefore remains the main constraint; simplifying fusion alone would leave most of the runtime intact.

The ablations also explain why the cheap projection needs learned association. Within the same semi-aligned system, replacing attention with concatenation reduces mAP from 63.4 to 31.1; addition falls to 15.7. Those results do not invalidate concatenation after accurate geometric alignment. They show that concatenation cannot repair the depth ambiguity introduced here. Replacing the TransFusion head with CenterPoint also lowers mAP to 56.2, so the reported detector quality is not attributable to fusion alone.

The comparison has two boundaries. Larger methods such as BEVFusion and SparseLIF report higher detection scores, and their latency figures come from other GPUs or estimated hardware conversions. The paper also contains a metric-order inconsistency in its calibration table: the clean 66.8/63.4 pair reverses the NDS/mAP order established in its main table. The qualitative robustness conclusion is plausible, but those columns should be checked against released evaluation outputs before quoting exact calibration-error deltas.

A useful deployment test would rerun the small and large variants, plus the existing production detector, on the same target hardware and calibration perturbations. If the camera benefit disappears under a fixed latency budget or fails on small distant objects, the additional sensor branch has not earned its cost. The nuScenes results motivate that test; they do not establish vehicle-level safety.

## High-Level Takeaways

- FastBEVFusion transfers depth disambiguation from a camera depth network to one LiDAR-guided cross-attention block over semi-aligned BEV features.
- Its 58.8 FPS result uses ResNet-18 and an RTX 4090. ResNet-50 improves detection but runs at 24.4 FPS under the reported setup.
- Staged training separates LiDAR detection, joint multimodal learning, and fusion-only adaptation to extrinsic noise.
- Same-hardware latency and calibrated robustness measurements remain the deciding evidence; cross-paper GPU estimates and a reversed metric table are inadequate substitutes.
