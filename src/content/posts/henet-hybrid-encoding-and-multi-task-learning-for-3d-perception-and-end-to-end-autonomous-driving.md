---
title: >-
  HENet++: Hybrid Encoding and Multi-task Learning for 3D Perception and
  End-to-end Autonomous Driving
date: '2026-10-13T09:00:00.000Z'
section: paper-shorts
postSlug: >-
  henet-hybrid-encoding-and-multi-task-learning-for-3d-perception-and-end-to-end-autonomous-driving
legacyPath: >-
  /paper
  shorts/2026/10/13/henet-hybrid-encoding-and-multi-task-learning-for-3d-perception-and-end-to-end-autonomous-driving.html
tags:
  - Autonomous Driving
  - Perception
field: BEV Perception & Mapping
summary: >-
  2025 – HENet++: Hybrid Encoding and Multi-task Learning for 3D Perception and
  End-to-end Autonomous Driving
---
## 2025 – HENet++: Hybrid Encoding and Multi-task Learning for 3D Perception and End-to-end Autonomous Driving

**Paper:** [arXiv:2511.07106](https://arxiv.org/abs/2511.07106) · [Full text, October 2026 revision](https://arxiv.org/html/2511.07106v2)

## Summary

> HENet++ combines expensive recent-frame encoding with cheaper temporal context, then gives sparse object tasks and dense scene tasks different representations. Its large multitask model reaches 64.1 NDS, 58.4 BEV segmentation mIoU, and 49.0 occupancy mIoU on nuScenes. Adding radar to its planning variant lowers the reported collision metric, but trajectory error remains higher than several compared planners. The paper supports a representation and training-efficiency argument; its logged-data planning results do not establish closed-loop driving performance.

## Core Insights

### Spend image compute unevenly across time

High resolution helps identify distant or small objects, while a longer history helps recover motion and occlusion. Applying the largest encoder to every historical image makes those benefits expensive to combine. HENet uses a large backbone and a richer view transformation for a short recent window, then a small backbone and cheaper transformation for older frames.

The original HENet builds dense BEV features with BEVStereo on its expensive branch and BEVDepth on its cheaper branch. A shared adjacent-frame fusion module passes information backward through the observed sequence and then forward to the current frame. This backward pass visits past observations; it does not consume future frames. Separate task encoders select features and spatial resolutions for detection and segmentation.

HENet++ extends that idea to sparse features. Why keep both sparse and dense representations? The source architecture sends object instances directly to detection, while grids retain the background and free-space context needed for segmentation, occupancy, and planning.

![HENet++ source Figure 2 showing hybrid image encoders feeding sparse instance and dense voxel representations](/assets/images/henet-plus-plus-source-figure-2.webp)
*Fig 1: Recent high-resolution and longer low-resolution observations support both sparse object features and dense scene features. Task-specific heads share the image evidence without requiring every task to use the same spatial representation. | source: [HENet++, Figure 2](https://arxiv.org/html/2511.07106v2#S2.F2)*

[View the figure at full resolution](/assets/images/henet-plus-plus-source-figure-2.webp).

The sparse branch adapts SparseBEV's queries, spatiotemporal sampling, and adaptive mixing. Its 900 initial queries sample both temporal branches through six decoder layers. Channel-mixing weights are shared, while point mixing differs with the number of sampled frames. The two resulting query sets cannot safely be averaged by index: they may have drifted toward different objects. A box head identifies duplicate instances, and channel-wise max pooling merges features associated with the same object.

This is feature fusion, not an ensemble of final detections. The distinction matters in the ablation: simply combining outputs from a weaker and stronger model introduces false positives and underperforms the stronger model. Learning to combine their intermediate evidence gives a different result.

### Match the representation and initialization to each task

Compared with [SparseDrive](</paper shorts/2024/05/30/sparsedrive-end-to-end-autonomous-driving-via-sparse-scene-representation.html>), HENet++ retains an explicit dense occupancy pathway alongside object queries. Dense features retain information that object queries can miss, including surfaces outside predefined foreground categories. Conversely, forcing object detection through a dense grid introduces depth, projection, and clustering steps before recovering instances. HENet++ lets the prediction target determine which representation a decoder consumes.

Detection uses sparse features with focal classification and L1 box losses. BEV segmentation uses a SegNet-style decoder and focal loss; occupancy uses a small MLP and focal loss. Depth estimation adds binary cross-entropy. The original HENet also separates detection and segmentation BEV resolutions: its ablation favors 0.4-metre cells for detection and 0.8-metre cells for segmentation. Independent feature selection and encoders raise the resulting multitask scores further.

Initialization creates another task conflict. Starting from a detection model favors detection, while occupancy initialization favors occupancy. The authors merge separately trained encoders using Regression Mean for convolutional and linear weights, with ordinary averaging for other parameters, then fine-tune the multitask model. The revised manuscript reports limited gains from direct model merging. Merging is useful here as initialization rather than a substitute for joint training.

The revised small-model ablation averages five seeds. Detection initialization gives $53.6\pm0.2$ NDS and $36.2\pm0.3$ occupancy mIoU; occupancy initialization gives $53.2\pm0.2$ and $37.3\pm0.2$. Merged initialization reaches $53.8\pm0.3$ and $37.6\pm0.2$, with BEV mIoU at $55.3\pm0.3$. These are modest gains, with overlapping variation on some axes. The merging hyperparameter defaults to 0.9; values from 0.7 to 0.9 perform comparably within the reported variation.

### Planning reads objects and scene context together

The planning extension retains the top 200 object detections and downsamples occupancy features to a $50\times50\times4$ grid. An ego query embeds historical trajectory, navigation waypoints, velocity, and acceleration. Three Transformer layers alternate interactions among ego and object queries, attention to their initial features, and attention to dense occupancy tokens. Each layer predicts future trajectories for the ego vehicle and surrounding objects.

Training adds L1 losses for ego planning and object prediction to the perception objective. A collision term penalizes predicted ego–object centre distances below three metres. This is a simplified geometric surrogate: it is not a full shape-aware collision calculation. Radar can enter both sparse and dense pathways through the RCBEVDet++ encoder and fusion modules.

The planner is fine-tuned for ten epochs without class-balanced group sampling. The paper separately describes twelve-epoch training for the HENet++ perception models and their single-task counterparts. Its large perception configuration uses VoVNet-99 at $640\times1600$ and ResNet-50 at $256\times704$. Frame counts differ across result tables: the main multitask table labels the large model as five plus eleven frames, while the controlled hybrid-encoding ablation reports five plus six. Those rows should not be treated as identical compute settings.

### Gains in collision score come with other costs

| nuScenes validation measure | Camera HENet++ | Camera + radar HENet++ |
| --- | ---: | ---: |
| UniAD-protocol average L2 | 1.44 m | 1.29 m |
| UniAD-protocol average collision rate | 0.17% | 0.13% |
| VAD/ST-P3-protocol average L2 | 0.61 m | 0.55 m |
| VAD/ST-P3-protocol average collision rate | 0.06% | 0.05% |

The protocols differ in temporal averaging and which objects count toward collisions. Their numbers cannot be interchanged. Radar improves both metrics within each protocol, but the compared SSR planner has lower trajectory error: 0.75 m under the UniAD protocol and 0.39 m under VAD/ST-P3. HENet++'s lower collision result is therefore a trade-off, not dominance on all planning axes.

The loss ablation helps explain that trade-off. Under the UniAD protocol, planning-only supervision gives 1.69 m L2 and 0.52% collision. Adding object prediction improves those to 1.36 m and 0.20%; adding the collision penalty reaches 1.29 m and 0.13%. The experiments support supervising surrounding-agent motion alongside ego motion, though they do not isolate all contributions of sparse versus dense planner inputs.

The training-efficiency evidence is also configuration-specific. In the sparse detection ablation, hybrid encoding reaches 65.1 NDS at 12.1 FPS and 72 training hours, compared with the larger SparseBEV baseline's 63.2 NDS, 9.6 FPS, and 100 hours. Those rows use A100 hardware and mixed precision. Other rows use RTX 3090 inference or FP32 training on A800s, so cross-table speed ratios would be misleading. The smaller HENet ablation even trades slower inference for shorter training and higher accuracy.

The October 2026 revision adds NAVSIM evaluation: camera-only HENet++ reaches 92.6 PDMS, compared with 91.7 for iPad, 84.0 for PARA-Drive, and 83.4 for UniAD in its table. HENet++ improves progress and non-collision scores over iPad but has lower time-to-collision score, 94.2 versus 94.9. NAVSIM uses non-reactive simulation from recorded scenes; it strengthens the planning evidence without supplying a reactive closed-loop test. The revision also adds RayIoU for occupancy and five-seed initialization ablations, making the revised results more informative than the original submission.

The remaining experiment would hold sensor input and runtime fixed while isolating sparse, dense, and combined planning context, then test whether the collision improvement survives interaction with moving agents. There is no on-road validation in this manuscript.

## High-Level Takeaways

- Hybrid encoding allocates large-model compute to recent high-resolution frames and smaller-model compute to longer temporal context.
- HENet++ separates sparse object features from dense scene features, allowing different output tasks to share image evidence without sharing every representation.
- Merged single-task weights help initialize joint training; direct merging provides limited gains in the revised manuscript.
- Radar and additional prediction losses reduce logged-scene collision metrics, while trajectory error and runtime remain separate trade-offs. Closed-loop evidence is still needed.
