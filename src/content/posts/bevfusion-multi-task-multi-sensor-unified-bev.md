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

### Method and reported result

BEVFusion converts camera and LiDAR features independently into bird's-eye view, fuses them there, and reuses the result for 3D detection and BEV map segmentation. Its central claim is architectural: BEV is not only an output coordinate system but the common representation where dense camera semantics and LiDAR geometry can meet without reducing the image to features sampled at LiDAR points.

## Summary

> This is the paper behind the modern “shared BEV trunk plus task heads” default. A different 2022 paper also called BEVFusion focuses on LiDAR-malfunction robustness; the note here covers the MIT multi-task, multi-sensor system.

## Core Insights

The camera path uses a Lift-Splat-style view transform; the LiDAR path flattens voxel features along height. A convolutional BEV encoder absorbs local misalignment after concatenation, then task-specific heads predict objects or map classes. Because both modalities become dense BEV features before fusion, a semantic task can use camera background evidence that point-level fusion would discard.

The engineering contribution matters as much as the diagram. The paper replaces generic cumulative-sum BEV pooling with precomputed intervals and specialized reduction, reporting more than a 40× reduction in view-transform latency. On nuScenes, the abstract reports gains of 1.3 points in mAP and NDS for 3D detection and 13.6 mIoU for BEV map segmentation, with 1.9× lower computation cost than the compared prior systems.

![Figure 2 from BEVFusion, showing camera and LiDAR features converted into a shared BEV before multi-task heads](/assets/images/bevfusion-unified-bev-paper-figure-2.png)
*BEVFusion's unification boundary is visible in the middle: modality-specific encoders end at BEV, after which the encoder and task interfaces are shared. source: [BEVFusion](https://arxiv.org/abs/2205.13542)*

![Figure 1 from BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird](/assets/images/bevfusion-multi-task-multi-sensor-unified-bev-source-figure-1.webp)
*Figure 1 Fig. 1 : BEVFusion unifies camera and LiDAR features in a shared BEV space instead of mapping one modality to the other. It preserves camera’s semantic density and LiDAR’s geometric structure. source: [BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird](https://arxiv.org/abs/2205.13542)*

![Figure 2 from BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird](/assets/images/bevfusion-multi-task-multi-sensor-unified-bev-source-figure-2.webp)
*Figure 2 Fig. 2 : BEVFusion extracts features from multi-modal inputs and converts them into a shared bird’s-eye view (BEV) space efficiently using view transformations. It fuses the unified BEV features with a fully-convolutional BEV encoder and supports different tasks with task-specific heads. source: [BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird](https://arxiv.org/abs/2205.13542)*


| Shared object | What is preserved | What can be lost |
| --- | --- | --- |
| Camera BEV | Dense semantics in a metric grid | Height detail and uncertain depth structure. |
| LiDAR BEV | Accurate ground-plane geometry | Sparse semantics and fine vertical structure after flattening. |
| Fused BEV | One tensor reusable across heads | Sensor provenance unless retained explicitly. |
| Task heads | Cheap specialization after the shared trunk | Cross-task interference remains a training problem. |

## High-Level Takeaways

- BEVFusion informs whether a production stack should pay once for a shared geometric representation and reuse it across sensors and tasks. The atomic unit is a BEV cell. Camera and LiDAR encoders remain modality-specific, while the BEV encoder is shared; task heads and losses remain separate.
- The decisive missing ablation compares shared and task-specific BEV encoders under equal parameters, latency, and augmentation, with gradient-conflict measurements. The paper demonstrates architectural reuse but does not establish that joint detection and segmentation always improve each other. At 10× spatial extent or resolution, dense BEV memory and convolution dominate. The shared-grid bet would fail when sparse object and map queries match downstream task quality and temporal stability at much lower compute.
- BEVFusion joins LSS's camera view transformation to a task-agnostic multi-sensor trunk and became the reference baseline for later robustness, interaction, and radar-camera work.
- A unified model becomes economical when sensors share an expensive spatial workspace and tasks specialize only after that workspace is built.
