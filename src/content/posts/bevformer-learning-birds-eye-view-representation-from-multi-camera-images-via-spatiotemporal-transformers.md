---
title: "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers"
date: '2022-03-31T04:00:00.000Z'
section: paper-shorts
postSlug: bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers
legacyPath: /paper shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: BEVFormer learns dense BEV features from multi-camera images with spatial cross-attention and temporal self-attention, making BEV a reusable perception representation.
---
## 2022 - BEVFormer

**arXiv:** [2203.17270](https://arxiv.org/abs/2203.17270)

**Code:** [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)

**Summary:** BEVFormer builds a bird's-eye-view feature map directly from surround cameras. It uses learnable BEV grid queries that attend into camera features for spatial evidence and into previous BEV features for temporal memory.

This paper matters because it made dense BEV a practical intermediate representation for camera-only driving perception. Later end-to-end systems often either build on this BEV-centric idea or react against its compute cost.

## Paper Insights

BEVFormer has two tailored attention mechanisms. Spatial cross-attention lets each BEV query sample relevant image features from camera views by projecting into 3D reference points. Temporal self-attention lets the current BEV reuse information from the previous timestep. The same BEV representation supports 3D object detection and map segmentation.

The reported nuScenes test result in the abstract is 56.9% NDS, a 9.0 point gain over the prior best method at the time. The tradeoff is exactly what later sparse/vectorized planners try to fix: dense BEV features are powerful but expensive.

![Figure 2 from BEVFormer showing the BEV encoder with spatial cross-attention and temporal self-attention](/assets/images/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers-paper-figure.png)
_Figure 2 shows the BEVFormer encoder: camera features, BEV queries, spatial cross-attention, temporal self-attention, and detection/segmentation heads. From the [BEVFormer paper](https://arxiv.org/abs/2203.17270), via the arXiv PDF._

**What to look at:**
- BEV queries define the dense bird's-eye grid.
- Spatial cross-attention connects each BEV cell to camera evidence.
- Temporal self-attention carries history without recomputing a long video window.

**Evals / Benchmarks / Artifacts:**

| Component | Detail | Why it matters |
| --------- | ------ | -------------- |
| Representation | Dense BEV feature grid | Gives downstream tasks a common spatial frame. |
| Spatial attention | BEV queries attend to multi-camera features | Bridges image space and ground-plane reasoning. |
| Temporal attention | Current BEV attends to previous BEV | Adds motion/history with low extra structure. |
| Result | 56.9% nuScenes NDS reported in the abstract | Marked BEVFormer as a strong camera-only perception baseline. |

## Decision Lens

BEVFormer informs whether multi-camera perception should first build a persistent metric BEV grid or reason directly in camera views. Its atomic unit is a learned BEV query tied to a ground-plane location; spatial cross-attention samples projected image features, while temporal self-attention carries the same grid across frames.

The representation buys a reusable geometry for 3D detection at the cost of projection assumptions and a dense query budget. The key missing control compares dense BEV queries with sparse object queries under identical backbone, temporal context, and latency. At 10× camera resolution or temporal length, feature sampling and BEV-query attention dominate. The BEV-first claim would fail if a view-space or sparse model matched 3D accuracy and temporal stability with materially lower memory and latency.

**Context:** BEVFormer became one of the reference points for dense BEV-oriented autonomous driving stacks.

**Takeaway:** Multi-camera perception becomes much easier to organize once the model learns a shared BEV workspace.
