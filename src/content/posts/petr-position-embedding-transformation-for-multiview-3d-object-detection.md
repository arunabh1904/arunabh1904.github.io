---
title: 'PETR: Position Embedding Transformation for Multi-View 3D Object Detection'
date: '2022-03-10T05:00:00.000Z'
section: paper-shorts
postSlug: petr-position-embedding-transformation-for-multiview-3d-object-detection
legacyPath: /paper shorts/2022/03/10/petr-position-embedding-transformation-for-multiview-3d-object-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – PETR: Position Embedding Transformation for Multi-View 3D Object Detection'
---
## 2022 – PETR

**arXiv:** [2203.05625](https://arxiv.org/abs/2203.05625)

**Code:** [megvii-research/PETR](https://github.com/megvii-research/PETR)

## Summary

> This is a different answer to the camera-to-3D problem: make perspective features position-aware, then let attention learn the correspondence.

## Core Insights

PETR discretizes 64 points along each camera ray, transforms those points with the camera calibration into the ego frame, normalizes the coordinates over a fixed region, and maps them through a 3D position encoder. The position tensor is added to the image feature at the same perspective location. A transformer decoder then lets learned object queries attend to all six views and predicts boxes and classes. The 3D points are hypotheses attached to image features; PETR does not first decide which depth is correct and then pool a BEV cell.

![PETR: Position Embedding Transformation for Multi-View 3D Object Detection source figure: The architecture of the proposed PETR paradigm.](/assets/images/petr-position-embedding-transformation-for-multiview-3d-object-detection-paper-figure.webp)
*Fig 1: PETR transforms camera-frustum coordinates into 3D position embeddings, combines them with multi-view image features, and decodes object queries into classes and 3D boxes. | source: [PETR: Position Embedding Transformation for Multi-View 3D Object Detection](https://arxiv.org/abs/2203.05625)*

![Figure 6 from PETR: Position Embedding Transformation for Multi-View 3D Object Detection](/assets/images/petr-position-embedding-transformation-for-multiview-3d-object-detection-source-figure-6.webp)
*Fig 2: Qualitative analysis of detection results in BEV and image views at a 0.25 score threshold with a ResNet-101 backbone. Read the BEV and camera panels together: a plausible 3D box must explain both its metric footprint and its projected image evidence. | source: [PETR: Position Embedding Transformation for Multi-View 3D Object Detection](https://arxiv.org/abs/2203.05625)*

![Figure 7 from PETR: Position Embedding Transformation for Multi-View 3D Object Detection](/assets/images/petr-position-embedding-transformation-for-multiview-3d-object-detection-source-figure-7.webp)
*Fig 3: Attention maps for a truck query respond strongly in both the front-left and back-left views, illustrating that global query attention can gather evidence for one object across cameras. The map is an explanation of the learned retrieval pattern, not a calibrated depth map. | source: [PETR: Position Embedding Transformation for Multi-View 3D Object Detection](https://arxiv.org/abs/2203.05625)*


The central ablation tests whether the representation actually carries 3D information. In the reported setup, ordinary 2D positional encoding reaches only 6.9 mAP, whereas 3D positional encoding reaches 30.5 mAP. Adding camera-view information alone is not enough: the 3D coordinate signal is the part that makes the feature tokens addressable in the ego frame. The position encoder's locality also matters. A $1\times1$ encoder raises mAP from 25.6 to 30.9; a $3\times3$ encoder mixes neighboring pixels whose frustum coordinates represent different rays and the model fails to train. That result is a useful warning for geometry-aware networks: spatial smoothing is not free when neighboring locations have different coordinate meanings.

On the nuScenes test set, the paper reports 44.1 mAP and 50.4 NDS for its strongest externally pretrained model. A 1056×384 configuration is reported at 10.7 FPS, but the paper's hardware and implementation details do not make that number a universal comparison. The method's cost is also structural: each query can attend to all multiview position-aware tokens, so image resolution and camera count increase the decoder's retrieval work. PETR trades the explicit locality of DETR3D sampling for the ability to discover multi-camera support globally.

## High-Level Takeaways

- PETR moves geometry from a query-side projection into every image token, making a perspective feature a metric address before decoding.
- The 2D-versus-3D positional encoding and $1\times1$-versus-$3\times3$ ablations show that coordinate fidelity, rather than generic spatial context, drives the gain.
- Global query attention can combine evidence across cameras, but its memory and latency grow with image tokens and camera count.
- PETRv2 aligns the same position-aware tokens through time; DETR3D remains the useful contrast when sparse, local retrieval is the stronger systems constraint.
