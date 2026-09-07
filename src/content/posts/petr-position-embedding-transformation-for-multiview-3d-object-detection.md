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

> PETR puts 3D geometry into perspective image tokens before a DETR-style decoder sees them. It discretizes camera frustums, transforms the samples into the ego frame, and encodes those coordinates with a $1\times1$ position encoder. Object queries can then attend globally to position-aware features across views. The paper reports 44.1 mAP and 50.4 NDS on nuScenes test; its central ablation is more revealing than the leaderboard, with 2D positional encoding at 6.9 mAP versus 3D positional encoding at 30.5 mAP.

## Core Insights

### Make a perspective token a metric address

Given a feature map from each camera, PETR samples 64 points along every camera ray using linear-increasing depth bins. Intrinsics and extrinsics transform those samples into a shared 3D region, which is normalized and passed through a pointwise position encoder. The resulting positional tensor is added to the image feature at that perspective location. A transformer decoder then uses learned object queries to attend to all position-aware tokens and regresses classes and 3D boxes. The geometry is attached before retrieval, so the query does not need to project a reference point and select a small local image neighborhood as in DETR3D.

The architecture figure should be read from left to right and then back through attention. The image backbone produces six perspective feature maps; the coordinate generator creates the same-shaped frustum mesh for each camera; the position encoder turns those coordinates into feature channels; and the decoder's object queries gather evidence from the concatenated views. PETR is not a dense BEV lifter: it keeps the camera feature layout and lets the query discover which position-aware tokens explain one object.

![PETR: Position Embedding Transformation for Multi-View 3D Object Detection source figure: The architecture of the proposed PETR paradigm.](/assets/images/petr-position-embedding-transformation-for-multiview-3d-object-detection-paper-figure.webp)
*Fig 1: Camera-frustum coordinates are transformed into 3D position embeddings, added to multiview image features, and read by object queries that decode 3D boxes and classes. | source: [PETR, Figure 2](https://arxiv.org/abs/2203.05625)*

### The position ablation isolates the useful signal

The paper's ablation compares positional encoding choices rather than only decoder depth. Standard 2D positional encoding reaches 6.9 mAP, while the 3D coordinate encoding reaches 30.5 mAP in the reported setup. This gap is the evidence that a generic row/column index does not give a camera token a usable ego-frame address. A simple $1\times1$ position encoder improves the reported model from 25.6 to 30.9 mAP. A $3\times3$ encoder instead mixes neighboring features whose frustum coordinates represent different rays, and training collapses. In this representation, locality in the feature map is not the same as locality in 3D.

PETR's strongest externally pretrained test configuration reports 44.1 mAP and 50.4 NDS. A 1056×384 setting is reported at 10.7 FPS, but that throughput is tied to the paper's hardware and implementation. Global query-to-image attention also scales with the number of camera tokens, so the flexible retrieval pattern has a predictable resolution and camera-count cost.

### Global attention can gather cross-view evidence

The qualitative figure pairs predicted boxes in BEV with their projections in the camera views. Read the same object across panels: a box is useful only when its metric footprint and image projections agree. The examples show PETR's ability to produce coherent cross-view boxes, but they do not measure calibration error or long-range depth accuracy.

![Figure 6 from PETR: Position Embedding Transformation for Multi-View 3D Object Detection](/assets/images/petr-position-embedding-transformation-for-multiview-3d-object-detection-source-figure-6.webp)
*Fig 2: Qualitative detection results are shown in BEV and image views at a 0.25 score threshold with a ResNet-101 backbone; the paired views make cross-view box consistency visible. | source: [PETR, Figure 6](https://arxiv.org/abs/2203.05625)*

The attention visualization makes the retrieval mechanism inspectable. For a truck query, strong responses appear in both front-left and back-left views, so the decoder can combine non-adjacent camera evidence for one object. The map is a learned attention pattern, not a calibrated depth probability or a guarantee that every highlighted pixel contributed causally to the box.

![Figure 7 from PETR: Position Embedding Transformation for Multi-View 3D Object Detection](/assets/images/petr-position-embedding-transformation-for-multiview-3d-object-detection-source-figure-7.webp)
*Fig 3: The attention map for a truck query responds in both the front-left and back-left cameras, illustrating global cross-view retrieval from 3D position-aware features. | source: [PETR, Figure 7](https://arxiv.org/abs/2203.05625)*

PETR pays for that flexibility with implicit depth learning. Its 64 depth samples are hypotheses, and the decoder must learn to select evidence across them. The paper does not provide an explicit per-pixel depth ground truth target in the core formulation, so a matched comparison with depth-supervised lifting remains useful when range-specific localization matters.

## High-Level Takeaways

- PETR moves geometry from query-side projection into the image-token embedding, making every perspective feature carry 3D coordinate hypotheses.
- The 2D-versus-3D positional encoding and $1\times1$-versus-$3\times3$ ablations show that coordinate fidelity matters more than generic neighboring context.
- Global attention can combine evidence across cameras, but memory and latency grow with image tokens, views, and decoder queries.
- The 44.1 mAP/50.4 NDS test result is a strong baseline; its limits are implicit depth selection, calibration sensitivity, and dense query-to-token attention.
