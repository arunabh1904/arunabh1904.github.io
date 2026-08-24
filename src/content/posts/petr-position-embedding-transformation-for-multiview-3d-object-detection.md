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

### Method and reported result

PETR assigns each multiview image feature a 3D position embedding derived from the camera frustum and calibration. Object queries then use global transformer attention over image features that already encode possible world coordinates. Unlike DETR3D, the query does not project one reference point to sample a local image feature; the geometry is distributed across the feature map before decoding.

## Summary

> This is a different answer to the camera-to-3D problem: make perspective features position-aware, then let attention learn the correspondence.

## Core Insights

PETR discretizes points along each camera ray, transforms them into the ego frame, normalizes the resulting coordinates over a region of interest, and maps them through a small network into a positional tensor. Learnable 3D anchor points initialize the object queries.

![PETR: Position Embedding Transformation for Multi-View 3D Object Detection source figure: The architecture of the proposed PETR paradigm.](/assets/images/petr-position-embedding-transformation-for-multiview-3d-object-detection-paper-figure.webp)
*The architecture of the proposed PETR paradigm. source: [PETR: Position Embedding Transformation for Multi-View 3D Object Detection](https://arxiv.org/abs/2203.05625)*

![Figure 6 from PETR: Position Embedding Transformation for Multi-View 3D Object Detection](/assets/images/petr-position-embedding-transformation-for-multiview-3d-object-detection-source-figure-6.webp)
*Figure 6 Qualitative analysis of detection results in BEV and image views. The score threshold is 0.25, while the backbone is ResNet-101. The 3D bounding boxes are drawn with different colors to distinguish different classes. source: [PETR: Position Embedding Transformation for Multi-View 3D Object Detection](https://arxiv.org/abs/2203.05625)*

![Figure 7 from PETR: Position Embedding Transformation for Multi-View 3D Object Detection](/assets/images/petr-position-embedding-transformation-for-multiview-3d-object-detection-source-figure-7.webp)
*Figure 7 Visualization of attention maps, generated from an object query (corresponding to the truck) on multi-view images. Both front-left and back-left views have a high response on the attention map. source: [PETR: Position Embedding Transformation for Multi-View 3D Object Detection](https://arxiv.org/abs/2203.05625)*


The central ablation is unusually clear. Standard 2D positional encoding reaches only 6.9 mAP; 3D positional encoding reaches 30.5 mAP in the same reported setup. A simple $1\times1$ position encoder raises mAP from 25.6 to 30.9, while $3\times3$ convolutions collapse training because they mix a feature with neighboring coordinates and break correspondence.

| Geometry interface | Retrieval pattern | Main cost |
| --- | --- | --- |
| DETR3D | Project each 3D query into local image features | Sparse support can miss object extent. |
| PETR | Embed 3D coordinates into all image tokens | Global query-image attention is dense. |
| Dense BEV lifting | Pool ray hypotheses into metric cells | Cost follows BEV area and resolution. |

PETR's strongest externally pretrained test model reports 44.1 mAP and 50.4 NDS on nuScenes. The paper also reports 10.7 FPS for a 1056×384 configuration, although device and implementation differences make cross-paper speed comparisons weak. PETR converges more slowly than DETR3D, consistent with learning global correspondence rather than receiving a local projection prior.

## High-Level Takeaways

- PETR informs where geometric coordinates should enter a camera transformer. Its atomic units are perspective features annotated with 3D positional hypotheses and object queries initialized by learned anchors. The architecture buys flexible global matching by paying attention over the whole multiview feature set.
- A matched test should compare PETR, DETR3D-style sampling, and BEV lifting at equal image tokens, queries, training length, and depth supervision. PETR loses if global attention's memory cost dominates or if its implicit ray matching remains less calibrated than explicit depth. At more cameras or higher resolution, query-image attention grows before object sparsity can help.
- PETR follows DETR3D but moves geometry from the query projection into the image-token embedding. PETRv2 aligns those embeddings over time; StreamPETR converts the resulting queries into recurrent object memory.
- Position embeddings can turn perspective tokens into a metric search space, but global flexibility is purchased with dense attention and slower geometric learning.
