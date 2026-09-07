---
title: 'PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images'
date: '2022-06-02T04:00:00.000Z'
section: paper-shorts
postSlug: petrv2-unified-3d-perception-from-multicamera-images
legacyPath: /paper shorts/2022/06/02/petrv2-unified-3d-perception-from-multicamera-images.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2022 – PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images'
---
## 2022 – PETRv2

**arXiv:** [2206.01256](https://arxiv.org/abs/2206.01256)

**Code:** [megvii-research/PETR](https://github.com/megvii-research/PETR)

## Summary

> The paper unifies the interface, not the output representation. Boxes use sparse 3D detection queries, BEV segmentation uses patch queries, and lanes use ordered anchor-point queries.

## Core Insights

PETRv2 keeps PETR's frustum-derived 3D position-aware image features, then transforms the previous frame's coordinates into the current ego frame before concatenating the two feature sets. A feature-guided position encoder uses image appearance to modulate the geometric embedding, so a nominal location can carry different evidence when the scene changes. One decoder serves three query contracts: detection queries are initialized over 3D space, segmentation queries over BEV patches, and lane queries as ordered 3D points. The shared feature interface is therefore narrower than “one output for every task.”

In the main nuScenes test configuration, PETRv2 reports 49.0 mAP and 58.2 NDS; the multiscale variant reaches 50.8 mAP and 59.1 NDS. Average velocity error drops from PETR's 0.808 m/s to 0.343 m/s. Those numbers combine temporal modeling, a changed encoder, and training choices, so the component ablation is more informative than the headline.


![Figure 3 from PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images](/assets/images/petrv2-unified-3d-perception-from-multicamera-images-source-figure-3.webp)
*Fig 1: Detection queries cover the 3D space, segmentation queries are initialized under BEV, and lane queries use ordered points. The query geometry is the task contract: it lets the decoder reuse calibrated evidence without pretending that boxes, regions, and curves have the same output structure. | source: [PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images](https://arxiv.org/abs/2206.01256)*

![Figure 1 from PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images](/assets/images/petrv2-unified-3d-perception-from-multicamera-images-source-figure-1.webp)
*Fig 2: Previous-frame frustum coordinates are pose-transformed into the current ego frame, concatenated with current image features, and passed through the feature-guided position encoder before task-specific queries decode boxes, BEV segmentation, and lanes. | source: [PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images](https://arxiv.org/abs/2206.01256)*


The component ablation makes the temporal claim testable. Adding history without coordinate alignment improves the reported validation result by 2.7 NDS and 0.5 mAP. Aligning the previous 3D coordinates adds another 2.1 NDS and 0.9 mAP, showing why simply concatenating frames is not equivalent to temporal fusion. Adding feature-guided position encoding reaches 49.6 NDS and 40.1 mAP in that ablation. The gain is therefore a chain: memory supplies another view, pose puts it in the same metric frame, and appearance conditions the position code.

The robustness study gives that chain a boundary. Extrinsic noise degrades every variant; feature guidance reduces the loss without removing it. Dropping one camera hurts mAP, especially for the wide rear view. An approximately 83 ms delay lowers mAP by 3.19 points, while delays above 0.3 s cause a much larger collapse. PETRv2's temporal alignment assumes the pose and timestamp contract it is given; the model cannot recover information that arrives from the wrong frame.

## High-Level Takeaways

- PETRv2 shares calibrated, position-aware image features while giving boxes, BEV regions, and lanes separate query geometries.
- The temporal gain comes from pose-aligning previous frustum coordinates before attention; frame concatenation alone is a weaker control.
- Extrinsic noise, camera loss, and delay tests are part of the model result because the representation assumes accurate calibration and synchronization.
- PETR provides the spatial interface; PETRv2 makes it temporal and multi-task, while StreamPETR later compresses selected object queries into recurrent state.
