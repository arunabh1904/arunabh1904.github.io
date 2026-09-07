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

> PETRv2 extends PETR's 3D position-aware image tokens across time and tasks. It pose-aligns the previous frame's frustum coordinates into the current ego frame, uses a feature-guided position encoder, and gives detection, BEV segmentation, and lane prediction separate query geometries. The main nuScenes test model reaches 49.0 mAP and 58.2 NDS; the multiscale variant reaches 50.8/59.1. Its strongest evidence is the alignment ablation: history without alignment helps, but putting both frames in the same metric frame adds another 2.1 NDS and 0.9 mAP.

## Core Insights

### Time is useful only after coordinates agree

PETRv2 keeps PETR's frustum-coordinate generator. For the previous frame, it applies the ego-pose transform that maps the historical 3D coordinates into the current ego frame; current and aligned historical image features are then concatenated and passed through the feature-guided position encoder (FPE). FPE sends the projected image features through a small MLP and sigmoid to produce an element-wise gate, multiplies that gate with the coordinate MLP's 3D position embedding, and adds the gated embedding to the image features for the decoder keys; the projected image features remain the decoder values. This preserves calibrated ray geometry while letting appearance decide which parts of the fixed positional signal are useful. The pose transform compensates the ego vehicle's motion. A moving car can still change position between frames, so its residual motion is not magically aligned; the decoder and velocity head must learn around that mismatch.

The paradigm figure is best read in the order of the coordinate contract. Current and previous cameras first become 3D position-aware features; pose alignment makes the historical coordinates current-frame addresses; then task queries retrieve evidence and task heads decode their own output. The shared part is the calibrated feature field, not a single universal output grid.

![Figure 1 from PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images](/assets/images/petrv2-unified-3d-perception-from-multicamera-images-source-figure-1-white.png)
*Fig 1: Previous-frame 3D coordinates are pose-transformed into the current ego frame, concatenated with current features, and encoded before task-specific queries decode detection, BEV segmentation, and lanes. | source: [PETRv2, Figure 1](https://arxiv.org/abs/2206.01256)*

### Query geometry preserves task differences

The same decoder supports three tasks through different query initialization. Detection queries are distributed in the full 3D space, segmentation queries are initialized in BEV, and lane queries represent ordered 3D points. That choice matters because a box is a sparse instance, a map is a region, and a lane is a structured curve. Sharing the feature field avoids duplicating the camera and temporal encoder while keeping the output contract explicit.

![Figure 3 from PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images](/assets/images/petrv2-unified-3d-perception-from-multicamera-images-source-figure-3-white.png)
*Fig 2: Detection queries cover 3D space, segmentation queries are initialized under BEV, and lane queries use ordered points; the different query geometries encode the tasks' different output structures. | source: [PETRv2, Figure 3](https://arxiv.org/abs/2206.01256)*

On nuScenes, PETRv2 reports 49.0 mAP and 58.2 NDS on the test split, with 0.343 m/s average velocity error versus PETR's 0.808 m/s in the paper's comparison. Its multiscale variant reaches 50.8 mAP and 59.1 NDS. The model also evaluates BEV segmentation on driveable area, lane, and vehicle classes and 3D lane detection on OpenLane. These task scores are evidence for a shared interface, not proof that all heads have identical capacity or calibration needs.

### The ablation separates memory, alignment, and appearance

The component table gives the temporal story in steps. Adding historical features without coordinate alignment improves the reported validation result by 2.7 NDS and 0.5 mAP. Applying the pose transform adds another 2.1 NDS and 0.9 mAP. Feature-guided position encoding reaches 49.6 NDS and 40.1 mAP in that ablation. The improvement is therefore not “more frames” alone: it is memory plus a metric transform plus appearance-conditioned geometry.

The robustness study tests whether that contract survives sensor errors. Extrinsic noise degrades every variant, and feature guidance reduces but does not remove the loss. Dropping one camera hurts mAP, especially for the wide rear view. An approximately 83 ms delay lowers mAP by 3.19 points; delays above 0.3 s create a much larger collapse. PETRv2 can share a temporal representation only as well as the poses, timestamps, and camera coverage that define it.

## High-Level Takeaways

- PETRv2 shares calibrated position-aware features while giving detection, BEV segmentation, and lanes their own query geometries.
- The alignment ablation shows why pose-transforming historical coordinates matters more than simply concatenating previous frames.
- Temporal modeling improves velocity and detection, but extrinsic noise, camera loss, and delay expose the representation's synchronization boundary.
- PETR supplies the spatial interface; PETRv2 makes it temporal and multi-task, while later StreamPETR compresses selected object queries into recurrent state.
