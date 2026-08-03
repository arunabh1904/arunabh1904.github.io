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

**Summary:** PETRv2 extends PETR's 3D position-aware image tokens across time and tasks. Ego pose aligns historical 3D coordinates before their position embeddings meet current queries. Separate query families then use the same multiview features for object detection, BEV segmentation, and 3D lane detection.

The paper unifies the interface, not the output representation. Boxes use sparse 3D detection queries, BEV segmentation uses patch queries, and lanes use ordered anchor-point queries.

## Paper Insights

Coordinate alignment gives historical features the position they occupy in the current ego frame. A feature-guided position encoder modulates geometric embeddings with appearance so the same nominal coordinate can carry different evidence. In the main nuScenes test configuration, PETRv2 reports 49.0 mAP and 58.2 NDS; its multiscale variant reaches 50.8 mAP and 59.1 NDS. Average velocity error drops from PETR's 0.808 m/s to 0.343 m/s.

| Added component | Validation effect | Interpretation |
| --- | --- | --- |
| Temporal frames without coordinate alignment | +2.7 NDS, +0.5 mAP | History helps, but unaligned geometry limits it. |
| Coordinate alignment | Further +2.1 NDS, +0.9 mAP | Ego-frame consistency carries most temporal value. |
| Feature-guided position encoding | Final 49.6 NDS, 40.1 mAP | Appearance adapts the geometric prior. |

The robustness study is as important as the leaderboard. Extrinsic noise degrades every variant; feature-guided encoding reduces but does not remove the loss. Dropping one camera reduces mAP, especially the wide rear camera. A delay of roughly 83 ms lowers mAP by 3.19 points, and delays above 0.3 s produce a much larger collapse. Temporal modeling does not substitute for sensor synchronization.

## Decision Lens

PETRv2 informs how one camera representation can support detection, mapping, and lanes without forcing one query geometry on all three. The shared object is a set of calibrated, position-aware image features. Task capacity stays explicit through query initialization and heads; temporal sharing occurs through aligned coordinates.

The rejection test compares this task-query design with a dense BEV trunk at matched resolution and latency, including extrinsic noise, camera loss, and timestamp jitter. PETRv2 loses if global attention or position encoding becomes brittle under real calibration drift, or if dense BEV gives stronger task consistency. At more tasks, query families and their competing gradients become the likely capacity bottleneck.

**Context:** PETR establishes 3D position embeddings; PETRv2 makes them temporal and multi-task. StreamPETR later retains selected PETR queries as recurrent object state instead of replaying the historical token set.

**Takeaway:** A unified camera model can share calibrated evidence while giving boxes, maps, and lanes different query contracts—and synchronization remains part of the model.
