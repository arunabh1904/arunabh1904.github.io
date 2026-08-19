---
title: 'DETR3D: 3D Object Detection from Multi-View Images via 3D-to-2D Queries'
date: '2021-10-14T04:00:00.000Z'
section: paper-shorts
postSlug: detr3d-multiview-images-via-3d-to-2d-queries
legacyPath: /paper shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2021 – DETR3D: 3D Object Detection from Multi-View Images via 3D-to-2D Queries'
---
## 2021 – DETR3D

**arXiv:** [2110.06922](https://arxiv.org/abs/2110.06922)

**Code:** [WangYueFt/detr3d](https://github.com/WangYueFt/detr3d)

### Method and reported result

DETR3D predicts a set of 3D boxes directly from multiview image features. Each object query carries a 3D reference point; calibration projects that point into every camera and feature level; sampled image evidence updates the query; and the decoder iteratively refines the reference point and box.

## Summary

> The model avoids both a dense BEV view transform and per-camera boxes followed by global NMS. Geometry enters as a retrieval operation: a metric hypothesis asks where supporting pixels should appear.

## Core Insights

DETR3D treats all cameras jointly, which is especially useful where an object crosses camera boundaries. On nuScenes validation, its CBGS configuration reports 34.9 mAP and 43.4 NDS. In overlap regions, the FCOS3D-initialized model reaches 26.8 mAP and 38.4 NDS versus 22.9 mAP and 32.9 NDS for the compared FCOS3D setup.

| Queries | mAP | NDS | Consequence |
| ---: | ---: | ---: | --- |
| 100 | 31.3 | 40.8 | Too few hypotheses constrain recall. |
| 600 | 34.7 | 42.0 | Most of the gain has arrived. |
| 900 | 34.6 | 42.5 | NDS peaks in the reported sweep. |
| 1500 | 34.6 | 42.0 | More queries no longer help. |

The query count therefore acts as a capacity and recall budget, not a free scaling knob. The paper also reports higher translation error than explicit per-pixel depth methods and identifies single-point feature sampling as a limitation. A projected reference point can retrieve a local feature, but it does not model object extent or ray-depth ambiguity by itself.

## High-Level Takeaways

- DETR3D informs whether camera-only 3D detection needs a scene-wide BEV field. Its atomic unit is an object query with a metric reference point. The image backbone is dense and shared across views, while geometric and temporal cost after the backbone scales with the bounded query set rather than BEV area.
- The matched falsification compares query retrieval and dense BEV lifting with the same image pyramid, depth supervision, and P99 budget. DETR3D loses if missed query initialization or point sampling harms long-range recall, or if dense scene context materially improves births and free-space reasoning. At crowded scale, query count and self-attention replace grid area as the limiting resource.
- DETR3D establishes 3D-to-2D query retrieval. PETR instead embeds 3D coordinates into every image feature before global attention; Sparse4D and SparseBEV make the query's sampled support larger and adaptive.
- A camera detector can reason in metric 3D without first materializing BEV, provided each hypothesis has a calibrated route back to image evidence.
