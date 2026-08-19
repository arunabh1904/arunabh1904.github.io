---
title: 'Sparse4D v2: Recurrent Temporal Fusion with a Sparse Model'
date: '2023-05-23T04:00:00.000Z'
section: paper-shorts
postSlug: sparse4dv2-recurrent-temporal-fusion-with-sparse-model
legacyPath: /paper shorts/2023/05/23/sparse4dv2-recurrent-temporal-fusion-with-sparse-model.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2023 – Sparse4D v2: Recurrent Temporal Fusion with a Sparse Model'
---
## 2023 – Sparse4D v2

**arXiv:** [2305.14018](https://arxiv.org/abs/2305.14018)

**Code:** [linxuewu/Sparse4D](https://github.com/linxuewu/Sparse4D)

### Method and reported result

Sparse4D v2 changes the first Sparse4D from explicit multi-frame image sampling into a recurrent instance model. Historical anchors and features are transformed into the current frame, merged with fresh single-frame proposals, and refined again. Temporal decoder cost becomes independent of the nominal history length because only the previous sparse state crosses the frame boundary.

## Summary

> The architectural change is a memory decision: transmit structured hypotheses frame by frame instead of repeatedly reopening the video.

## Core Insights

The decoder reserves one layer for fresh anchors and uses later layers for recurrent temporal instances. Camera-parameter encoding exposes calibration to the instance update. Dense LiDAR-derived depth supervision stabilizes the camera backbone during training but is absent from inference. Efficient Deformable Aggregation fuses sampling and weighted reduction into one operator.

| Change | Reported effect | Why it matters |
| --- | --- | --- |
| Efficient aggregation | 6.3→3.1 GB train memory; 13.7→20.3 FPS | Sparse algorithms still need fused kernels. |
| Recurrent temporal fusion | +9.8 mAP, +12.5 NDS over single-frame | Compact state retains useful history. |
| Fresh single-frame layer | +3.5 mAP over all-temporal decoder | A recurrent model still needs query birth. |
| Dense depth supervision | Prevents a reported training collapse | Train-time geometry can stabilize a camera-only graph. |

The low-resolution ResNet-50 model reports 43.9 mAP and 53.9 NDS at 20.3 FPS. The pretrained high-resolution configuration reaches 50.5 mAP and 59.4 NDS at 8.4 FPS. The paper notes that its head cost does not grow with image resolution, although the dense camera backbone still does.

## High-Level Takeaways

- Sparse4D v2 informs when temporal compression should occur. Its atomic unit is a recurrent 3D instance rather than a frame, pixel, or BEV cell. The expensive image backbone remains dense; savings arrive in evidence retrieval, temporal storage, and decoder reuse.
- The rejection test compares recurrent instances with fixed-window resampling at the same history information, image backbone, and P99 latency. The recurrent design loses if old errors accumulate, new-object recall falls, or dense weak evidence is needed before a query is born. At 10× actors, instance count, duplicate resolution, and self-attention replace history length as the bottleneck.
- Sparse4D v1 samples several timestamps explicitly. v2 makes the anchors recurrent; v3 adds stronger denoising, quality estimation, and tracking supervision to keep those instances reliable.
- Temporal cost becomes bounded when the model carries forward a scene hypothesis rather than a stack of observations—but query birth and state quality become first-class problems.
