---
title: 'StreamPETR: Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection'
date: '2023-03-21T00:00:00.000Z'
section: paper-shorts
postSlug: streampetr-object-centric-temporal-modeling-for-multiview-3d-detection
legacyPath: /paper shorts/2023/03/21/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection.html
tags:
  - Other
field: 'BEV Perception & Mapping'
summary: '2023 – StreamPETR: Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection'
---
## 2023 – StreamPETR

**arXiv:** [2303.11926](https://arxiv.org/abs/2303.11926)

**Code:** [exiawsh/StreamPETR](https://github.com/exiawsh/StreamPETR)

### Method and reported result

StreamPETR models a long camera sequence by propagating a bounded memory of object queries. At each frame, ego-motion-compensated historical queries interact with current multi-view features and new queries. The top foreground queries enter a FIFO queue; background queries are discarded.

## Summary

> The model's claim is about state compression. A dense temporal BEV stores evidence for every cell, while StreamPETR keeps the hypotheses most likely to matter for detection and tracking.

## Core Insights

The propagation transformer combines three inputs: memory queries, current image features, and fresh learnable queries that can discover new objects. Motion-aware layer normalization conditions features on ego pose, time interval, and estimated velocity before interaction. This avoids treating an old object query as if it were observed in the current frame at the same coordinates.

The paper reports 67.6 NDS and 65.3 AMOTA for its strongest online nuScenes configuration. Its lightweight ResNet-50 model reports 45.0 mAP at 31.7 FPS on an RTX 3090, 2.3 mAP above and 1.8× faster than the cited SOLOFusion comparison. Those results establish a strong accuracy-throughput point; they do not measure worst-case latency or the safety cost of dropping a low-confidence object from memory.

![Figure 3 from StreamPETR, showing historical object queries transformed, updated with current images, and filtered into a recurrent memory queue](/assets/images/streampetr-paper-figure-3.png)
*Fig 1: Only top foreground queries survive into the next frame, so query selection is both the efficiency mechanism and a temporal recall risk. | source: [StreamPETR](https://arxiv.org/abs/2303.11926)*

![Figure 1 from StreamPETR: Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection](/assets/images/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection-source-figure-1.webp)
*Fig 2: Different temporal fusion methods from bird-eye-view (BEV) space, perspective view, and our proposed object-centric. RF indicates receptive field. | source: [StreamPETR: Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection](https://arxiv.org/abs/2303.11926)*

![Figure 6 from StreamPETR: Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection](/assets/images/streampetr-object-centric-temporal-modeling-for-multiview-3d-detection-source-figure-6.webp)
*Fig 3: Visualization results of StreamPETR. On the BEV plane (right), the groud-truth and predictions are drawn in green and blue rectangles respectively. | source: [StreamPETR: Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection](https://arxiv.org/abs/2303.11926)*


| Temporal design | State carried forward | Primary tradeoff |
| --- | --- | --- |
| Dense BEV recurrence | Every spatial cell | Broad context at high memory and warp cost. |
| Perspective feature history | Multi-view image features | Retains pixels but repeatedly samples a growing history. |
| StreamPETR | Top object queries | Long history at low cost, but background evidence is discarded. |
| Fresh queries | New object hypotheses | Recovers births that memory cannot contain. |

## High-Level Takeaways

- StreamPETR informs whether long-term temporal state should be scene-centric or object-centric. Its atomic unit is an object query carrying reference position, feature, velocity, and time. The camera backbone is shared; temporal capacity is allocated by top-k foreground selection rather than grid size.
- The missing test evaluates object birth, temporary occlusion, re-entry, and false-track persistence at fixed memory and compute, then compares against dense BEV recurrence. At 10× actors, the fixed query queue becomes a recall bottleneck and query self-attention grows. The object-memory bet would fail if safety-relevant background or map changes materially affect detection before a fresh query can recover them.
- StreamPETR turns the PETR camera detector into an online recurrent system and helps establish object queries as compact temporal memory for later sparse end-to-end driving stacks.
- A long temporal horizon is affordable when the model stores state, age, and motion for objects instead of replaying the video.
