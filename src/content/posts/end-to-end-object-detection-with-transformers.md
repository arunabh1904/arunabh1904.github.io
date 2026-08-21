---
title: 'End-to-End Object Detection with Transformers'
date: '2020-05-26T00:00:00.000Z'
section: paper-shorts
postSlug: end-to-end-object-detection-with-transformers
legacyPath: /paper shorts/2020/05/26/end-to-end-object-detection-with-transformers.html
tags: [Other]
field: 'Vision Foundations'
summary: '2020 – DETR: object detection as direct set prediction'
---
## 2020 – DETR

**arXiv:** [2005.12872](https://arxiv.org/abs/2005.12872)

**Code:** [facebookresearch/detr](https://github.com/facebookresearch/detr)

### Method and reported result

DETR reformulates detection as direct set prediction. A CNN produces image features, a transformer encoder models global context, and a decoder maps a fixed set of learned object queries to class and box predictions. Hungarian matching assigns each ground-truth object to one prediction, making duplicate suppression part of the training objective rather than a separate non-maximum-suppression stage.

## Summary

> DETR's durable idea is the contract between queries and matching: predict a bounded unordered set, then train one prediction to own each target. That contract later became a foundation for object-centric 3D perception.

## Core Insights

On COCO, DETR reports accuracy and runtime comparable to a heavily optimized Faster R-CNN baseline. It performs particularly well on large objects and extends to panoptic segmentation with a small mask head. The result established that anchors, proposal heuristics, and NMS were not essential components of a competitive detector.

![End-to-End Object Detection with Transformers source figure: DETR directly predicts (in parallel) the final set of detections by combining a common CNN with a transformer architecture.](/assets/images/end-to-end-object-detection-with-transformers-paper-figure.webp)
_DETR directly predicts (in parallel) the final set of detections by combining a common CNN with a transformer architecture. Source: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), Figure 1, via arXiv HTML._


The simplification moves complexity into optimization. The original model uses a long 500-epoch schedule, and the paper identifies slow convergence and weaker small-object performance as open problems. Doubling feature resolution helps small objects but makes encoder self-attention much more expensive.

| Design choice | Function |
| --- | --- |
| Fixed object queries | Bound the output set. |
| Bipartite matching | Enforce one-to-one ownership. |
| No-object class | Represent unused query slots. |
| Global attention | Let predictions reason over the full image and one another. |

## High-Level Takeaways

- Object queries are useful because matching gives them an assignment semantics; the vector alone is not the contribution.
- Set prediction removes duplicate-handling heuristics but introduces a hard optimization and query-budget problem.
- For driving, 3D reference points and calibrated feature sampling must be added because a learned 2D query has no metric location.
- DETR is the conceptual ancestor of many query-based detectors, not a drop-in autonomous-driving perception stack.
