---
title: 'Deformable DETR: Deformable Transformers for End-to-End Object Detection'
date: '2020-10-08T00:00:00.000Z'
section: paper-shorts
postSlug: deformable-detr-deformable-transformers-for-end-to-end-object-detection
legacyPath: /paper shorts/2020/10/08/deformable-detr-deformable-transformers-for-end-to-end-object-detection.html
tags: [Other]
field: 'Vision Foundations'
summary: '2020 – Deformable DETR: Deformable Transformers for End-to-End Object Detection'
---
## 2020 – Deformable DETR

**arXiv:** [2010.04159](https://arxiv.org/abs/2010.04159)

**Code:** [fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)

**Venue:** [ICLR 2021 Oral](https://openreview.net/forum?id=gZ9hCDWe6ke)

## Summary

> DETR made object detection a set-prediction problem, but dense attention over image features made it slow to optimize and weak on small objects. Deformable DETR gives each query a reference point and lets each attention head sample a small, learned set of nearby features at every scale. On COCO validation with a ResNet-50 backbone, the base model reaches 43.8 AP after 50 epochs, versus 43.3 AP for DETR-DC5 after 500 epochs. The speedup comes from restricting where attention looks; it also makes reference points and offsets part of the detector's error surface.

## Core Insights

### Mechanism

In DETR, every image query can compare itself with every location in the feature map. The encoder's cost therefore grows badly when the feature map is made dense enough for small objects, and the model spends a long schedule learning which locations matter. Deformable attention changes the lookup rule. For query q with reference point p, head m predicts K two-dimensional offsets and K normalized weights. The operator bilinearly samples the feature map at p plus each offset, sums the K values, and combines the heads. The multi-scale form repeats this over four feature levels, so the query receives both fine and coarse evidence without a separate top-down pyramid.

![Deformable attention predicts offsets and weights around a reference point, then aggregates only those sampled image features.](/assets/images/deformable-detr-deformable-attention-source-figure-2.svg)
*Fig 1: The deformable attention module samples a small set of learned locations around each query reference point instead of comparing against every image location. | source: [Deformable DETR, Figure 2](https://arxiv.org/abs/2010.04159)*

In the encoder, each feature-map pixel is its own reference point. In the decoder, an object query predicts a normalized reference point, and the box head predicts offsets relative to that point. This couples the attention route to the box being refined: a useful reference point makes it easier to retrieve evidence for the corresponding object. With K fixed, encoder attention is linear in the number of feature-map pixels, while decoder cross-attention depends on the number of queries and sampled points rather than the full image area.

### Evidence

The main COCO validation comparison separates the convergence claim from the optional refinements:

| Model | Epochs | AP | AP$_S$ | FLOPs | Inference |
| --- | ---: | ---: | ---: | ---: | ---: |
| DETR-DC5 | 500 | 43.3 | 22.5 | 187 G | 12 FPS |
| DETR-DC5+ | 50 | 36.2 | 16.3 | 187 G | 12 FPS |
| Deformable DETR | 50 | 43.8 | 26.4 | 173 G | 19 FPS |
| + iterative refinement + two-stage proposals | 50 | 46.2 | 28.8 | 173 G | 19 FPS |

These are COCO 2017 validation results with ResNet-50 backbones; runtime was measured on an NVIDIA Tesla V100. DETR-DC5+ is the fairer short-schedule control because it adds focal loss and raises the query count to 300. The base Deformable DETR result therefore says more than the 46.2 AP endpoint: it reaches a higher AP than the 500-epoch DETR-DC5 model in one tenth of the training epochs, while the modified 50-epoch control remains at 36.2 AP.

The attention ablation explains the gain. Moving from one feature level to multi-scale inputs adds 1.7 AP and 2.9 small-object AP. Increasing the samples per head from one to four adds 0.9 AP. Allowing the attention operation itself to exchange information across scales adds another 1.5 AP. In the reported setting, adding FPN does not improve performance because the multi-scale deformable attention already performs the cross-level exchange.

### Boundary

Sparse routing trades coverage for a learned sampling decision. If a reference point or offset misses the evidence, the query cannot recover it through an all-pixels fallback. The custom operator also performs unordered memory accesses: despite comparable FLOPs, the paper reports that the model is still 25% slower than Faster R-CNN with FPN, though it is 1.6 times faster than DETR-DC5. The evidence is a 2D COCO study. It does not test calibration error, missing camera views, or metric 3D reference points, where later systems add geometry to the query and sampling location.

## High-Level Takeaways

- Deformable attention replaces an all-location search with a fixed number of learned samples around each reference point.
- The strongest result is convergence: 43.8 AP in 50 epochs against 43.3 AP for DETR-DC5 in 500 epochs.
- Multi-scale inputs and cross-scale sampling explain the small-object gain; FPN is redundant in the reported configuration.
- The mechanism makes learned reference points and offsets a central failure mode, and measured latency still depends on memory access rather than FLOPs alone.
