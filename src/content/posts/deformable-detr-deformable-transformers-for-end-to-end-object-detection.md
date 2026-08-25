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

> DETR made object detection a set-prediction problem, but dense attention over image features left it slow to train and weak on small objects. Deformable DETR replaces that dense lookup with a few learned samples around a reference point at each feature scale. On COCO validation with a ResNet-50 backbone, the base model reaches 43.8 AP after 50 epochs; DETR-DC5 reaches 43.3 AP after 500. Small-object AP rises from 22.5 to 26.4. Sparse sampling makes high-resolution, multi-scale features tractable, but moves the burden onto the quality of the reference points and learned offsets.

## Core Insights

[DETR](/paper%20shorts/2020/05/26/end-to-end-object-detection-with-transformers.html) asks every image query to compare itself with every location in the feature map. Encoder attention therefore grows quadratically with image resolution. At initialization, attention is also nearly uniform, so the model needs a long schedule to discover the few locations that matter. Higher-resolution features help with small objects, but make both problems worse.

Deformable attention narrows the search. Given a query and a reference point, each attention head predicts a small set of offsets and a weight for each sampled location. Bilinear interpolation retrieves those features, and their weighted sum becomes the query update. The multi-scale version repeats the operation across four feature levels; the default model uses eight heads and four samples per head per level. In the encoder, a feature-map location serves as its own reference point. In the decoder, each object query predicts a reference point that acts as an initial guess for the box center.

![Deformable attention predicts sampling offsets and weights around a query reference point, retrieves only those image features, and aggregates them across attention heads.](/assets/images/deformable-detr-deformable-attention-source-figure-2.svg)
*Each query predicts where to sample and how strongly to weight each sampled feature instead of attending to every image location. source: [Deformable DETR](https://arxiv.org/abs/2010.04159)*

The important change is the routing rule. Dense attention lets a query inspect the entire feature map, but pays for every possible query-key pair. Deformable attention fixes the number of sampled keys, so encoder complexity grows linearly with feature-map area and multi-scale features become affordable without a separate feature pyramid.

The COCO comparison separates the convergence result from the final detector variants:

| Model | Epochs | AP | AP$_S$ | FLOPs | Inference |
| --- | ---: | ---: | ---: | ---: | ---: |
| DETR-DC5 | 500 | 43.3 | 22.5 | 187 G | 12 FPS |
| DETR-DC5+ | 50 | 36.2 | 16.3 | 187 G | 12 FPS |
| Deformable DETR | 50 | 43.8 | 26.4 | 173 G | 19 FPS |
| + refinement and two-stage proposals | 50 | 46.2 | 28.8 | 173 G | 19 FPS |

*COCO 2017 validation results with ResNet-50 backbones; runtime was measured on an NVIDIA Tesla V100. DETR-DC5+ adds focal loss and increases the query count to 300 for a closer 50-epoch control.*

The base result is more informative than the 46.2 AP endpoint. Deformable DETR slightly exceeds the 500-epoch DETR-DC5 result after one tenth as many epochs, while the 50-epoch DETR-DC5+ control remains at 36.2 AP. The attention ablation also identifies where the gain comes from: multi-scale inputs add 1.7 AP, increasing the samples per head from one to four adds 0.9 AP, and allowing attention to exchange information across scales adds another 1.5 AP. Once that cross-scale exchange is present, adding FPN or BiFPN does not materially improve AP in the reported setting.

Sparse routing is still a trade-off. A query can only use the locations it samples, so a poor reference point or offset can exclude the needed evidence. The custom operator also uses irregular memory access; despite comparable FLOPs, the paper reports that Deformable DETR remains 25% slower than Faster R-CNN with FPN. The evidence is limited to 2D detection on COCO. It does not test calibrated multi-camera geometry, where [DETR3D](/paper%20shorts/2021/10/14/detr3d-multiview-images-via-3d-to-2d-queries.html) projects sparse 3D object queries into camera features, or [BEVFormer](/paper%20shorts/2022/03/31/bevformer-learning-birds-eye-view-representation-from-multi-camera-images-via-spatiotemporal-transformers.html), where BEV queries retrieve image evidence near geometry-derived reference points.

## High-Level Takeaways

- Deformable attention replaces an all-pixels search with a fixed number of learned samples around each reference point, making high-resolution and multi-scale features practical inside a DETR-style detector.
- The paper's strongest result is convergence: the base ResNet-50 model reaches 43.8 AP in 50 epochs, while DETR-DC5 reaches 43.3 AP in 500.
- Sparse sampling trades dense coverage for learned routing. Reference points and offsets become part of the model's error surface, and irregular memory access keeps measured speed from following FLOPs exactly.
- The COCO result does not establish that sparse sampling remains reliable under calibration error, occlusion, or missing camera views. Multiview driving models inherit the mechanism but add a new geometric failure mode.
