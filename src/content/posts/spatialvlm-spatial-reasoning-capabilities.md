---
title: 'SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning'
date: '2024-01-22T00:00:00.000Z'
section: paper-shorts
postSlug: spatialvlm-spatial-reasoning-capabilities
legacyPath: /paper shorts/2024/01/22/spatialvlm-spatial-reasoning-capabilities.html
tags: [Vision-Language Models, Spatial Reasoning]
field: 'Vision-Language Models'
summary: '2024 – SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning'
---

## 2024 – SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning

**arXiv:** [2401.12168](https://arxiv.org/abs/2401.12168)

## Summary

> SpatialVLM builds quantitative spatial supervision from ordinary internet images. Expert models estimate segmentation, depth, captions, and object geometry; rules then turn those estimates into spatial question-answer pairs. Fine-tuning on this synthetic corpus improves qualitative and quantitative reasoning about distance, size, and spatial relations.

## Core Insights


![Figure 6 from SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning](/assets/images/spatialvlm-spatial-reasoning-capabilities-source-figure-6.webp)
*Fig 1: SpatialVLM as reward generator for robotics tasks. SpatialVLM provides a “natural-language queriable" distance estimation tool, and can be used for robotics tasks. | source: [SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning](https://arxiv.org/abs/2401.12168)*

![Figure 2 from SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning](/assets/images/spatialvlm-spatial-reasoning-capabilities-source-figure-2.webp)
*Fig 2: The synthesis pipeline filters scene-level images, applies segmentation, depth, and captioning models, then lifts the result into 3D point clouds from which spatial attributes and bounding boxes are extracted. | source: [SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning](https://arxiv.org/abs/2401.12168)*


The paper addresses a data problem. Captions rarely state metric relationships, while manual 3D annotations are expensive. SpatialVLM uses pretrained experts to generate the missing targets, then teaches a language model to answer spatial questions.

Synthetic scale comes with inherited error. Depth estimates, segmentations, and rules become labels, so their biases can be learned as geometry. The method improves spatial language without replacing calibrated depth or pose when metric error matters.

## High-Level Takeaways

- SpatialVLM turns pretrained perception models into a spatial-data generator.
- Quantitative targets make spatial error more measurable than ordinary caption loss.
- Synthetic supervision broadens coverage but inherits the errors of its expert pipeline.
