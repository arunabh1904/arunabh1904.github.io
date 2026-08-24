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

![SpatialVLM pipeline lifting internet images into synthetic 3D spatial questions](/assets/images/spatialvlm-paper-figure-2.png)
*The pipeline combines segmentation, depth, captions, and geometric rules to produce spatial supervision at scale. source: [SpatialVLM](https://arxiv.org/abs/2401.12168)*

![Figure 6 from SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning](/assets/images/spatialvlm-spatial-reasoning-capabilities-source-figure-6.webp)
*Figure 6 SpatialVLM as reward generator for robotics tasks. SpatialVLM provides a “natural-language queriable" distance estimation tool, and can be used for robotics tasks. For example, for the task “pick orange tea bottle", the reward/cost function can be the a function of the response of “What is the distance between the yellow gripper fingers and the orange tea bottle". source: [SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning](https://arxiv.org/abs/2401.12168)*

![Figure 2 from SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning](/assets/images/spatialvlm-spatial-reasoning-capabilities-source-figure-2.webp)
*Figure 2 An overview of our data synthesis pipeline. (a) We use CLIP to filter noisy internet images and only keep scene-level photos. (b) We apply pre-trained expert models on internet-scale images so that we get object-centric segmentation, depth and caption. (c) We lift the 2D image into 3D point clouds, which can be parsed by shape analysis rules to extract useful properties like 3D bounding box. source: [SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning](https://arxiv.org/abs/2401.12168)*


The paper addresses a data problem. Captions rarely state metric relationships, while manual 3D annotations are expensive. SpatialVLM uses pretrained experts to generate the missing targets, then teaches a language model to answer spatial questions.

Synthetic scale comes with inherited error. Depth estimates, segmentations, and rules become labels, so their biases can be learned as geometry. The method improves spatial language without replacing calibrated depth or pose when metric error matters.

## High-Level Takeaways

- SpatialVLM turns pretrained perception models into a spatial-data generator.
- Quantitative targets make spatial error more measurable than ordinary caption loss.
- Synthetic supervision broadens coverage but inherits the errors of its expert pipeline.
