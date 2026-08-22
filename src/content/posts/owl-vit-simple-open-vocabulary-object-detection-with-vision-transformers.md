---
title: 'OWL-ViT: Simple Open-Vocabulary Object Detection with Vision Transformers'
date: '2022-05-12T00:00:00.000Z'
section: paper-shorts
postSlug: owl-vit-simple-open-vocabulary-object-detection-with-vision-transformers
legacyPath: /paper shorts/2022/05/12/owl-vit-simple-open-vocabulary-object-detection-with-vision-transformers.html
tags:
  - Vision-Language Models
  - Open-Vocabulary Detection
field: 'Vision-Language Models'
summary: '2022 – OWL-ViT: Simple Open-Vocabulary Object Detection with Vision Transformers'
---

## 2022 – OWL-ViT: Simple Open-Vocabulary Object Detection with Vision Transformers

**arXiv:** [2205.06230](https://arxiv.org/abs/2205.06230)

**Code:** [google-research/scenic](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)

## Summary

> OWL-ViT transfers a contrastively pretrained image-text model into an open-vocabulary detector with minimal architectural change. It removes image-token pooling, attaches lightweight box and classification heads to the visual tokens, and uses text embeddings as class queries. The paper reports consistent detection gains as image-text pretraining and model size increase.

## Core Insights

![OWL-ViT transfer from contrastive image-text pretraining to text-conditioned object detection](/assets/images/owl-vit-paper-figure-1.svg)
_Contrastive pretraining learns the image and text encoders. Detection fine-tuning exposes individual image tokens, adds box heads, and compares object features against text or image queries. Source: [OWL-ViT](https://arxiv.org/abs/2205.06230), Figure 1._

OWL-ViT asks how much of CLIP's interface can survive localization. The answer is most of it. Text remains the open-vocabulary classifier, but the visual encoder must stop compressing the whole image into one pooled embedding. Individual tokens now carry object features and box coordinates.

The design supports zero-shot text-conditioned detection and few-shot image-conditioned detection. Its scaling study shows that stronger image-level pretraining transfers to detection, but detection fine-tuning is still required. Contrastive alignment alone does not produce a detector.

## High-Level Takeaways

- OWL-ViT turns image-text similarity into object-level classification by preserving unpooled visual tokens.
- Text and example images can both act as open-vocabulary queries.
- The simple transfer works because localization heads add a spatial output contract that contrastive pretraining did not supply.
