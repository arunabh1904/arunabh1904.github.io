---
title: 'Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model'
date: '2024-08-20T09:00:00.000Z'
section: paper-shorts
postSlug: transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model
legacyPath: /paper shorts/2024/08/20/transfusion-predict-the-next-token-and-diffuse-images-with-one-multimodal-model.html
tags: [Multimodal AI]
field: Omni-Models
summary: '2024 – Transfusion: one transformer, next-token language modeling, and image diffusion.'
---

## 2024 – Transfusion

**arXiv:** [2408.11039](https://arxiv.org/abs/2408.11039)  
**Conference:** Technical report

**Summary:** Transfusion combines next-token prediction for discrete data with diffusion for continuous image data in a single transformer. It studies models up to 7B parameters trained on text-image mixtures and compares the recipe with discrete image-token language modeling.

## Paper Insights

The paper separates the question of a shared transformer from the question of a shared loss. Text remains autoregressive while images use diffusion; modality-specific encoders and decoders handle the interface. The authors report that this hybrid recipe scales better than quantizing images into discrete tokens in their setting and can compress images to 16 patches with modality-specific layers.

| Design choice | Why it is useful |
| --- | --- |
| Continuous visual objective | Avoids forcing image generation through a discrete-token bottleneck. |
| Shared transformer | Preserves cross-modal interaction. |
| Modality-specific I/O | Lets each modality use a suitable representation. |

**Limits:** Hybrid objectives make loss weighting, training diagnostics, and serving more complicated than one next-token objective.

**Takeaway:** A unified model does not require a unified loss; use the objective that matches the modality.

