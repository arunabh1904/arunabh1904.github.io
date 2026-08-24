---
title: 'Multimodal Few-Shot Learning with Frozen Language Models'
date: '2021-06-25T00:00:00.000Z'
section: paper-shorts
postSlug: multimodal-few-shot-learning-with-frozen-language-models
legacyPath: /paper shorts/2021/06/25/multimodal-few-shot-learning-with-frozen-language-models.html
tags: [Vision-Language Models, Multimodal Generation]
field: 'Vision-Language Models'
summary: '2021 – Multimodal Few-Shot Learning with Frozen Language Models'
---

## 2021 – Multimodal Few-Shot Learning with Frozen Language Models

**arXiv:** [2106.13884](https://arxiv.org/abs/2106.13884)

## Summary

> This paper shows that a frozen autoregressive language model can acquire a visual interface without changing its weights. A trained vision encoder maps each image into a continuous prefix. Interleaving those prefixes with text examples lets the model perform captioning, visual question answering, and few-shot concept learning through the language model's existing generation interface.

## Core Insights

![Examples of multimodal few-shot generation from a frozen language model](/assets/images/multimodal-few-shot-frozen-lm-paper-figure-1.png)
*Image prefixes let a frozen language model condition generation on visual examples and reuse knowledge learned from text. source: [Multimodal Few-Shot Learning with Frozen Language Models](https://arxiv.org/abs/2106.13884)*

![Figure 4 from Multimodal Few-Shot Learning with Frozen Language Models](/assets/images/multimodal-few-shot-learning-with-frozen-language-models-source-figure-4.webp)
*Figure 4 Examples of (a) the Open-Ended miniImageNet evaluation (b) the Fast VQA evaluation. source: [Multimodal Few-Shot Learning with Frozen Language Models](https://arxiv.org/abs/2106.13884)*

![Figure 3 from Multimodal Few-Shot Learning with Frozen Language Models](/assets/images/multimodal-few-shot-learning-with-frozen-language-models-source-figure-3.webp)
*Figure 3 Inference-Time interface for Frozen . The figure demonstrates how we can support (a) visual question answering, (b) outside-knowledge question answering and (c) few-shot image classification via in-context learning. source: [Multimodal Few-Shot Learning with Frozen Language Models](https://arxiv.org/abs/2106.13884)*


The key decision is where adaptation lives. The language model stays fixed. Only the visual pathway learns to produce embeddings that the decoder can interpret as context. This makes visual conditioning cheap relative to retraining the full generator.

The experiments show multimodal in-context learning across several benchmarks, including learning names for novel objects from a handful of examples. The bottleneck is also clear: the prefix must compress all useful visual evidence into a representation accepted by a language model that was never trained to see.

## High-Level Takeaways

- A visual prefix can reuse a frozen language model's few-shot behavior.
- Freezing the decoder reduces training cost but concentrates responsibility in the visual connector.
- Generation quality does not prove that fine spatial detail survived the prefix.
