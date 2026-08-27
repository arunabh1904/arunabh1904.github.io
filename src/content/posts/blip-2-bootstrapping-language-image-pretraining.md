---
title: 'BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models'
date: '2023-01-30T00:00:00.000Z'
section: paper-shorts
postSlug: blip-2-bootstrapping-language-image-pretraining
legacyPath: /paper shorts/2023/01/30/blip-2-bootstrapping-language-image-pretraining.html
tags: [Vision-Language Models, Multimodal Generation]
field: 'Vision-Language Models'
summary: '2023 – BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models'
---

## 2023 – BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models

**arXiv:** [2301.12597](https://arxiv.org/abs/2301.12597)

## Summary

> BLIP-2 bridges a frozen image encoder and frozen language model with a lightweight Querying Transformer, or Q-Former. One pretraining stage learns visual-language representations; a second learns visual-to-language generation. The paper reports 8.7 percentage points over Flamingo-80B on zero-shot VQAv2 with 54 times fewer trainable parameters.

## Core Insights


![Figure 1 from BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models](/assets/images/blip-2-bootstrapping-language-image-pretraining-source-figure-1.webp)
*Fig 1: Overview of BLIP-2’s framework. We pre-train a lightweight Querying Transformer following a two-stage strategy to bridge the modality gap. | source: [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models](https://arxiv.org/abs/2301.12597)*

![Figure 5 from BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models](/assets/images/blip-2-bootstrapping-language-image-pretraining-source-figure-5.webp)
*Fig 2: Effect of vision-language representation learning on vision-to-language generative learning. Without representation learning, the Q-Former fails the bridge the modality gap, leading to significantly lower performance on zero-shot VQA. | source: [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models](https://arxiv.org/abs/2301.12597)*


The Q-Former is both an adapter and a compression boundary. A small set of learned queries attends to image features, so the decoder receives a predictable number of visual tokens regardless of image resolution.

Freezing both large endpoints makes training efficient and lets BLIP-2 reuse stronger unimodal checkpoints. The same choice limits adaptation. If the frozen image encoder discards text, count, or location, the Q-Former cannot reconstruct that evidence.

## High-Level Takeaways

- BLIP-2 concentrates multimodal learning in a small query-based bridge.
- Two-stage training separates representation alignment from conditional generation.
- Parameter efficiency comes with a fixed visual bottleneck and frozen endpoint assumptions.
