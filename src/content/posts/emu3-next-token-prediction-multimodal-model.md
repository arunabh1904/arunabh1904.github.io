---
title: 'Emu3: Next-Token Prediction Is All You Need'
date: '2024-09-28T00:00:00.000Z'
section: paper-shorts
postSlug: emu3-next-token-prediction-multimodal-model
legacyPath: /paper shorts/2024/09/28/emu3-next-token-prediction-multimodal-model.html
tags: [Omni-Model Architectures, Multimodal Generation]
field: 'Omni-Model Architectures'
summary: '2024 – Emu3: Next-Token Prediction Is All You Need'
---

## 2024 – Emu3: Next-Token Prediction Is All You Need

**arXiv:** [2409.18869](https://arxiv.org/abs/2409.18869)

## Summary

> Emu3 represents text, images, and video as discrete tokens and trains one transformer with next-token prediction. The same model performs visual understanding and generation without a diffusion decoder or a CLIP-style contrastive objective. Its evaluations compare the shared autoregressive recipe against specialist models in image generation, visual understanding, and video generation.

## Core Insights

![Emu3 comparison across image generation, visual understanding, and video generation](/assets/images/emu3-paper-figure-1.png)
*One next-token model is evaluated across tasks usually handled by separate model families. source: [Emu3](https://arxiv.org/abs/2409.18869)*

![Figure 3 from Emu3: Next-Token Prediction Is All You Need](/assets/images/emu3-next-token-prediction-multimodal-model-source-figure-3.webp)
*Figure 3 Qualitative results of Emu3 text-to-image generation. source: [Emu3: Next-Token Prediction Is All You Need](https://arxiv.org/abs/2409.18869)*

![Figure 2 from Emu3: Next-Token Prediction Is All You Need](/assets/images/emu3-next-token-prediction-multimodal-model-source-figure-2.webp)
*Figure 2 Reconstruction samples. Left: Original and reconstructed videos at 540 960 resolution, showcasing a sampling of 8 frames at 30 FPS. Right: original and reconstructed 512 512 resolution images. Zoom in to see the details. source: [Emu3: Next-Token Prediction Is All You Need](https://arxiv.org/abs/2409.18869)*


Emu3 makes tokenization carry the burden of unification. Once visual signals become discrete symbols, the transformer can train and decode them like text. This removes separate task objectives but makes visual sequence length and tokenizer fidelity central constraints.

The result supports a simple common interface, not equal representation needs. Understanding may prefer invariance, while generation must preserve local appearance. One token stream can serve both while still forcing a compromise in what each token encodes.

## High-Level Takeaways

- Emu3 tests pure next-token prediction across text, image, and video.
- Discrete visual tokens simplify training while making compression quality decisive.
- A unified objective does not remove the conflict between semantic and reconstructive features.
