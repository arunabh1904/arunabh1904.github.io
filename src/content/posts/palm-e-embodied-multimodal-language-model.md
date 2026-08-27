---
title: 'PaLM-E: An Embodied Multimodal Language Model'
date: '2023-03-06T00:00:00.000Z'
section: paper-shorts
postSlug: palm-e-embodied-multimodal-language-model
legacyPath: /paper shorts/2023/03/06/palm-e-embodied-multimodal-language-model.html
tags: [Vision-Language-Action, Robotics]
field: 'Vision-Language-Action & Robotics'
summary: '2023 – PaLM-E: An Embodied Multimodal Language Model'
---

## 2023 – PaLM-E: An Embodied Multimodal Language Model

**arXiv:** [2303.03378](https://arxiv.org/abs/2303.03378)

## Summary

> PaLM-E inserts continuous sensor and state embeddings into a pretrained language model's token sequence. One model trains across robotic planning, visual question answering, captioning, and language data. The 562-billion-parameter variant retains general language ability while reporting state-of-the-art OK-VQA performance and positive transfer across embodied tasks.

## Core Insights


![Figure 2 from PaLM-E: An Embodied Multimodal Language Model](/assets/images/palm-e-embodied-multimodal-language-model-source-figure-2.webp)
*Fig 1: PaLM-E-562B can do zero-shot multimodal chain-of-thought reasoning, can tell visually-conditioned jokes given an image, and demonstrates an array of robot-relevant multimodal-informed capabilities including perception, visually-grounded dialogue, and planning. PaLM-E also generalizes, zero-shot, to multi-image prompts despite only being trained on single-image prompts. | source: [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378)*

![Figure 3 from PaLM-E: An Embodied Multimodal Language Model](/assets/images/palm-e-embodied-multimodal-language-model-source-figure-3.webp)
*Fig 2: Overview of transfer learning demonstrated by PaLM-E: across three different robotics domains, using PaLM and ViT pretraining together with the full mixture of robotics and general visual-language data provides a significant performance increase compared to only training on the respective in-domain data. See Tab. | source: [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378)*


The model treats images, state estimates, and text as parts of one sentence. Learned encoders map continuous observations into the embedding space consumed by the language model. The decoder then produces text plans rather than low-level motor commands.

This is an embodied language model, not yet a full vision-language-action policy. Its results support transfer between internet-scale semantic tasks and robot planning. They do not show that language-token prediction alone supplies metric geometry, control frequency, or closed-loop recovery.

## High-Level Takeaways

- PaLM-E makes continuous robot observations readable by a large language model.
- Joint training transfers semantic knowledge into embodied planning tasks.
- Text plans still require a downstream controller that handles geometry and execution timing.
