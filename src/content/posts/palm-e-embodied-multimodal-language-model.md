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

![PaLM-E examples of multimodal reasoning, dialogue, and robot planning](/assets/images/palm-e-paper-figure-2.png)
_PaLM-E uses one multimodal sequence interface across visual reasoning and robot planning tasks. Source: [PaLM-E](https://arxiv.org/abs/2303.03378), Figure 2._

The model treats images, state estimates, and text as parts of one sentence. Learned encoders map continuous observations into the embedding space consumed by the language model. The decoder then produces text plans rather than low-level motor commands.

This is an embodied language model, not yet a full vision-language-action policy. Its results support transfer between internet-scale semantic tasks and robot planning. They do not show that language-token prediction alone supplies metric geometry, control frequency, or closed-loop recovery.

## High-Level Takeaways

- PaLM-E makes continuous robot observations readable by a large language model.
- Joint training transfers semantic knowledge into embodied planning tasks.
- Text plans still require a downstream controller that handles geometry and execution timing.
