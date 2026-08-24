---
title: 'Kosmos-2: Grounding Multimodal Language Models to the World'
date: '2023-06-26T00:00:00.000Z'
section: paper-shorts
postSlug: kosmos-2-grounding-multimodal-language-models
legacyPath: /paper shorts/2023/06/26/kosmos-2-grounding-multimodal-language-models.html
tags: [Vision-Language Models, Visual Grounding]
field: 'Vision-Language Models'
summary: '2023 – Kosmos-2: Grounding Multimodal Language Models to the World'
---

## 2023 – Kosmos-2: Grounding Multimodal Language Models to the World

**arXiv:** [2306.14824](https://arxiv.org/abs/2306.14824)

## Summary

> Kosmos-2 makes grounding part of autoregressive text generation. Referring phrases are written as links to location tokens that encode bounding boxes. Training on the GrIT corpus of grounded image-text pairs lets one model describe regions, resolve referring expressions, and mix ordinary language with spatial references.

## Core Insights

![Kosmos-2 generating text grounded to image regions with bounding boxes](/assets/images/kosmos-2-paper-figure-1.png)
*Kosmos-2 emits language and location tokens in one sequence, so a phrase can point back to visible evidence. source: [Kosmos-2](https://arxiv.org/abs/2306.14824)*

![Figure 1 from Kosmos-2: Grounding Multimodal Language Models to the World](/assets/images/kosmos-2-grounding-multimodal-language-models-source-figure-1.webp)
*Figure 1 Kosmos-2 is a multimodal large language model that has new capabilities of multimodal grounding and referring. Kosmos-2 can understand multimodal input, follow instructions, perceive object descriptions ( e.g. , bounding boxes), and ground language to the visual world. source: [Kosmos-2: Grounding Multimodal Language Models to the World](https://arxiv.org/abs/2306.14824)*

![Figure 4 from Kosmos-2: Grounding Multimodal Language Models to the World](/assets/images/kosmos-2-grounding-multimodal-language-models-source-figure-4.webp)
*Figure 4 Input format of evaluation on (1) phrase grounding and (2) referring expression comprehension. source: [Kosmos-2: Grounding Multimodal Language Models to the World](https://arxiv.org/abs/2306.14824)*


The key interface is serialization. A box no longer comes from a separate detector head. It is represented in the same output stream as the referring phrase. That makes grounding available to downstream generation and instruction-following tasks.

The format gives spatial predictions a visible target, but coordinate tokens are still two-dimensional. They establish phrase-region correspondence without supplying depth, pose, metric scale, or object persistence over time.

## High-Level Takeaways

- Kosmos-2 turns boxes into tokens that can appear inside generated language.
- One output stream unifies referring and description tasks.
- Pixel coordinates make grounding inspectable but do not by themselves provide geometry.
