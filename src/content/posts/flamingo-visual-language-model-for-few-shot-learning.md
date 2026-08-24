---
title: 'Flamingo: A Visual Language Model for Few-Shot Learning'
date: '2022-04-29T00:00:00.000Z'
section: paper-shorts
postSlug: flamingo-visual-language-model-for-few-shot-learning
legacyPath: /paper shorts/2022/04/29/flamingo-visual-language-model-for-few-shot-learning.html
tags: [Vision-Language Models, Multimodal Generation]
field: 'Vision-Language Models'
summary: '2022 – Flamingo: A Visual Language Model for Few-Shot Learning'
---

## 2022 – Flamingo: A Visual Language Model for Few-Shot Learning

**arXiv:** [2204.14198](https://arxiv.org/abs/2204.14198)

## Summary

> Flamingo connects pretrained vision and language models while leaving most of both systems intact. A Perceiver Resampler compresses variable visual features into a fixed token set, and gated cross-attention layers let a frozen language model read those tokens. The result accepts interleaved images, video, and text and adapts to new tasks from a few in-context examples.

## Core Insights

![Flamingo examples across few-shot visual tasks and multi-image dialogue](/assets/images/flamingo-paper-figure-1.svg)
*One model changes tasks through interleaved visual and textual prompts instead of task-specific fine-tuning. source: [Flamingo](https://arxiv.org/abs/2204.14198)*

![Figure 4 from Flamingo: A Visual Language Model for Few-Shot Learning](/assets/images/flamingo-visual-language-model-for-few-shot-learning-source-figure-4.webp)
*Figure 4 gated xattn-dense layers. To condition the LM on visual inputs, we insert new cross-attention layers between existing pretrained and frozen LM layers. The keys and values in these layers are obtained from the vision features while the queries are derived from the language inputs. They are followed by dense feed-forward layers. These layers are gated so that the LM is kept intact at initialization for improved stability and performance. source: [Flamingo: A Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)*

![Figure 1 from Flamingo: A Visual Language Model for Few-Shot Learning](/assets/images/flamingo-visual-language-model-for-few-shot-learning-source-figure-1.webp)
*Figure 1 Selected examples of inputs and outputs obtained from Flamingo -80B. Flamingo can rapidly adapt to various image/video understanding tasks with few-shot prompting (top). Out of the box, Flamingo is also capable of multi-image visual dialogue (bottom). More examples in Appendix C. source: [Flamingo: A Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)*


Flamingo's interface has two bottlenecks. The resampler decides which visual evidence survives. Gated cross-attention decides where that evidence enters the language model. This keeps training tractable and preserves the language model, but a fixed visual token budget can discard small objects or fine geometry.

The paper reports state-of-the-art few-shot results across image and video benchmarks, often beating models fine-tuned on far more task-specific data. The comparison supports in-context transfer, not open access or cheap reproduction: the largest model and training mixture are substantial.

## High-Level Takeaways

- Flamingo made interleaved multimodal prompting a practical few-shot interface.
- Resampling controls visual context cost but creates an information bottleneck.
- Gated cross-attention adapts a strong frozen decoder without forcing full end-to-end retraining.
