---
title: 'GRIT: Teaching MLLMs to Think with Images'
date: '2025-05-21T00:00:00.000Z'
section: paper-shorts
postSlug: grit-teaching-mllms-to-think-with-images
legacyPath: /paper shorts/2025/05/21/grit-teaching-mllms-to-think-with-images.html
tags: [Vision-Language Models, Visual Reasoning]
field: 'Alignment & Post-Training'
summary: '2025 – GRIT: Teaching MLLMs to Think with Images'
---

## 2025 – GRIT: Teaching MLLMs to Think with Images

**arXiv:** [2505.15879](https://arxiv.org/abs/2505.15879)

## Summary

> GRIT trains multimodal models to interleave language reasoning with bounding-box coordinates that identify the image regions used along the way. Its GRPO-GR reinforcement-learning objective rewards final answers and grounded output format without requiring annotated reasoning chains or box labels. The paper reports effective training with as few as 20 image-question-answer examples.

## Core Insights


![Figure 1 from GRIT: Teaching MLLMs to Think with Images](/assets/images/grit-teaching-mllms-to-think-with-images-source-figure-1.webp)
*Fig 1: Comparison of reasoning with pure natural language and our grounded reasoning that mixes explicit bounding boxes for image regions with a chain of natural language thoughts. Our GRIT method enables MLLMs to perform grounded reasoning with only 20 training samples, realizing a clear and reliable process of thinking with images. | source: [GRIT: Teaching MLLMs to Think with Images](https://arxiv.org/abs/2505.15879)*

![Figure 2 from GRIT: Teaching MLLMs to Think with Images](/assets/images/grit-teaching-mllms-to-think-with-images-source-figure-2.webp)
*Fig 2: GRPO-GR samples groups of completions and combines grounded-reasoning format, target-counting, and GPT-assisted answer-accuracy rewards to update the multimodal model. | source: [GRIT: Teaching MLLMs to Think with Images](https://arxiv.org/abs/2505.15879)*


The method changes the reasoning interface. A language-only chain can sound coherent while drifting away from the image. GRIT requires the model to name coordinates as it reasons, which gives the reward function a structural target and the reader an evidence trail.

Boxes improve inspectability, but they do not prove causal use. A model can learn plausible regions and plausible text together without depending on the selected pixels. Counterfactual masking remains necessary to test whether the grounded trace caused the answer.

## High-Level Takeaways

- GRIT puts visible region references inside the reasoning sequence.
- Reinforcement learning avoids manually labeling full grounded reasoning chains.
- Box-formatted reasoning is easier to inspect but still needs causal grounding tests.
