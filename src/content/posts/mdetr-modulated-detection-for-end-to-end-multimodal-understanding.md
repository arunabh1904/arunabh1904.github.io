---
title: 'MDETR: Modulated Detection for End-to-End Multimodal Understanding'
date: '2021-04-26T00:00:00.000Z'
section: paper-shorts
postSlug: mdetr-modulated-detection-for-end-to-end-multimodal-understanding
legacyPath: /paper shorts/2021/04/26/mdetr-modulated-detection-for-end-to-end-multimodal-understanding.html
tags: [Vision-Language Models, Visual Grounding]
field: 'Vision-Language Models'
summary: '2021 – MDETR: Modulated Detection for End-to-End Multimodal Understanding'
---

## 2021 – MDETR: Modulated Detection for End-to-End Multimodal Understanding

**arXiv:** [2104.12763](https://arxiv.org/abs/2104.12763)

**Code:** [ashkamath/mdetr](https://github.com/ashkamath/mdetr)

## Summary

> MDETR makes detection conditional on free-form text instead of a fixed category list. It fuses image and language features early, then predicts boxes aligned to phrases. Pretraining on 1.3 million image-text pairs with explicit phrase-object alignment transfers to phrase grounding, referring-expression comprehension, segmentation, few-shot detection, and visual question answering.

## Core Insights

![MDETR localizing an unusual text query for a pink elephant](/assets/images/mdetr-paper-figure-1.png)
_The text query changes what the detector searches for. The example tests composition beyond a fixed detector label. Source: [MDETR](https://arxiv.org/abs/2104.12763), Figure 1._

MDETR removes the frozen detector that earlier multimodal systems treated as a visual front end. A DETR-style model receives both image features and raw text, so object queries are trained against the language used to describe the scene.

The paper reports state-of-the-art results on several grounding benchmarks and competitive GQA and CLEVR results. Its central cost is supervision: the pretraining pairs require phrase-to-box alignment, which is much scarcer than ordinary image-caption data.

## High-Level Takeaways

- MDETR turns detection into a text-conditioned prediction problem.
- Early fusion gives rich phrase-box interaction but requires joint computation for each query.
- Explicit grounding labels improve localization while limiting the scale of available pretraining data.
