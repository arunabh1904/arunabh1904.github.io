---
title: 'GLIP: Grounded Language-Image Pre-training'
date: '2021-12-07T00:00:00.000Z'
section: paper-shorts
postSlug: glip-grounded-language-image-pretraining
legacyPath: /paper shorts/2021/12/07/glip-grounded-language-image-pretraining.html
tags: [Vision-Language Models, Visual Grounding]
field: 'Vision-Language Models'
summary: '2021 – GLIP: Grounded Language-Image Pre-training'
---

## 2021 – GLIP: Grounded Language-Image Pre-training

**arXiv:** [2112.03857](https://arxiv.org/abs/2112.03857)

**Code:** [microsoft/GLIP](https://github.com/microsoft/GLIP)

## Summary

> GLIP unifies object detection and phrase grounding so both tasks can train one language-aware detector. Its 27 million grounding examples combine 3 million human annotations with 24 million web image-text pairs grounded through self-training. Zero-shot evaluation reports 49.8 AP on COCO and 26.9 AP on LVIS without using COCO images during pretraining.

## Core Insights

![GLIP zero-shot transfer to several object detection domains through text prompts](/assets/images/glip-paper-figure-1.png)
_A prompt names the categories for each target dataset, and the same grounded model produces boxes without task-specific label heads. Source: [GLIP](https://arxiv.org/abs/2112.03857), Figure 1._

GLIP rewrites detection labels as phrases. That puts box-supervised detection data and phrase-grounding data into one training format. Self-training then turns much larger web corpora into noisy grounding supervision.

The scale improves zero-shot and few-shot transfer across object-level tasks. The boundary is pseudo-label quality: web captions provide vocabulary, but the model's own boxes decide which phrase-region pairs become training evidence.

## High-Level Takeaways

- GLIP makes detection and grounding share one language-conditioned objective.
- Self-training increases grounding scale from millions of manual pairs to tens of millions of examples.
- Open vocabulary grows with language data, while localization remains sensitive to noisy generated boxes.
