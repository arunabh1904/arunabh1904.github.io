---
title: 'LXMERT: Learning Cross-Modality Encoder Representations from Transformers'
date: '2019-08-20T00:00:00.000Z'
section: paper-shorts
postSlug: lxmert-learning-cross-modality-encoder-representations
legacyPath: /paper shorts/2019/08/20/lxmert-learning-cross-modality-encoder-representations.html
tags:
  - Vision-Language Models
  - Multimodal Pretraining
field: 'Vision-Language Models'
summary: '2019 – LXMERT: Learning Cross-Modality Encoder Representations from Transformers'
---

## 2019 – LXMERT: Learning Cross-Modality Encoder Representations from Transformers

**arXiv:** [1908.07490](https://arxiv.org/abs/1908.07490)

**Code:** [airsplay/lxmert](https://github.com/airsplay/lxmert)

## Summary

> LXMERT separates object relationships, language context, and cross-modal reasoning into three encoders. Five pretraining tasks supervise both the individual modalities and their interaction. The paper reports state-of-the-art results on VQA and GQA, while NLVR2 accuracy improves from the previous 54 percent to 76 percent.

## Core Insights

![LXMERT architecture with object, language, and cross-modality encoders](/assets/images/lxmert-paper-figure-1.png)
_Object and language encoders first build modality-specific context. Cross-attention layers then exchange evidence before a joint output is formed. Source: [LXMERT](https://arxiv.org/abs/1908.07490), Figure 1._

LXMERT makes the supervision mixture explicit. Masked language modeling teaches the text side. Masked object feature regression and label prediction teach the visual side. Cross-modal matching and image question answering pressure the two streams to interact.

The result is a useful decomposition of multimodal pretraining, but the tasks remain tied to detector regions and curated datasets. Its gains show that structured cross-modal objectives matter. They do not establish that this recipe scales efficiently to internet-sized retrieval.

## High-Level Takeaways

- LXMERT assigns separate capacity to objects, language, and their interaction.
- Its five objectives make alignment a training mixture rather than one contrastive comparison.
- Detector regions improve object-level structure but cap the vocabulary and evidence available downstream.
