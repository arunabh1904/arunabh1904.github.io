---
title: 'BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation'
date: '2022-01-28T00:00:00.000Z'
section: paper-shorts
postSlug: blip-bootstrapping-language-image-pretraining
legacyPath: /paper shorts/2022/01/28/blip-bootstrapping-language-image-pretraining.html
tags: [Vision-Language Models, Multimodal Generation]
field: 'Vision-Language Models'
summary: '2022 – BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation'
---

## 2022 – BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

**arXiv:** [2201.12086](https://arxiv.org/abs/2201.12086)

**Code:** [salesforce/BLIP](https://github.com/salesforce/BLIP)

## Summary

> BLIP trains one model for image-text retrieval, matching, and caption generation, then uses that model to improve its own web data. On the same 14 million pretraining images, the paper reports gains over ALBEF of 2.7 points in average recall@1 on COCO retrieval, 2.8 CIDEr on captioning, and 1.6 points on VQA. The result is specific to the reported task recipes and does not show that one shared representation is optimal for every downstream use.

## Core Insights

![BLIP multimodal mixture of encoder-decoder architecture and pretraining objectives](/assets/images/blip-paper-figure-2.png)
*BLIP shares most parameters across three modes. The image-text encoder learns contrastive alignment, the image-grounded text encoder learns matching, and the image-grounded text decoder learns caption generation. source: [BLIP](https://arxiv.org/abs/2201.12086)*

![Figure 2 from BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](/assets/images/blip-bootstrapping-language-image-pretraining-source-figure-2.webp)
*Figure 2 Pre-training model architecture and objectives of BLIP (same parameters have the same color). We propose multimodal mixture of encoder-decoder, a unified vision-language model which can operate in one of the three functionalities: (1) Unimodal encoder is trained with an image-text contrastive (ITC) loss to align the vision and language representations. source: [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)*

![Figure 4 from BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](/assets/images/blip-bootstrapping-language-image-pretraining-source-figure-4.webp)
*Figure 4 Examples of the web text and the synthetic text . Green texts are accepted by the filter, whereas red texts are rejected. source: [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)*


BLIP makes generation part of the visual representation recipe. The same multimodal mixture of encoder-decoder, or MED, switches attention masks to act as a text encoder, an image-grounded text encoder, or an image-grounded text decoder. Image-text contrastive loss learns a shared space. Image-text matching loss learns pairwise fusion. Language-modeling loss makes the visual evidence usable one word at a time.

The second contribution is data bootstrapping. A captioner generates synthetic descriptions for web images, while a separately tuned filter removes weak original and synthetic pairs. Both start from the pretrained MED and are tuned on COCO. This CapFilt loop turns the model into part of the data pipeline instead of treating scraped alt text as fixed supervision.

The reported ablations also expose a tradeoff. Nucleus sampling produced more diverse synthetic captions and better downstream results than beam search, while sharing weights between the captioner and filter made the filter less effective at rejecting the captioner's mistakes. Diversity helps only when the quality check retains some independence.

## High-Level Takeaways

- BLIP places contrastive alignment, cross-modal matching, and caption generation inside one parameter-sharing scheme.
- CapFilt treats noisy web text as something the model can rewrite and filter, not simply consume.
- The strongest evidence is the controlled 14 million-image comparison across retrieval, captioning, and VQA.
- The model still trains substantial vision-language machinery together; [BLIP-2](/paper%20shorts/2023/01/30/blip-2-bootstrapping-language-image-pretraining.html) later asks how much of that bridge can be learned while the large vision and language endpoints remain frozen.
