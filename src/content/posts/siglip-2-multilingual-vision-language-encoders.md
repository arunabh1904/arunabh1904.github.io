---
title: 'SigLIP 2: Multilingual Vision-Language Encoders with Dense Features'
date: '2025-02-20T00:00:00.000Z'
section: paper-shorts
postSlug: siglip-2-multilingual-vision-language-encoders
legacyPath: /paper shorts/2025/02/20/siglip-2-multilingual-vision-language-encoders.html
tags:
  - Vision-Language Models
  - Contrastive Learning
field: 'Vision-Language Models'
summary: '2025 – SigLIP 2: Multilingual Vision-Language Encoders with Dense Features'
---

## 2025 – SigLIP 2: Multilingual Vision-Language Encoders with Dense Features

**arXiv:** [2502.14786](https://arxiv.org/abs/2502.14786)

**Models and code:** [google-research/big_vision](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/image_text)

## Summary

> SigLIP 2 keeps SigLIP's pairwise sigmoid alignment, then adds captioning, self-distillation, masked prediction, online data curation, multilingual data, and variable-resolution training. Across the released model scales, the paper reports gains over SigLIP in classification, retrieval, vision-language transfer, localization, and dense prediction.

## Core Insights

![SigLIP 2 training recipe combining sigmoid alignment with captioning and self-supervised objectives](/assets/images/siglip-2-paper-figure-1.svg)
_SigLIP 2 does not replace the dual encoder. It broadens what the visual encoder must preserve by adding captioning and dense self-supervised signals to the original sigmoid loss. Source: [SigLIP 2](https://arxiv.org/abs/2502.14786), Figure 1._

[SigLIP](/paper%20shorts/2023/10/01/sigmoid-loss-for-language-image-pre-training-siglip.html) changed how image-text pairs compete. SigLIP 2 changes the supervision mixture. Captioning adds token-level language pressure. Self-distillation and masked prediction add local visual pressure. Online curation changes which examples survive training.

The result is a more general visual encoder without abandoning separately computable image and text embeddings. That preserves the retrieval interface and its systems advantages. It also makes attribution harder: the paper combines several established techniques, so no single objective explains the full gain.

## High-Level Takeaways

- SigLIP 2 extends the dual encoder by widening its supervision, not by changing its inference interface.
- Captioning and self-supervision make spatial and dense features less optional during pretraining.
- The combined recipe improves many capabilities, but its coupled changes make the value of each ingredient harder to isolate.
