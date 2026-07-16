---
title: 'TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation'
date: '2024-12-04T09:00:00.000Z'
section: paper-shorts
postSlug: tokenflow-unified-image-tokenizer-for-multimodal-understanding-and-generation
legacyPath: /paper shorts/2024/12/04/tokenflow-unified-image-tokenizer-for-multimodal-understanding-and-generation.html
tags: [Multimodal AI]
field: Omni-Models
summary: '2024 – TokenFlow: a dual-codebook tokenizer for visual semantics and reconstruction detail.'
---

## 2024 – TokenFlow

**arXiv:** [2412.03069](https://arxiv.org/abs/2412.03069)  
**GitHub:** [ByteFlow-AI/TokenFlow](https://github.com/ByteFlow-AI/TokenFlow)  
**Conference:** CVPR 2025

**Summary:** TokenFlow is a unified image tokenizer built around two aligned codebooks: one for semantic features used by understanding and one for pixel-level detail used by generation. It aims to avoid the usual tradeoff of using a reconstruction-focused VQ encoder for both tasks.

## Paper Insights

The key mechanism is dual codebooks connected through a shared mapping, so shared indices expose both semantic and fine-grained information. The paper reports 7.2% average improvement over LLaVA-1.5 13B on its understanding comparison, FID 0.63 at 384×384 reconstruction, and GenEval 0.55 at 256×256 autoregressive generation.

| Signal | Reported value | Why it matters |
| --- | --- | --- |
| Understanding | 7.2% average improvement over LLaVA-1.5 13B | Tests whether discrete input can retain semantic value. |
| Reconstruction | FID 0.63 at 384×384 | Tests fine visual detail. |
| Generation | GenEval 0.55 at 256×256 | Tests the generation side of the tokenizer. |

**Limits:** The design adds tokenizer complexity; results depend on the selected comparisons and resolutions.

**Takeaway:** The tokenizer is an architectural decision: semantic and reconstruction features can be aligned without being identical.

