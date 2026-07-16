---
title: 'Wan: Open and Advanced Large-Scale Video Generative Models'
date: '2025-03-26T09:00:00.000Z'
section: paper-shorts
postSlug: wan-open-and-advanced-large-scale-video-generative-models
legacyPath: /paper shorts/2025/03/26/wan-open-and-advanced-large-scale-video-generative-models.html
tags: [Video Generation]
field: 'Video & Interactive World Models'
summary: '2025 – Wan: an open video-generation suite built around diffusion transformers, a VAE, and large-scale curation.'
---

## 2025 – Wan

**arXiv:** [2503.20314](https://arxiv.org/abs/2503.20314)  
**GitHub:** [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)  
**Conference:** Technical report

**Summary:** Wan is an open suite of video foundation models built on diffusion transformers. The report centers the video VAE, scalable pre-training, data curation, automated evaluation, and model-size/data-size scaling.

## Paper Insights

Wan supplies a systems-oriented video reference rather than only a generative-model result. It releases 1.3B and 14B models, supports several downstream tasks such as image-to-video and editing, and reports that the 1.3B model can run with 8.19 GB of VRAM.

| Component | Decision it informs |
| --- | --- |
| Video VAE | How aggressively to compress space and time before the transformer. |
| Data pipeline | Caption quality, motion filtering, and curation affect the usable training signal. |
| Model family | How to trade quality against a consumer-scale deployment target. |

## Decision Lens

Wan informs how to divide a video-generation budget among latent compression, transformer scale, data curation, and deployability. A video VAE compresses spatial and temporal structure before the generative model operates, while the released 1.3B and 14B variants expose a concrete quality–resource tradeoff; the smaller model's reported 8.19 GB requirement makes consumer hardware an explicit design target.

The broad task support shows that the stack is reusable, but it does not reveal whether model scale, VAE quality, captions, or motion filtering produces the largest marginal gain. A compute- and data-matched component ablation with temporal-fidelity metrics is the missing decision evidence. At ten times the duration or resolution, VAE artifacts and temporal drift can compound faster than model quality improves. The scaling story would be falsified if a smaller transformer with better data or weaker compression matches the 14B model on motion consistency and prompt adherence at much lower cost.

**Limits:** Video quality does not establish action-conditioned dynamics, causal control, or long-horizon world consistency.

**Takeaway:** Treat the video autoencoder and data pipeline as first-class architecture choices, not preprocessing details.
