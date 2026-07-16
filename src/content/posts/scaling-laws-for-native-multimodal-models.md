---
title: 'Scaling Laws for Native Multimodal Models'
date: '2025-04-10T09:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-native-multimodal-models
legacyPath: /paper shorts/2025/04/10/scaling-laws-for-native-multimodal-models.html
tags: [Multimodal AI]
field: Omni-Models
summary: '2025 – Scaling laws comparing native early-fusion and late-fusion multimodal models.'
---

## 2025 – Scaling Laws for Native Multimodal Models

**arXiv:** [2504.07951](https://arxiv.org/abs/2504.07951)  
**Conference:** ICCV 2025 (oral)

**Summary:** This paper trains 457 native multimodal models with different architectures and mixtures to compare early fusion, late fusion, and modality-expert designs. It revisits the assumption that attaching a pretrained vision encoder is inherently better than learning multimodal representations from the start.

## Paper Insights

The authors find no inherent late-fusion advantage in their study. Early-fusion models perform better at lower parameter counts, train more efficiently, and are simpler to deploy; adding Mixture of Experts lets the model learn modality-specific weights and improves performance.

| Decision | Evidence in the paper |
| --- | --- |
| Early vs. late fusion | No inherent advantage for late fusion in the studied regime. |
| Small-model efficiency | Early fusion is stronger at lower parameter counts. |
| Specialization | MoE modality-specific weights improve the native model. |

**Limits:** Scaling-law conclusions are conditional on the architectures, modalities, objectives, and mixtures studied; they should guide a proxy-run plan, not replace one.

**Takeaway:** Treat early fusion plus learned specialization as a serious baseline rather than assuming a pretrained visual tower is necessary.

