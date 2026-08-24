---
title: 'Scaling Laws for Native Multimodal Models'
date: '2025-04-10T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-native-multimodal-models
legacyPath: /paper shorts/2025/04/10/scaling-laws-for-native-multimodal-models.html
tags: [Multimodal AI]
field: 'Multimodal Scaling & Data Mixtures'
summary: "2025 – Scaling Laws for Native Multimodal Models"
---

## 2025 – Scaling Laws for Native Multimodal Models

**arXiv:** [2504.07951](https://arxiv.org/abs/2504.07951)  
**Conference:** ICCV 2025 (oral)

### Method and reported result

Scaling Laws for Native Multimodal Models trains 457 models across different architectures and mixtures to compare early fusion, late fusion, and modality-expert designs. The study tests whether attaching a pretrained vision encoder is inherently better than learning multimodal representations from the start.

## Summary

> The authors find no inherent late-fusion advantage in their study. Early-fusion models perform better at lower parameter counts, train more efficiently, and are simpler to deploy; adding Mixture of Experts lets the model learn modality-specific weights and improves performance.

## Core Insights

![Scaling curves comparing early fusion late fusion and mixture-of-experts native multimodal models](/assets/images/scaling-laws-for-native-multimodal-models-paper-figure.png)
*Figure 1 summarizes the architecture trade-off: early and late fusion reach similar loss scaling, but allocate compute differently between parameters and tokens; sparse early fusion shifts that allocation further toward data. source: [Scaling Laws for Native Multimodal Models](https://arxiv.org/abs/2504.07951)*

![Figure 12 from Scaling Laws for Native Multimodal Models](/assets/images/scaling-laws-for-native-multimodal-models-source-figure-12.webp)
*Figure 12 MoE specialization frequency. Percentage of text and image tokens routed to each expert on interleaved data from Obelics. Experts are ordered for better visualization. The first layer shows the highest amount of unimodal experts. source: [Scaling Laws for Native Multimodal Models](https://arxiv.org/abs/2504.07951)*


| Decision | Evidence in the paper |
| --- | --- |
| Early vs. late fusion | No inherent advantage for late fusion in the studied regime. |
| Small-model efficiency | Early fusion is stronger at lower parameter counts. |
| Specialization | MoE modality-specific weights improve the native model. |

## High-Level Takeaways

- This study informs the architectural choice between early fusion and late fusion for native multimodal training. In the reported regime, a shared early-fusion stream is simpler and more efficient at smaller parameter counts, while modality-aware mixture-of-experts weights recover specialization without duplicating the whole model. The result argues that modality-specific capacity can live inside a unified transformer rather than behind separate towers.
- The evidence rejects an inherent late-fusion advantage only for the tested modalities, tokenization, and compute range. A decisive missing comparison would equalize active parameters, communication cost, context length, and modality-specific preprocessing across both designs. At ten times the number of modalities or sequence length, shared attention and routing contention may reverse the result. The claim would be falsified if late fusion becomes more sample- or compute-efficient once high-bandwidth modalities and matched systems costs are included.
- Scaling-law conclusions are conditional on the architectures, modalities, objectives, and mixtures studied; they should guide a proxy-run plan, not replace one.
- Treat early fusion plus learned specialization as a serious baseline rather than assuming a pretrained visual tower is necessary.
