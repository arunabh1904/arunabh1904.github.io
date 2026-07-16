---
title: 'Scaling Laws for Generative Mixed-Modal Language Models'
date: '2023-01-10T09:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-generative-mixed-modal-language-models
legacyPath: /paper shorts/2023/01/10/scaling-laws-for-generative-mixed-modal-language-models.html
tags: [Scaling Laws]
field: Omni-Models
summary: '2023 – Mixed-modal scaling laws that model both modality contributions and interactions.'
---

## 2023 – Scaling Laws for Generative Mixed-Modal Language Models

**arXiv:** [2301.03728](https://arxiv.org/abs/2301.03728)  
**Conference:** Technical report

**Summary:** This work runs more than 250 experiments across seven modalities, model sizes from 8M to 30B parameters, and 5–100B training tokens. It extends unimodal scaling laws with terms for both individual modality contributions and cross-modal synergy or competition.

## Paper Insights

The useful shift is from asking whether a mixture is good to estimating how each modality changes loss under a particular model and data budget. The paper also reports modality alternation during training, hyperparameter guidance, and links between mixed-modal competition and training stability; a 30B speech-text model serves as a larger validation run.

| Component | Role |
| --- | --- |
| Unimodal terms | Capture the usual model- and data-scaling effects. |
| Interaction term | Represents synergy or competition between modalities. |
| Proxy experiments | Estimate a mixture before the expensive run. |

**Limits:** Scaling fits describe the studied training regime; interactions can change with model family, tokenization, and evaluation target.

**Takeaway:** Data mixture is an optimization variable with interactions, not a fixed recipe to inherit.

