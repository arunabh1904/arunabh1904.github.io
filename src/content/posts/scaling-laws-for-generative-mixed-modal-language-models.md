---
title: 'Scaling Laws for Generative Mixed-Modal Language Models'
date: '2023-01-10T09:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-generative-mixed-modal-language-models
legacyPath: /paper shorts/2023/01/10/scaling-laws-for-generative-mixed-modal-language-models.html
tags: [Scaling Laws]
field: 'Multimodal Scaling & Data Mixtures'
summary: "2023 – Scaling Laws for Generative Mixed-Modal Language Models"
---

## 2023 – Scaling Laws for Generative Mixed-Modal Language Models

**arXiv:** [2301.03728](https://arxiv.org/abs/2301.03728)  
**Conference:** Technical report

**Summary:** Scaling Laws for Generative Mixed-Modal Language Models runs more than 250 experiments across seven modalities, model sizes from 8M to 30B parameters, and 5–100B training tokens. Its law includes both individual modality contributions and cross-modal synergy or competition.

## Paper Insights

The useful shift is from asking whether a mixture is good to estimating how each modality changes loss under a particular model and data budget. The paper also reports modality alternation during training, hyperparameter guidance, and links between mixed-modal competition and training stability; a 30B speech-text model provides a larger validation run.

| Component | Role |
| --- | --- |
| Unimodal terms | Capture the usual model- and data-scaling effects. |
| Interaction term | Represents synergy or competition between modalities. |
| Proxy experiments | Estimate a mixture before the expensive run. |

## Decision Lens

This paper informs how to choose a modality mixture before committing to an expensive mixed-modal run. It extends unimodal loss scaling with an interaction term that represents synergy or competition between modalities, then estimates the curve from proxy experiments. The 30B speech–text validation matters because it tests whether small-run mixture behavior extrapolates beyond the fitting regime.

The curve establishes predictability for the studied modalities and budgets, not a universal law of beneficial mixing. The most important missing test is transfer across architectures, tokenizers, and data-quality regimes without refitting the interaction from scratch. At ten times the scale, changing data entropy and curriculum order can make a fixed interaction term nonstationary. The framework is falsified if mixtures selected by proxy fits consistently lose to simple baselines such as temperature sampling when evaluated at the target scale and full training cost.

**Limits:** Scaling fits describe the studied training regime; interactions can change with model family, tokenization, and evaluation target.

**Takeaway:** Data mixture is an optimization variable with interactions, not a fixed recipe to inherit.
