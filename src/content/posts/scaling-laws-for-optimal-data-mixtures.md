---
title: 'Scaling Laws for Optimal Data Mixtures'
date: '2025-07-12T09:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-optimal-data-mixtures
legacyPath: /paper shorts/2025/07/12/scaling-laws-for-optimal-data-mixtures.html
tags: [Scaling Laws]
field: Omni-Models
summary: '2025 – Estimating compute-aware data mixtures from small training runs.'
---

## 2025 – Scaling Laws for Optimal Data Mixtures

**arXiv:** [2507.09404](https://arxiv.org/abs/2507.09404)  
**Conference:** Technical report

**Summary:** This paper proposes scaling laws that predict loss from model size $N$, training tokens $D$, and a domain-weight vector $h$. The objective is to choose data mixtures systematically rather than through trial and error at the full pre-training scale.

## Paper Insights

The authors validate the prediction framework across LLM, native multimodal, and large vision-model pre-training. Their claim is operational: a few small runs can estimate parameters that extrapolate to new mixtures and larger scales, then yield compute-aware optimal domain weights for a chosen target.

| Input | Use |
| --- | --- |
| $N$ and $D$ | Defines the scale and budget. |
| Domain weights $h$ | Defines the candidate mixture. |
| Target-domain loss | Defines what “optimal” means. |

**Limits:** An extrapolated optimum is only as trustworthy as the small-run fit and the stability of rankings outside the observed regime.

**Takeaway:** Allocate data with an uncertainty-aware fitted model, then update the allocation as new evidence arrives.

