---
title: 'Scaling Laws for Optimal Data Mixtures'
date: '2025-07-12T09:00:00.000Z'
section: paper-shorts
postSlug: scaling-laws-for-optimal-data-mixtures
legacyPath: /paper shorts/2025/07/12/scaling-laws-for-optimal-data-mixtures.html
tags: [Scaling Laws]
field: 'Multimodal Scaling & Data Mixtures'
summary: "2025 – Scaling Laws for Optimal Data Mixtures"
---

## 2025 – Scaling Laws for Optimal Data Mixtures

**arXiv:** [2507.09404](https://arxiv.org/abs/2507.09404)  
**Conference:** Technical report

**Summary:** Scaling Laws for Optimal Data Mixtures predicts loss from model size $N$, training tokens $D$, and a domain-weight vector $h$. The objective is to choose data mixtures systematically rather than through trial and error at full pre-training scale.

## Paper Insights

The authors validate the prediction framework across LLM, native multimodal, and large vision-model pre-training. Their claim is operational: a few small runs can estimate parameters that extrapolate to new mixtures and larger scales, then yield compute-aware optimal domain weights for a chosen target.

| Input | Use |
| --- | --- |
| $N$ and $D$ | Defines the scale and budget. |
| Domain weights $h$ | Defines the candidate mixture. |
| Target-domain loss | Defines what “optimal” means. |

## Decision Lens

This paper informs how to allocate a finite pretraining budget across domains when the target metric is known. Small proxy runs fit loss as a function of model size, total data, and mixture weights; optimization then chooses a compute-aware mixture for the target domain. The important conceptual move is that “optimal data” is conditional on the evaluation target, not an intrinsic property of a dataset.

Validation across language, native multimodal, and vision settings supports the fitting procedure, but it leaves distribution drift and data quality largely outside the law. The missing experiment chooses a mixture once, then carries it across a substantial scale jump and a changed model family without retuning. At ten times the corpus size, duplication and domain quality may change faster than mixture weights capture. The method is falsified if its predicted optimum repeatedly underperforms robust heuristic mixtures after including the proxy-search cost.

**Limits:** An extrapolated optimum is only as trustworthy as the small-run fit and the stability of rankings outside the observed regime.

**Takeaway:** Allocate data with an uncertainty-aware fitted model, then update the allocation as new evidence arrives.
