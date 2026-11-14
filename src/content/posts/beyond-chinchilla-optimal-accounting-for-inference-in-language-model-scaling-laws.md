---
title: "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws"
date: '2023-12-31T00:00:00.000Z'
section: paper-shorts
postSlug: beyond-chinchilla-optimal-accounting-for-inference-in-language-model-scaling-laws
legacyPath: /paper shorts/2023/12/31/beyond-chinchilla-optimal-accounting-for-inference-in-language-model-scaling-laws.html
tags: [Language Models, Scaling Laws, Inference]
field: Language Models
summary: "2023 – Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws"
---

## 2023 – Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws

**Paper:** [Sardana et al., arXiv:2401.00448v3](https://arxiv.org/abs/2401.00448v3). First submitted December 31, 2023; this note reads the April 2025 revision.

## Summary

> A model that minimizes training compute need not minimize the cost of serving the resulting capability. Beyond Chinchilla-Optimal adds lifetime inference demand to the allocation problem, favoring smaller models trained longer when enough requests amortize the extra training. Its 47-run experiment supports continued learning at high token-to-parameter ratios, while also showing that ordinary Chinchilla fits overestimate how much that additional data buys.

## Core Insights

[Chinchilla](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html) minimizes pretraining loss under a training-compute budget. This paper fixes the desired loss and minimizes the combined training and inference cost. Let $N$ be parameters, $D_{\mathrm{tr}}$ training tokens, and $D_{\mathrm{inf}}$ total input and output tokens served over the model's lifetime. The simplified objective is

$$
\min_{N,D_{\mathrm{tr}}}\;6ND_{\mathrm{tr}}+2ND_{\mathrm{inf}}
\quad\text{subject to}\quad
E+\frac{A}{N^\alpha}+\frac{B}{D_{\mathrm{tr}}^\beta}=\ell.
$$

The inference term changes the optimum because every future token pays for the chosen parameter count. Training longer can purchase a smaller model with the same predicted loss. The calculation assumes sufficient unique training data, an advance estimate of demand, and equal demand for equal-loss models. It does not optimize task-specific quality or account for demand increasing when latency falls.

### More training still helps, but the extrapolation becomes optimistic

The empirical sweep trains 47 MPT-style models from 150M to 6B parameters on web text and code, without repeating the corpus. The exact data mixture is not specified. Models use ALiBi, grouped-query attention, 4,096-token context, and Lion; learning rates and batch sizes vary with scale. Reported loss is the final training loss smoothed over ten batches, rather than an independently held-out validation loss.

Only the 150M model reaches 10,000 tokens per parameter. The larger-model sweep is uneven: 370M reaches 1,000, 2.5B reaches 500, and 6B is tested at 20. The source plot therefore supports continued improvement in the tested ranges, not a demonstrated 10,000-token ratio for every model size.

![Source Figure 3a shows pretraining loss declining with tokens per parameter across six model sizes, with only 150M extending to 10,000.](/assets/images/beyond-chinchilla-source-figure-3a.png)
*Fig 1: Additional unique tokens continue to reduce loss, but the tested range narrows substantially for larger models. The longest 150M run also uses a larger batch than its shorter counterparts. This is the original Figure 3a panel. | source: [Beyond Chinchilla-Optimal, Figure 3](https://arxiv.org/html/2401.00448v3#S2.F3)*

[Open the figure at full size](/assets/images/beyond-chinchilla-source-figure-3a.png)

Downstream evaluation spans knowledge, commonsense, reading comprehension, language understanding, and symbolic problems. The overall Gauntlet score averages task accuracies after chance correction and normalization. That aggregate improves with training, but category-level behavior is less reliable; many symbolic tasks remain near chance. Continued loss improvement does not establish every desired capability.

Fitting the loss law using only runs below roughly 100 tokens per parameter predicts overly large gains in the longer-training regime. Including higher-ratio runs changes the fitted curves, yet none of the fits captures the long 150M runs well. The economic argument for overtraining survives; its precise optimal allocation requires measurements in the regime being recommended. The later [overtraining study by Gadre et al.](/paper%20shorts/2024/03/13/language-models-scale-reliably-with-overtraining-and-on-downstream-tasks.html) examines that forecasting problem with a separate controlled testbed.

### Serving dollars depend on utilization, not just FLOPs

The cost extension separates training, prompt processing, and autoregressive generation. Its illustrative setting assumes 50% utilization for training and prompt processing, 1% for generation, requests with 70 input and 215 output tokens, and INT8 inference on different A100 hardware from training. These are scenario assumptions and historical prices, not measurements of a current serving stack.

Under that scenario, the model predicts that replacing a Chinchilla-style 30B model with a 16B model trained on 3.35T tokens saves 17% at 1.5B requests. This is a modeled equal-loss comparison, not a deployed pairwise experiment. Fixed utilization also omits model-dependent batching, sequence length, and latency constraints.

For a deployment decision, I would estimate demand and measure prefill/decode costs on the intended hardware, then validate the proposed smaller model's loss and target-task quality. A favorable lifetime-cost calculation cannot rescue an inaccurate long-training forecast.

## High-Level Takeaways

- Inference demand changes the training allocation: additional pretraining can be worthwhile when it reduces the size of a frequently served model.
- The empirical sweep supports continued improvement within its tested ranges; only the smallest model reaches 10,000 tokens per parameter.
- Fitting near the training-compute optimum can overestimate the gains from extreme overtraining.
- Dollar savings require workload and utilization assumptions, while equal pretraining loss remains a proxy for the capability being served.
