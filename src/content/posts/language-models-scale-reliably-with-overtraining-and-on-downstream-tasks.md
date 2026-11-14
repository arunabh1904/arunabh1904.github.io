---
title: "Language Models Scale Reliably with Over-Training and on Downstream Tasks"
date: '2024-03-13T00:00:00.000Z'
section: paper-shorts
postSlug: language-models-scale-reliably-with-overtraining-and-on-downstream-tasks
legacyPath: /paper shorts/2024/03/13/language-models-scale-reliably-with-overtraining-and-on-downstream-tasks.html
tags:
  - Scaling Laws
  - Pretraining
field: "Language Models"
summary: "2024 – Language Models Scale Reliably with Over-Training and on Downstream Tasks"
---

## 2024 – Language Models Scale Reliably with Over-Training and on Downstream Tasks

**Paper:** [Gadre et al., arXiv:2403.08540v2](https://arxiv.org/abs/2403.08540v2) · **Experiments:** [mlfoundations/scaling](https://github.com/mlfoundations/scaling)

## Summary

> Small controlled runs can predict both the loss of heavily overtrained models and average downstream error, but those are different forecasting tasks with different costs. The paper predicts C4 validation loss for a 1.4B model trained on 900B RedPajama tokens within 0.7% relative error using fitting runs costing about 300 times less compute. Predicting average benchmark error requires a more expensive fitting population, while individual-task forecasts remain substantially noisier.

## Core Insights

A model can be deliberately trained past the token-to-parameter ratio that minimizes training loss per FLOP. This spends more training compute to obtain a smaller deployed model. The paper asks whether the resulting loss remains predictable, rather than solving the deployment-budget optimization itself.

Let $M=D/N$ be training tokens per parameter and $C=6ND$ approximate training compute. Rewriting the [Chinchilla loss model](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html) under the simplifying assumption of equal parameter and data exponents gives

$$
L(C,M)=E+\left(aM^\eta+bM^{-\eta}\right)C^{-\eta}.
$$

The exponent of compute stays fixed as $M$ changes, while the multiplier shifts the curve. The source plot tests that prediction across three training corpora using a common C4 evaluation loss.

![Source Figure 2 plots reducible C4 loss against training FLOPs for C4, RedPajama, and RefinedWeb models at different token-to-parameter ratios.](/assets/images/overtraining-downstream-scaling-source-figure-2.png)
*Fig 1: Curves at different token multipliers have approximately parallel slopes within each training distribution. Overtraining shifts the loss–compute trade-off without requiring a separate compute exponent for every multiplier in this fitted regime. | source: [Gadre et al., Figure 2](https://arxiv.org/html/2403.08540v2#S2.F2)*

[Open the figure at full size](/assets/images/overtraining-downstream-scaling-source-figure-2.png)

### The small-model recipe is selected, not arbitrary

The 104-model testbed spans 11M–6.9B parameters and token multipliers from 5 to 640 on C4, RedPajama, and RefinedWeb. Training samples without replacement; unavailable high-token combinations are omitted. Thus overtraining here does not mean repeatedly traversing a fixed small corpus.

Models share GPT-NeoX tokenization, a 2,048-token context, rotary embeddings, and an OpenLM decoder architecture with query/key LayerNorm. The optimizer uses AdamW with warmup and cosine decay. Before building the testbed, a 435-run architecture and hyperparameter search selects small configurations that follow a coherent scaling trend. Some unusually strong intermediate configurations are excluded because they lie off that trend; Appendix F associates them with more optimizer steps.

That preparation limits the “300× cheaper” interpretation. The quoted ratio concerns the selected runs used to fit the law, not the entire exploratory search. A new organization does not necessarily obtain the same reliable fitting population from five arbitrary runs.

### Downstream prediction adds a second fitted relationship

The authors fit average top-1 error as $\operatorname{Err}(L)=\epsilon-k\exp(-\gamma L)$, then compose it with the loss forecast. The default metric averages 17 evaluations selected because at least one 154M model exceeds chance by ten percentage points. The suite covers tasks such as ARC-Easy, BoolQ, HellaSwag, LAMBADA, and PIQA. Appendix ablations also retain predictability when averaging all 46 available evaluations.

Loss fitting uses four small near-compute-optimal runs plus one overtrained 11M run. Downstream fitting additionally uses a 1.4B model. This raises the reported fitting investment from approximately 100 to 1,000 A100 hours. Removing that larger point changes relative error on the 6.9B RedPajama model's 17-task average from 0.05% to 10.64%.

The 0.05% figure is relative error in a forecast of aggregate benchmark error, not an accuracy improvement. At the same target scale, individual RedPajama task-error forecasts have relative errors of 5.21% for ARC-Easy, 14.39% for LAMBADA, and 25.73% for zero-shot HellaSwag. A highly accurate average can coexist with materially wrong task-specific predictions.

### Recipe and evaluation shifts define the boundary

The loss-to-error mapping is fitted separately for each training distribution. A model with better loss on one validation set need not have better downstream performance when its training corpus differs. The later [loss-perspective emergence study](/paper%20shorts/2024/03/23/understanding-emergent-abilities-from-the-loss-perspective.html) complements this result by examining individual task curves and their near-chance regions.

The scaling fit also becomes unreliable at very low token multipliers and on some distribution shifts, including code and Penn Tree Bank evaluation after English-filtered C4 training. In-distribution loss is not immune: the heavily overtrained RedPajama run deviates by 15.4% on its corresponding Paloma evaluation, outperforming the prediction. The authors suggest near-duplicates as a possible explanation, rather than establishing that cause.

The practical lesson is to budget separate validation for the intended forecast. Average downstream behavior may be predictable after a modest larger-scale anchor; a particular difficult capability, a changed corpus, or performance after post-training requires new evidence.

## High-Level Takeaways

- Overtraining can preserve predictable loss scaling within a controlled recipe, even when both model size and tokens per parameter exceed the fitting range.
- Cheap loss prediction does not imply equally cheap downstream prediction; the latter benefits from a larger anchor run.
- Aggregate benchmark error can be forecast accurately while individual tasks remain poorly predicted.
- The fitting runs, configuration-selection process, training distribution, and evaluation metric all belong in the claim's scope.
