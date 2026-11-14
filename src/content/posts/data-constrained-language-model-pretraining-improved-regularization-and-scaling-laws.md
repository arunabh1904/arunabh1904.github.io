---
title: 'Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws'
date: '2026-06-05T00:00:00.000Z'
section: paper-shorts
postSlug: data-constrained-language-model-pretraining-improved-regularization-and-scaling-laws
legacyPath: /paper shorts/2026/06/05/data-constrained-language-model-pretraining-improved-regularization-and-scaling-laws.html
tags: [Scaling Laws, Regularization, Data Efficiency]
field: 'Language Models'
summary: '2026 – Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws'
---

## 2026 – Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws

**Paper:** [Xu et al., arXiv:2606.06888v2](https://arxiv.org/abs/2606.06888v2) · [Code](https://github.com/yixinw-lab/dc_pretrain)

## Summary

> When unique data is fixed and compute is plentiful, random input masking can improve an already strongly regularized autoregressive model. The paper pairs that training intervention with SoftQ, a scaling law in which the value of more data depends on model size. Its approximately 1.3× data-efficiency estimate comes from fitted infinite-model limits, not a measured reduction in training FLOPs.

## Core Insights

### Compare against a properly regularized baseline

The starting question is whether masked diffusion models' advantage on repeated data comes from their objective or from weak regularization in the autoregressive comparison. At 257M parameters and 100M unique DCLM tokens, stronger weight decay reduces autoregressive validation loss from 3.88 to 3.42 and diffusion loss from 3.60 to 3.48. This weakens a simple claim that diffusion is inherently better in this regime.

Those numbers are not a clean ranking of likelihoods: autoregressive evaluation uses exact negative log-likelihood, diffusion uses a bound, and the checkpoint-selection protocols differ. The more informative intervention is within the autoregressive model. Masked-input regularization, or MIR, adds a next-token loss on randomly masked inputs to the clean-input loss. The architecture and ordinary autoregressive decoding remain unchanged.

The selected masking rate is sampled between zero and 0.5 per sequence, and the auxiliary loss has weight 0.4. MIR requires two forward passes per batch. Its claim concerns extracting more value from fixed unique data, with extra compute available; it is not a free improvement at matched training cost.

![Source Figure 1 shows MIR's validation-loss gains and the model-size-dependent value of unique data.](/assets/images/masked-input-regularization-softq-source-figure-1.png)
*Fig 1: Left, MIR improves a strongly regularized baseline across five model sizes, with five-seed error bars. Right, the gap between unique-data budgets grows with model size; SoftQ follows that pattern while the additive baseline predicts a constant gap. | source: [Paper, Figure 1](https://arxiv.org/html/2606.06888v2#S1.F1)*

[Open the source figure at full size](/assets/images/masked-input-regularization-softq-source-figure-1.png).

### The measured gain is consistent but bounded

On 100M unique DCLM tokens, MIR improves every matched seed across the 72M–1.4B ladder. At 1.4B, mean validation loss falls from 3.347 to 3.317. Gains also transfer to Stack-V2 when the hyperparameters are chosen on DCLM. This supports robustness to input corruption as a useful addition to weight decay, rather than an effect confined to one text sample.

Downstream results improve six of eight reported metrics. BoolQ rises from 45.11% to 55.29%, while PiQA and LAMBADA accuracy decline slightly. These are small-data models with limited absolute capability, so the large BoolQ percentage-point change should be read alongside the mixed task results. The appendix's toy memorization model supplies intuition for masking; it does not establish a complete mechanism for Transformer pretraining.

### SoftQ models a tuned loss surface

The scaling study tunes epochs, learning rate, and weight decay for each of 20 model-size/data-budget combinations. It estimates the best attainable loss under that search, rather than the loss at a fixed compute budget. SoftQ couples capacity $N$ and unique tokens $U$:

$$
L(N,U)=E+\left(\frac{A}{N^\rho}+\frac{B}{U^{\rho/(1+\alpha)}}\right)^{\alpha/\rho}.
$$

The extra shape parameter $\rho$ controls the transition between model-limited and data-limited behavior. On held-out 400M-token configurations, SoftQ's loss RMSE is 0.00595 versus 0.03106 for the additive Chinchilla form. An independent published grid gives the same fit ranking. This is evidence for the coupled shape within the studied regime, not a replacement for all [Chinchilla-style compute allocation](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html).

The 1.28–1.34× unique-data estimate matches fitted MIR asymptotes to the baseline's infinite-model curve at 200M–400M tokens. It therefore depends on both extrapolations. Heavy per-cell tuning, models no larger than 1.4B, and a fixed architecture and optimizer limit portability. A practical next test is whether the improvement survives a transferable hyperparameter recipe and an explicitly matched compute budget.

## High-Level Takeaways

- Strong regularization changes the autoregressive-versus-diffusion comparison on repeated data.
- MIR adds training work while retaining the same inference interface.
- The value of unique data can depend on model capacity, motivating a coupled loss surface.
- Fitted data-equivalence and measured compute efficiency are different claims.
