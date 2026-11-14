---
title: 'Training Compute-Optimal Large Language Models'
date: '2022-03-29T00:00:00.000Z'
section: paper-shorts
postSlug: training-compute-optimal-large-language-models
legacyPath: /paper shorts/2022/03/29/training-compute-optimal-large-language-models.html
tags: [Language Models, Scaling Laws, Pretraining]
field: 'Language Models'
summary: '2022 – Training Compute-Optimal Large Language Models'
---

## 2022 – Training Compute-Optimal Large Language Models

**arXiv:** [2203.15556](https://arxiv.org/abs/2203.15556)

## Summary

> Hoffmann et al. (Chinchilla) ask how to divide a fixed training-FLOP budget between model parameters and tokens. Across more than 400 language models, their three estimation methods agree on a near-balanced frontier: both $N$ and $D$ should grow approximately as $C^{1/2}$. The conclusion is about compute-optimal pretraining loss under a particular data and schedule regime, not a universal ratio for every downstream or inference objective.

## Core Insights

### The question is allocation, not whether scale helps

Kaplan's earlier study showed smooth loss curves but favored parameter growth. Chinchilla varies both model size and token budget directly: more than 400 models span roughly 70M–16B parameters and 5B–500B tokens. It uses fixed-size data sweeps, IsoFLOP profiles, and a parametric fit

$$
\widehat L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
$$

All three approaches point to $N_{\mathrm{opt}}\propto C^{0.5}$ and $D_{\mathrm{opt}}\propto C^{0.5}$, unlike Kaplan's approximate $C^{0.73}/C^{0.27}$ split. The practical reading is “grow parameters and tokens together” within the measured range, not “always train for exactly twenty tokens per parameter.” The fitted constants and data mix still determine the finite-scale optimum.

![Chinchilla Figure 3: Iso-loss contours and compute-optimal frontier](/assets/images/chinchilla-compute-frontier-paper-figure.png)
*Fig 1: The contours show how model size and tokens trade off at fixed loss; the lower envelope is the compute-efficient frontier used to choose Chinchilla. | source: [Training Compute-Optimal Large Language Models, Figure 3](https://arxiv.org/abs/2203.15556)*

### Training schedule is part of the law

The models use a matched cosine horizon: the decay schedule is set to the number of tokens in that run. The authors report that overestimating the horizon by more than about 25% hurts loss, which is a reminder that an IsoFLOP point is not just $(N,D)$; it includes an optimization schedule. C4 and GitHub fits produce similar exponents, but this does not remove sensitivity to corpus quality, tokenizer, architecture, and optimizer.

The chosen model, Chinchilla, has about 70B parameters and 1.4T tokens. It uses roughly the same training FLOPs as Gopher's 280B parameters and 300B tokens, yet outperforms Gopher across the reported Pile and capability evaluations: Pile perplexity improves from 7.75 to 7.16, MMLU rises by 7.6 points, and BIG-bench improves by 10.7 points. The comparison is valuable because the two models share a family and compute budget, but it is not a complete causal study of data quality or post-training.

### The boundary is larger than the headline ratio

Appendix A shows a non-uniform dataset mixture and more than one pass over some sources, including Wikipedia. The main evidence uses model sizes below 16B and assumes the relevant training point is before one full epoch over the effective corpus. The paper itself flags the concavity of the fitted loss surface, data quality and leakage, and the limited number of directly comparable large runs.

The result therefore changes the default experiment: if a model looks undertrained, add tokens before adding parameters. But for a constrained language, repeated data, a changing mixture, or an inference-cost objective, the optimal allocation can move. [Data-constrained scaling](/paper%20shorts/2023/05/25/scaling-data-constrained-language-models.html) and the replication/diagnostic notes below test exactly those boundaries.

## High-Level Takeaways

- Chinchilla estimates a near-balanced compute frontier: parameters and training tokens grow at roughly the same exponent.
- The schedule and data mixture are part of the experiment; a token/parameter ratio is not portable by itself.
- The 70B/1.4T Chinchilla run demonstrates the value of correcting allocation at a matched compute budget.
- Capability gains track the reported evaluations but do not prove that loss is a universal capability proxy.
- Read [the replication attempt](/paper%20shorts/2024/04/15/chinchilla-scaling-a-replication-attempt.html) for why fitted coefficients and uncertainty deserve separate scrutiny.
