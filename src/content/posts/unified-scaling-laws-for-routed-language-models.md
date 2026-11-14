---
title: 'Unified Scaling Laws for Routed Language Models'
date: '2022-02-02T00:00:00.000Z'
section: paper-shorts
postSlug: unified-scaling-laws-for-routed-language-models
legacyPath: /paper shorts/2022/02/02/unified-scaling-laws-for-routed-language-models.html
tags: [Language Models, Scaling Laws, Mixture of Experts]
field: 'Language Models'
summary: '2022 – Unified Scaling Laws for Routed Language Models'
---

## 2022 – Unified Scaling Laws for Routed Language Models

**arXiv:** [2202.01169](https://arxiv.org/abs/2202.01169)

## Summary

> Clark et al. show why parameter count alone is not enough for sparsely routed Transformers. A routed model has a dense capacity $N$ and an expert count $E$: each token touches only a subset of a much larger parameter pool. Across 168 models and three routing methods, validation loss follows a bilinear law in log model size and a saturating transform of expert count, yielding an effective dense-equivalent parameter count and a practical warning that expert gains diminish with scale.

## Core Insights

### Routing splits size from compute

The experiments use decoder-only Transformers with dense base sizes from 15M to 1.3B, expert counts from 1 to 512, a 2,048-token context, and 250,000 training steps (about 130B tokens) for every model. The routed layers replace every other feed-forward block. A dense model has total parameter count $P=N$; a routed model has approximately $P\propto NE$, while the per-token inference compute remains close to the dense model of size $N$.

The three methods are Sinkhorn-BASE (S-BASE), deterministic hash routing, and reinforcement-learning routing (RL-R). Routing improves validation loss across the tested sizes. S-BASE is the strongest default, RL-R is competitive at smaller sizes but more sensitive, and hash routing has a larger initial overhead.

![Source Figure 4 comparing routed-model scaling fits](/assets/images/unified-scaling-laws-for-routed-language-models-source-figure-4.png)
*Fig 1: Fits for S-BASE, RL-R, and HASH as expert count grows. The image is cropped from source Figure 4; dotted lines include the interaction term that the separable fit misses at larger base sizes. | source: [Unified Scaling Laws for Routed Language Models](https://arxiv.org/abs/2202.01169)*

### Expert gains depend on model size

The main fit is

$$
\log L(N,E)=a\log N+b\log\widehat E+c\log N\log\widehat E+d,
$$

where $\widehat E$ saturates between an effective start and maximum expert count. The interaction term matters because the expert-scaling slope changes with $N$: larger dense models benefit less from adding experts. A separable law overpredicts performance at large $N$ and $E$; the bilinear term captures that interference, while saturation prevents extrapolating endless gains.

Changing variables from $(N,E)$ to inference FLOPs $F$ and parameter-utilization ratio $B=P/F$ makes the fit nearly shared across routing variants, including changes to the number of selected experts $K$ and routed-layer frequency $R$. Solving the fitted law for a dense model with equal loss gives an effective parameter count $\bar N$. In the measured regime, a 5M base model with 128 experts can match the fitted loss of a 55M dense model while using far less inference compute, but it still stores many more parameters.

### The recommendation is bounded by the sweep

The paper's practitioner guidance is specific: use routing for the tested base sizes, target roughly 64–128 experts, use $K=1$, and route layers at frequency $0.5\le R\le1$. The predicted routing cutoff is method-dependent: S-BASE reaches a much larger effective parameter count before its benefit disappears than RL-R or hash routing. These numbers depend strongly on the fixed 130B-token training budget, the English corpus, and the 15M–1.3B base-size range.

Appendix F checks that the analysis is qualitatively robust to token count, and Appendix C explains the all-to-all communication and capacity-factor trade-offs in expert parallelism. Those systems costs are not in the loss law. A routed model can be compute-efficient per token and still be expensive to train or serve because it stores, moves, and synchronizes a larger parameter pool.

## High-Level Takeaways

- MoE scaling needs at least two axes: active model size and total expert capacity.
- A log-size interaction term captures diminishing expert gains; a separable power law is too optimistic at large sizes.
- Effective parameter count is useful for comparing dense and routed models, but it hides storage and communication costs.
- S-BASE and roughly 64–128 experts are sensible defaults only inside this paper's training and size regime.
- Routing laws should be validated across both $N$ and $E$; a single model size cannot establish how an architecture scales.
