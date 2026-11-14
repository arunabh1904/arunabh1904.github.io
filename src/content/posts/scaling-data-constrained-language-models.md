---
title: 'Scaling Data-Constrained Language Models'
date: '2023-05-25T00:00:00.000Z'
section: paper-shorts
postSlug: scaling-data-constrained-language-models
legacyPath: /paper shorts/2023/05/25/scaling-data-constrained-language-models.html
tags: [Language Models, Scaling Laws, Data Mixtures]
field: 'Multimodal Scaling & Data Mixtures'
summary: '2023 – Scaling Data-Constrained Language Models'
---

## 2023 – Scaling Data-Constrained Language Models

**arXiv:** [2305.16264](https://arxiv.org/abs/2305.16264)

## Summary

> Muennighoff et al. extend compute-optimal scaling to the regime where unique data runs out. Across more than 400 models from 10M to 9B parameters, they find that repeating the full corpus for up to roughly four epochs is nearly as good as fresh data, after which the return decays quickly. Their law replaces raw tokens and parameters with effective quantities whose value saturates under repetition and excess capacity; code augmentation can extend the useful data budget by about 2×.

## Core Insights

### Repetition is useful before it is wasteful

The experiments train GPT-2-style models on carefully nested C4 subsets, varying both parameter count and epochs up to 900B total tokens. At a fixed compute budget, one epoch is usually best per FLOP, but the difference is often negligible through four epochs and downstream performance remains similar. At 40 epochs, repeating is effectively worthless. This is not a license to upsample a tiny subset indefinitely: the paper repeats the entire available corpus and evaluates on held-out data.

![Source Figure 1: return and allocation when data is repeated](/assets/images/scaling-data-constrained-language-models-source-figure-1.png)
*Fig 1: Repeated data initially preserves useful return, then flattens; under the constraint, the efficient frontier shifts toward smaller models trained for more epochs. | source: [Scaling Data-Constrained Language Models, Figure 1](https://arxiv.org/abs/2305.16264)*

### Fit effective data, not token count

Let $U_D$ be unique tokens, $R_D$ the number of repeats, and $D_C$ the available data budget. The paper models effective data as

$$
D'=U_D+U_D R_D^*\left(1-e^{-R_D/R_D^*}\right).
$$

It uses the symmetric construction

$$
N'=U_N+U_N R_N^*\left(1-e^{-R_N/R_N^*}\right)
$$

for effective parameters, where $U_N$ is the compute-optimal size for the unique data and $R_N$ measures excess capacity. Substituting $N'$ and $D'$ into the Chinchilla loss law gives a data-constrained frontier. Because $R_N^*<R_D^*$ in the fit, extra compute is better spent on epochs than on parameters once unique data is scarce.

The fixed-100M-token experiment makes the consequence concrete. A one-epoch compute-optimal model is only about 7M parameters, but the lowest observed loss uses roughly 20–60× more parameters and epochs—at the cost of about 7,000× more FLOPs. The point is not that the latter is compute-efficient; it is that a strict one-epoch rule can leave signal unused when the deployment budget is already available.

### New data can be synthetic or another domain

The paper tests code filling and relaxed filtering because natural-language data is the bottleneck. Mixing Python code supplies an additional effective 2× data budget when evaluating natural-language tasks, and up to 50% code shows no downstream deterioration in the reported setting. Perplexity filtering helps on noisy corpora; deduplication helps less on the downstream metrics used here, although it can still reduce memorization.

The boundaries are important. The law uses held-out loss, while code changes the training distribution and can help code or reasoning tasks differently. Learning-rate schedules, tokenizer, corpus quality, and the fixed C4 construction are inherited from the experiment. [Chinchilla](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html) is the one-epoch/unconstrained-data limit; this paper describes what happens after that assumption fails.

## High-Level Takeaways

- Repeating a complete corpus for a few epochs can be nearly as useful as fresh data, but returns decay rapidly after that.
- Effective data and effective parameters are the right variables for a data-constrained frontier.
- Once unique data is scarce, extra compute shifts toward more epochs and smaller models than unconstrained Chinchilla allocation predicts.
- Code augmentation and selective filtering can extend the useful data budget, but their value is metric- and domain-dependent.
- Keep held-out evaluation and repetition policy explicit; subset upsampling is a different experiment from full-corpus epoching.
