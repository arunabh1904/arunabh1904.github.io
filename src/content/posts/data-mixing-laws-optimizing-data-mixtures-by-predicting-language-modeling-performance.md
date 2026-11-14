---
title: 'Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance'
date: '2024-03-25T00:00:00.000Z'
section: paper-shorts
postSlug: data-mixing-laws-optimizing-data-mixtures-by-predicting-language-modeling-performance
legacyPath: /paper shorts/2024/03/25/data-mixing-laws-optimizing-data-mixtures-by-predicting-language-modeling-performance.html
tags: [Language Models, Data Mixtures, Scaling Laws]
field: 'Multimodal Scaling & Data Mixtures'
summary: '2024 – Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance'
---

## 2024 – Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance

**arXiv:** [2403.16952](https://arxiv.org/abs/2403.16952)

## Summary

> Ye et al. make the data mixture itself a fitted variable. For each validation domain, an exponential function of the training proportions predicts losses on unseen mixtures; nested scaling laws then move those predictions from small models and short runs to a target scale. A 1B RedPajama experiment reaches the default mixture's loss with 0.73× as many steps and matches a roughly 1.48× longer default run, while the continual-pretraining experiment predicts how much Python can be added without sacrificing Pile performance.

## Core Insights

### Mixtures have measurable interactions

For training proportions $r_1,\ldots,r_M$, the adopted domain-loss law is

$$
L_i(r_1,\ldots,r_M)=c_i+k_i\exp\left(\sum_j t_{ij}r_j\right).
$$

The constant $c_i$ represents loss not reducible by reweighting, while $t_{ij}$ records how training domain $j$ affects validation domain $i$. A negative coefficient means facilitation; a positive coefficient means conflict. On five coarse Pile domains, most interactions are sparse, with examples of dialogue helping internet text and symbolic data hurting prose. This is more informative than ranking domains by their standalone loss because the value of a domain is relational.

![Source Figure 1 showing the nested scaling-law pipeline for data mixtures](/assets/images/data-mixing-laws-source-figure-1.png)
*Fig 1: Small runs first extrapolate training steps and model size, then fit the data-mixing law over unseen proportions. | source: [Data Mixing Laws, Figure 1](https://arxiv.org/abs/2403.16952)*

### Fit a target scale before optimizing the mixture

Directly fitting the mixture law at the target model size is expensive. The authors instead train 70M–410M models for 30B tokens on selected mixtures, fit a training-step law for each size, extrapolate to the target token count, fit a size law, and only then fit the mixture function. They choose mixtures from the simplex so the small runs cover its boundaries, and validate on 40 candidate mixtures not all used for fitting.

The target experiment trains a 1B model for 100B tokens on RedPajama. The optimized mixture reaches the default mixture's validation loss after 73% of the steps and eventually reaches a loss comparable to a default-mixture model trained for 1.48× as many tokens. It beats the compared DoReMi and DoGE mixtures in the same 1B/100B setting. The result is a prediction-and-search pipeline, not proof that the same weights are optimal at another scale.

### The same law can protect an old capability

In continual pretraining, Pythia-70M is trained on mixtures of the Pile and Python code. Fitting the two validation losses predicts the critical Python proportion that preserves the original Pile loss while improving Python. This reframes catastrophic forgetting as a constrained mixture problem: the goal is not “add as much new data as possible,” but find the frontier between target improvement and retained competence.

The paper's own caveat is central. Other work finds that mixture rankings can change with model size and training tokens, so the small-scale optimum can be suboptimal at the target scale. The law also assumes enough representative mixture samples and a validation distribution that can be modeled explicitly or by learned implicit domains. It is strongest as an experiment-design tool: spend a small budget to measure the local surface before committing to a large recipe.

## High-Level Takeaways

- Data mixtures have interactions; a domain's value depends on the validation target and the other domains present.
- Nested step, size, and mixture fits can turn small runs into a target-scale search, but each extrapolation adds a boundary.
- The RedPajama result shows meaningful compute savings in one controlled target; it does not establish scale-invariant weights.
- Continual pretraining can be posed as a loss-preservation constraint on the mixture, making the data schedule measurable.
- [DoReMi](/paper%20shorts/2023/05/17/doremi-optimizing-data-mixtures-speeds-up-language-model-pretraining.html) optimizes a worst-domain excess-loss objective relative to a reference model; this paper instead fits the target validation mixture explicitly.
