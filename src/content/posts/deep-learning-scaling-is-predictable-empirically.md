---
title: 'Deep Learning Scaling is Predictable, Empirically'
date: '2017-12-01T00:00:00.000Z'
section: paper-shorts
postSlug: deep-learning-scaling-is-predictable-empirically
legacyPath: /paper shorts/2017/12/01/deep-learning-scaling-is-predictable-empirically.html
tags: [Scaling Laws, Deep Learning]
field: 'Training Systems & Reliability'
summary: '2017 – Deep Learning Scaling is Predictable, Empirically'
---

## 2017 – Deep Learning Scaling is Predictable, Empirically

**arXiv:** [1712.00409](https://arxiv.org/abs/1712.00409)

## Summary

> Hestness et al. measure learning curves across machine translation, language modeling, image classification, and speech recognition. In every domain they find a broad power-law region, but the exponents are empirical and domain-specific rather than a universal theoretical constant. The result is a planning tool with a boundary: once the curve reaches an irreducible-error regime, more of the same data and capacity is no longer the missing ingredient.

## Core Insights

### The curve is a property of a controlled family

For each task, the authors choose a strong model family, shuffle the training data, train on successive shards spanning roughly two to three orders of magnitude, and evaluate every model on the same held-out set. They search over a reduced model and optimizer space, then retain the best model at each shard size. This matters because the reported curve is not the trajectory of one frozen model; it is the best observed performance as the available data changes.

The power-law region appears in all four domains. The fitted exponents range from about $-0.07$ for word language models to about $-0.35$ for image cross-entropy; machine translation is about $-0.13$, and speech recognition about $-0.30$. Within a domain, changing the architecture or optimizer mainly shifts the intercept in this experiment. It does not justify treating one exponent as a law of learning in general.

![Cropped source Figure 1 showing NMT learning curves and their composite fit](/assets/images/deep-learning-scaling-is-predictable-empirically-source-figure-1.png)
*Fig 1: The NMT curves are fitted separately for model families and then combined into a best-fit learning curve. The image is cropped from the source Figure 1. | source: [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409)*

### Data and capacity remain coupled

The best model size grows sublinearly with data. In the paper's notation, if loss scales as $L\propto D^{-\alpha}$ and the required model size scales as $N\propto D^{\beta_p}$, then the observed $\beta_p$ lies between roughly $0.5$ and $1$. More data can therefore expose a capacity bottleneck, but not every extra token warrants a proportional increase in parameters.

The curves also have three useful regions. With very little data, a model is in a best-guess regime. In the middle, loss falls approximately as a power law. Near the floor, irreducible uncertainty and label noise dominate. A deviation from the middle region is thus diagnostic: it may indicate poor optimization or model search, or it may say that the task needs new information rather than another larger run.

### What the exponent does—and does not—buy

The authors use the curves to forecast compute, data, and model-size requirements, and to argue that small experiments can guide larger choices when they stay inside the same regime. Appendix B makes the boundary explicit: a toy counting model has an expected error lower bound with exponent $-1/2$, unlike the smaller exponents measured in real tasks. The empirical exponents absorb data structure, architecture, optimization, and the metric; they are not recovered from the toy bound.

The main limitations are equally important. The model family is selected rather than universal, hyperparameter search can change which point is called “best,” and the held-out distribution shares the task's data-generating assumptions. Use the fit locally, retain the controls, and test whether the next run is still on the same branch.

## High-Level Takeaways

- Power-law learning curves recur across modalities, but their exponents are measured properties of a task and training protocol.
- A scaling fit is most useful in the middle regime; the small-data and irreducible-error regions need different diagnoses.
- Model capacity and data should grow together, with the required capacity growing sublinearly in data in these experiments.
- Small proxy runs can guide planning only when architecture, optimizer, data distribution, and evaluation remain comparable.
- [Kaplan et al.'s language-model study](/paper%20shorts/2020/01/23/scaling-laws-for-neural-language-models.html) later specializes this empirical picture to Transformer language models.
