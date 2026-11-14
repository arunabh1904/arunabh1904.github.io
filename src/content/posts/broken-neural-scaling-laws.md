---
title: 'Broken Neural Scaling Laws'
date: '2022-10-26T00:00:00.000Z'
section: paper-shorts
postSlug: broken-neural-scaling-laws
legacyPath: /paper shorts/2022/10/26/broken-neural-scaling-laws.html
tags: [Scaling Laws, Deep Learning, Evaluation]
field: 'Training Systems & Reliability'
summary: '2022 – Broken Neural Scaling Laws'
---

## 2022 – Broken Neural Scaling Laws

**arXiv:** [2210.14891](https://arxiv.org/abs/2210.14891) · **Conference:** ICLR 2023

## Summary

> Caballero et al. replace the default monotonic power law with a smoothly broken power law (BNSL). A small number of connected power-law segments can express plateaus, inflection points, and even non-monotonic behavior such as double descent. The broad benchmark is a useful warning: a good fit on the observed range is not enough; sharp breaks require measurements near the transition before they become predictable.

## Core Insights

### A power law is a special case, not a default truth

The BNSL writes a metric $y$ as an offset plus a product of smoothly activated power-law factors:

$$
y=a+(bx^{-c_0})\prod_i\left(1+\left(\frac{x}{d_i}\right)^{1/f_i}\right)^{-c_i f_i}.
$$

With no breaks, this reduces to an ordinary power law. Each $d_i$ locates a transition, $c_i$ changes the slope, and $f_i$ controls sharpness. On a log-log plot, the function is a sequence of approximately linear segments joined continuously. This lets the model represent an upstream loss curve and a downstream metric without forcing both to be monotonic.

![Source Figure 1 illustrating a smoothly broken power law](/assets/images/broken-neural-scaling-laws-source-figure-1.png)
*Fig 1: BNSL joins several power-law segments at smooth or sharp breaks. The source figure makes the model's extra degrees of freedom visible before the empirical fits. | source: [Broken Neural Scaling Laws, Figure 1](https://arxiv.org/abs/2210.14891)*

### Extrapolation is tested, not assumed

The authors fit only black points and reserve green points at larger scales for extrapolation. Across the large-scale vision benchmark, BNSL is the best extrapolator for 69.44% of tasks; for language, it is best for 75%. The evaluation spans upstream and downstream vision/language, audio, video, diffusion, robotics, retrieval, quantization, RL, and other settings. This diversity demonstrates expressivity, but it also means the benchmark mixes metrics, data sources, and noise levels rather than estimating one shared law.

The most diagnostic examples are not the average wins. BNSL models double descent in Transformer translation and a delayed inflection in four-digit addition. The fitting code uses a grid search followed by nonlinear least squares, chooses the number of breaks with held-out points, and scores extrapolation with log errors. Those choices are part of the result: a flexible function can fit many shapes, so the held-out tail carries the evidence.

### Sharp transitions have an information limit

The arithmetic experiment shows the boundary clearly. A break appears near dataset size 415, but accurate extrapolation requires at least one training point around or beyond the break (about 415 in a noiseless simulation and 720 in the noisy runs). Points far before a sharp transition cannot identify its location or post-break slope, even with infinitely many seeds. Conversely, points far below a break can become irrelevant once enough post-break points are available.

This is the paper's most useful operational claim. “Emergence” is not automatically unpredictable, but a hidden sharp break is not recoverable from a distant pre-break fit. The authors' appendix repeats this test for many domains, including sparse models and multimodal evaluation; those examples strengthen the diagnostic, not a claim that every metric follows BNSL.

## High-Level Takeaways

- Use a broken law when the metric can plateau, bend, or reverse; a monotonic power law cannot express those behaviors.
- Held-out larger scales are the meaningful test of extrapolation, especially for a flexible fit.
- Sharp breaks require measurements near the break; no curve-fitting method can infer an unobserved transition reliably.
- The benchmark supports BNSL's expressivity across many tasks, not one universal neural scaling function.
- [Hestness et al.](/paper%20shorts/2017/12/01/deep-learning-scaling-is-predictable-empirically.html) remains the useful baseline for the broad power-law regime this paper extends.
