---
title: 'Scaling Laws for Optimal Data Mixtures'
date: '2025-07-12T00:00:00.000Z'
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

## Summary

> This paper turns data-mixture selection into a fitted prediction problem. A target loss is modeled as a function of model size $N$, training tokens $D$, and domain weights $h$; small runs then predict unseen mixtures and larger models. Experiments cover language, native multimodal, and large-vision pre-training, and use the fitted law to choose mixtures for a target domain instead of relying only on full-scale trial and error.

## Core Insights

### Fit the target, not an abstract notion of “good data”

For domains $D_1,\ldots,D_k$, the training mixture is $h\in\Delta_k$, and the model is evaluated on a target distribution that may be one of the training domains or a held-out domain. The quantity to predict is the target loss after training a model of size $N$ for $D$ tokens with weights $h$.

The additive law is

$$
L=E+\left(\sum_i C_i h_i^{\gamma_i}\right)^{-1}
+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
$$

It makes only the bias term depend on the mixture, so the optimal $h$ is independent of scale. The joint law keeps the same mixture-dependent bias but also makes $A_h$ and $B_h$ functions of $h$; its mixed derivatives with respect to scale and mixture are nonzero, so the optimal mixture can move as $N$ and $D$ change. This is the paper's central conceptual distinction: the best mixture is conditional on the target and budget, not an intrinsic ranking of datasets.

![Small-scale mixture runs extrapolated to larger models and unseen weights](/assets/images/scaling-laws-for-optimal-data-mixtures-paper-figure.png)
*Fig 1: The source figure pairs fitted small-scale loss curves with predictions at new domain weights and a larger model; the right panel compares the additive and joint optima against alternatives. | source: [Scaling Laws for Optimal Data Mixtures, Figure 1](https://arxiv.org/abs/2507.09404)*

### A few proxy runs can predict, but errors are domain-specific

The experiments fit laws on small runs and evaluate them on larger models and new weights. For LLMs, training uses SlimPajama domains with 412M–1.4B fitting models and evaluates a 7B model at 150B tokens. Native multimodal fitting uses 106M–932M models and evaluates 2.3B and 8B models at 100–160B tokens. The large-vision experiment fits up to 531M and evaluates a 1.1B model.

The mean relative error is usually low, but it is not uniform. In the validation table, the joint law gives 0.42% on multimodal interleaved data and 0.37% on text, while the LLM Wikipedia target reaches 2.09% and the vision synthetic target 5.94%. That spread is a useful warning: a good average fit does not guarantee a good optimum for every domain. The authors report that around ten mixture runs suffice for the three-domain multimodal and four-domain language cases, while six- and eight-domain language experiments need about twenty.

The fitting procedure is part of the method. Because the joint law can have up to 37 parameters for eight domains, the authors use random initializations and basin-hopping around L-BFGS, rather than repeatedly starting L-BFGS on a fixed grid. In Figure 2, basin-hopping reaches a low Huber objective with fewer L-BFGS calls than the repeated-start baseline; the shaded bands show the 25–75% quantiles over 100 trials.

![Basin-hopping versus repeated L-BFGS fitting for the joint mixture law](/assets/images/scaling-laws-for-optimal-data-mixtures-source-figure-2.webp)
*Fig 2: Huber fitting loss versus L-BFGS calls on the three-domain multimodal experiment; basin-hopping converges faster across the repeated trials. | source: [Scaling Laws for Optimal Data Mixtures, Figure 2](https://arxiv.org/abs/2507.09404)*

### Optimize mixture weights after deciding what success means

For multimodal pre-training, the three domains are text-only, interleaved documents, and paired image-caption data. The joint law predicts an optimum that changes with compute: as token budget grows, interleaved data becomes less important, while larger models rely more on text. The source simplex plot shows optimal weights across model sizes and token counts, with the additive law marked by red crosses; it is a visual reminder that the optimum is a trajectory through the mixture simplex, not one fixed recipe.

![Optimal domain weights across multimodal compute budgets](/assets/images/scaling-laws-for-optimal-data-mixtures-source-figure-5.webp)
*Fig 3: The simplex places text, image-caption, and interleaved domains at its vertices; markers show optimal weights across model sizes and token counts, while color encodes training tokens. | source: [Scaling Laws for Optimal Data Mixtures, Figure 5](https://arxiv.org/abs/2507.09404)*

The end-to-end tests are persuasive because the target changes. For the 7B language models, the mixture optimized for OpenHermes reaches CORE 58, compared with 56 for the average-training-domain optimum, 53 for uniform weights, and 52 for the base SlimPajama distribution. For native multimodal and large-vision models, optimized mixtures also beat uniform and prior-work mixtures on their reported validation losses. The method does not claim that one mixture wins everywhere; it gives a way to specify the target first and solve for the weights.

## High-Level Takeaways

- Data mixtures should be optimized against a target loss and compute budget, not treated as universal dataset quality scores.
- The joint law is the useful extension when mixture choice should change with scale; the additive law remains a strong, simpler baseline.
- Small proxy runs can save full-scale trials, but validation errors vary substantially by domain and target.
- The method assumes fixed mixtures, no scarcity-driven repetition, and a pretraining loss proxy; dynamic curricula and downstream utility remain open.
