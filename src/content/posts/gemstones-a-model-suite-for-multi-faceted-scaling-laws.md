---
title: 'Gemstones: A Model Suite for Multi-Faceted Scaling Laws'
date: '2025-02-07T00:00:00.000Z'
section: paper-shorts
postSlug: gemstones-a-model-suite-for-multi-faceted-scaling-laws
legacyPath: /paper shorts/2025/02/07/gemstones-a-model-suite-for-multi-faceted-scaling-laws.html
tags: [Language Models, Scaling Laws, Experiment Design]
field: 'Language Models'
summary: '2025 – Gemstones: A Model Suite for Multi-Faceted Scaling Laws'
---

## 2025 – Gemstones: A Model Suite for Multi-Faceted Scaling Laws

**arXiv:** [2502.06857](https://arxiv.org/abs/2502.06857) · **Conference:** NeurIPS 2025

## Summary

> Gemstones releases over 4,000 checkpoints from 22 Transformer shapes, 50M–2B parameters, and more than 10 trillion cumulative tokens. Its central finding is experimental-design sensitivity: the fitted compute frontier changes when one varies width/depth, cooldown, embeddings, token sampling, or the fitting procedure. FLOPs can favor a deep model while wall-clock time strongly favors a wide one, so a scaling prescription must name the resource it optimizes.

## Core Insights

### A parameter count hides model shape

The suite samples width and depth at fixed parameter targets from 50M to 2B, trains each model on Dolma 1.7 for 350B tokens, and saves checkpoints every 2B tokens. Learning rate is transferred across shapes with a width–depth rule, while separate runs ablate cooldown and halve the learning rate. The authors also build a FLOP counter because the usual $6ND$ approximation does not treat extremely wide and narrow models equally.

They fit compute-optimal loss using the Chinchilla-style parametric law and a lower-convex-hull approach. The latter matters because a sparse two-dimensional shape sweep contains many models that are never optimal; putting every point into IsoFLOP bins can fit the suboptimal interior instead of the attainable envelope.

![Gemstones source Figure 17: FLOP and GPU-hour overspend by architecture](/assets/images/gemstones-model-suite-scaling-source-figure-17.png)
*Fig 1: The same architectural mistake has different costs: the left panel measures excess FLOPs, while the right measures excess GPU hours. This is source Figure 17. | source: [Gemstones, Figure 17](https://arxiv.org/abs/2502.06857)*

### The fitted slope is a design artifact unless the design is reported

On their full hot-training set, Approach 1 without embeddings gives a model-size/compute slope of about 0.569. Using only data at or below 100B tokens raises it to 0.627; using only points above 120B raises it to 0.967; cooldown moves it to 0.724. Approach 3 is more stable but still shifts from about 0.714 to 0.693–0.735 under comparable resampling. These are not competing “true exponents”; they are the output of different observation sets and frontiers.

The convex hull is designed to make that dependence explicit. In the 100B-token slice, the authors' hull is less sensitive to the suboptimal points than Chinchilla-style binning. Sampling every 2B versus every 10B tokens also matters more for Approach 1 than Approach 3. A law estimated from sparse or late checkpoints should not be compared to a law estimated from the full trajectory without aligning the fitting protocol.

### FLOPs and time are different objectives

The deeper Gemstones often reach lower loss or benchmark error at equal FLOPs, but typical open-source parallelism makes them much slower. In the source Figure 17, the least efficient shapes overspend only about 50% FLOPs yet more than 200% GPU hours. If a training team values calendar time or accelerator rental, the FLOP-optimal model can be the wrong choice. The same distinction appears in inference when memory movement, communication, or latency dominates arithmetic.

The suite's limitations are useful context: all models use one corpus and one optimizer family, the largest is 2B, and extreme shapes required rollbacks or minor learning-rate adjustments. The open checkpoints make those boundaries inspectable, but they do not establish a universal width/depth law.

## High-Level Takeaways

- Scaling laws are conditional on the model family, shape distribution, checkpoint sampling, and fit procedure.
- A convex-hull fit is safer than binning when the explored architecture space has many suboptimal interior points.
- FLOP-optimal and time-optimal architectures can diverge dramatically; report the resource that the law optimizes.
- Width, depth, embeddings, cooldown, and learning rate should be treated as experimental axes, not hidden constants.
- [The Hitchhiker's Guide](/paper%20shorts/2024/10/15/a-hitchhikers-guide-to-scaling-law-estimation.html) supplies the complementary question: how many and which checkpoints are enough to estimate the law.
