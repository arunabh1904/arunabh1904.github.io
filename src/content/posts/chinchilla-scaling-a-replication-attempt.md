---
title: 'Chinchilla Scaling: A Replication Attempt'
date: '2024-04-15T00:00:00.000Z'
section: paper-shorts
postSlug: chinchilla-scaling-a-replication-attempt
legacyPath: /paper shorts/2024/04/15/chinchilla-scaling-a-replication-attempt.html
tags: [Language Models, Scaling Laws, Reproducibility]
field: 'Training Systems & Reliability'
summary: '2024 – Chinchilla Scaling: A Replication Attempt'
---

## 2024 – Chinchilla Scaling: A Replication Attempt

**arXiv:** [2404.10102](https://arxiv.org/abs/2404.10102)

## Summary

> Besiroglu et al. reconstruct the data behind Chinchilla's parametric fit and rerun its third estimation procedure. Their best fit agrees with Chinchilla's other two methods and the 20-tokens-per-parameter model, while the published Approach 3 coefficients fit the reconstruction poorly and report implausibly narrow confidence intervals. The lesson is methodological: scaling exponents are only useful when the data extraction, optimizer convergence, parameter precision, and uncertainty procedure are reproducible.

## Core Insights

### Reproduce the data before debating the law

The authors digitize the scatter points from Chinchilla's Figure 4: model size, training FLOPs, and loss encoded by position and color. Their reconstructed set contains 240 usable observations, with five very high-loss points at a token/parameter ratio below 0.4 treated as outliers in the main fit. Digitizing the plot introduces uncertainty: the y-axis has no tick marks and the color scale has only 256 values, limiting loss precision to roughly 0.01.

![Iso-loss contours and reconstructed Chinchilla data](/assets/images/chinchilla-scaling-replication-source-figure-4.png)
*Fig 1: Reconstructed observations sit on the loss contours and efficient frontier used in the replication. This source image reproduces the Chinchilla Figure 4 contour view. | source: [Chinchilla Scaling: A Replication Attempt](https://arxiv.org/abs/2404.10102)*

### The optimizer and rounding change the extrapolation

They fit the same parametric form,

$$
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta},
$$

with a Huber objective in log space. Their estimate is approximately $E=1.8172$, $\alpha=0.3478$, and $\beta=0.3658$, implying a compute-optimal model-size exponent $a=\beta/(\alpha+\beta)\approx0.513$. The unrounded Chinchilla source parameters imply $a\approx0.454$; the rounded body-text values fit even worse. The authors report that averaging rather than summing the Huber loss gave the original optimizer a poor loss scale and caused early termination.

This is not a claim that one digit caused the entire disagreement. Rounding $\beta$ from about 0.2849 to 0.28 introduces a roughly 13% relative bias in the finite-data term at $10^{11}$ tokens, but the unrounded published parameters still have a significantly worse likelihood than the converged replication. The fitted policy from the replication is close to the 20 tokens per parameter used by Chinchilla; the rounded source fit suggests about 70, which is inconsistent with the model it was meant to explain.

### Uncertainty is part of the prescription

The reported confidence interval for the compute-allocation exponent is roughly 0.454–0.455. Bootstrapping the reconstructed data gives a standard error near 0.018, implying a much wider 80% interval. The paper estimates that a confidence interval as tight as 0.001 would require on the order of 600,000 comparable runs, not the roughly 400 observations available. These numbers depend on the reconstruction and bootstrap model, but they make the qualitative point: precision in a fitted exponent is not evidence of precision in the underlying law.

The appendix repeats the fit without dropping the five outliers and reaches the same broad conclusion, with larger uncertainty. The remaining limitations are straightforward: the source data are digitized, the original full dataset is unavailable, and the fit still assumes the Chinchilla functional form. Treat this note as a reproducibility audit of Approach 3, not a replacement for a new large controlled scaling sweep.

## High-Level Takeaways

- Scaling-law replication starts with recovering the actual observations and their measurement error.
- Optimizer convergence and parameter rounding can materially change a compute-optimal extrapolation.
- Confidence intervals must reflect the number and structure of runs; extra significant figures do not create information.
- The replication supports Chinchilla's balanced allocation through the other two methods, while diagnosing Approach 3's original fit.
- [Chinchilla's original note](/paper%20shorts/2022/03/29/training-compute-optimal-large-language-models.html) remains the source of the 70B/1.4T experiment; this paper tests its parametric evidence.
