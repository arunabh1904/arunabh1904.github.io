---
title: 'Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains'
date: '2020-06-18T00:00:00.000Z'
section: paper-shorts
postSlug: fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains
legacyPath: /paper shorts/2020/06/18/fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains.html
tags:
  - Other
field: 'Vision Foundations'
summary: "2020 – Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
---
## 2020 – Fourier Features

**arXiv:** [2006.10739](https://arxiv.org/abs/2006.10739)

**Project:** [Fourier Feature Networks](https://bmild.github.io/fourfeat/)

**Code:** [tancik/fourier-feature-networks](https://github.com/tancik/fourier-feature-networks)

### Method and reported result

A plain MLP has a spectral bias: it learns smooth, low-frequency functions much more easily than sharp boundaries or fine texture. This paper fixes that failure mode by mapping each coordinate through sinusoidal Fourier features before the MLP sees it.

## Summary

> That small input change matters for BEV and mapping work because many spatial tasks are low-dimensional coordinate regression problems with high-frequency structure. Lane boundaries, occupancy edges, radiance fields, and implicit maps all ask a network to represent sharp geometry from coordinates.

## Core Insights

The paper studies why coordinate MLPs struggle when the target function contains high-frequency detail. The method samples or chooses Fourier bases, maps an input coordinate through sine and cosine features, and feeds the expanded representation into a standard MLP. Using neural tangent kernel analysis, the authors show that this transformation changes the effective kernel into a stationary kernel whose bandwidth can be tuned by the Fourier feature scale.

The empirical evidence comes from image regression and low-dimensional vision and graphics tasks. The important caveat is that the frequency scale is a real modeling choice. Too little bandwidth leaves the MLP smooth; too much bandwidth can overfit or make optimization brittle.

![Figure 2 from Fourier Features showing how Fourier mappings change the MLP neural tangent kernel](/assets/images/fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains-paper-figure.png)
*Figure 2 shows why the input mapping matters: Fourier features reshape the effective NTK into a frequency-aware kernel. From the [Fourier Features paper](https://arxiv.org/abs/2006.10739), via ar5iv. source: [Fourier Features paper](https://arxiv.org/abs/2006.10739)*

![Figure 4 from Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](/assets/images/fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains-source-figure-4.webp)
*Figure 4 We find that a sparse random sampling of Fourier features can perform as well as a dense set of features and that the width of the distribution matters more than the shape. Here, we generate random 1D signals from noise and report the test-set accuracy of different trained models that use a sparse set (16 out of 1024) of random Fourier features sampled from different distributions. Each subplot represents a different family of 1D signals. source: [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)*

![Figure 1 from Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](/assets/images/fourier-features-let-networks-learn-high-frequency-functions-in-low-dimensional-domains-source-figure-1.webp)
*Figure 1 Fourier features improve the results of coordinate-based MLPs for a variety of high-frequency low-dimensional regression tasks, both with direct (b, c) and indirect (d, e) supervision. We visualize an example MLP (a) for an image regression task (b), where the input to the network is a pixel coordinate and the output is that pixel’s color. source: [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)*


**What to look at:**
- The contribution is an input representation, not a new network family.
- Fourier features let a coordinate network expose high frequencies early instead of asking hidden layers to discover them slowly.
- The bandwidth parameter controls the smoothness/detail tradeoff.

### Reported evidence

| Signal | Detail | Why it matters |
| ------ | ------ | -------------- |
| Core method | Sinusoidal coordinate mapping before an MLP | A small representation change recovers high-frequency regression. |
| Theory | NTK analysis of the transformed MLP | Explains the frequency bias instead of only showing examples. |
| Tasks | Low-dimensional vision and graphics regression | Matches the coordinate-heavy structure of implicit scene and map models. |

**Compact result slice:**

| Input mapping | 2D natural image PSNR | 3D shape IoU |
| ------------- | --------------------- | ------------ |
| No mapping | 19.32 | 0.864 |
| Gaussian Fourier features | 25.57 | 0.973 |

## High-Level Takeaways

- Fourier features inform whether an MLP should learn geometry directly from raw coordinates or receive a fixed sinusoidal basis that exposes high spatial frequencies. The atomic unit is a coordinate-value pair; a bandwidth-controlled projection maps the coordinate into periodic features before the shared MLP.
- The experiments show that the embedding changes the effective kernel and overcomes spectral bias on images and 3D signals. They do not prescribe one bandwidth for noisy, multiscale driving geometry. The missing test jointly sweeps bandwidth, learned encodings, and coordinate noise at matched parameters. At 10× spatial extent or resolution, aliasing and basis size become limiting. The claim would weaken if a learned positional encoding matched high-frequency reconstruction while adapting more robustly across scales.
- Fourier features became one of the standard ways to make coordinate networks useful for detailed spatial signals.
- If an MLP is asked to learn geometry from raw coordinates, give it a frequency basis first; otherwise the model starts with the wrong smoothness prior.
